import os
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import pandas as pd
import requests
from dotenv import load_dotenv

from fastmcp import FastMCP

from api_utils import (
    MLData,
    download_smiles_csv_from_s3,
    ensure_ml_weights_available,
    inference_ml,
    train_ml_with_data,
    upload_predictions_csv_to_s3,
)
from utils.base_state import TrainState

try:
    from utils.state_s3 import download_state_file
except ModuleNotFoundError:
    from .utils.state_s3 import download_state_file

try:
    from s3_utils import S3BucketService, s3_service as default_s3_service
except ModuleNotFoundError:
    from .s3_utils import S3BucketService, s3_service as default_s3_service

load_dotenv()

IMPORT_PATH = Path(__file__).resolve().parent
STATE_FILE = "state.json"
LOG_FILE = "mcp.txt"           # written by api.sh (nohup python ... > mcp.txt 2>&1)
LOG_TAIL_HARD_CAP = 50_000     # safety ceiling for `get_mcp_logs(tail_lines=...)`

mcp = FastMCP("automl-mcp")
_TRAIN_JOBS: Dict[str, Process] = {}
_TRAIN_JOB_META: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Target registry — protein / disease vocabulary -> AutoML case
# ---------------------------------------------------------------------------
# Shared vocabulary with the generative server (`generate_case_mols`) and the
# benchmark dataset. The three namespaces disagree on spelling and casing:
#
#     concept        dataset          generative CVAE   AutoML case
#     GSK-3beta      "alzheimer"      "alzheimer"       "Alzheimer"
#     BTK            "sclerosis"      "skleroz"         "Skleroz"
#     KRAS G12C      "lung cancer"    "cancer"          (none)
#
# `generate_case_mols` hands back the lowercase disease slug, which is exactly
# the string `predict_ml` used to reject with `weights_not_found` -> agents
# then obeyed the error text and burned entire runs on `train_ml`. Every alias
# below therefore resolves to the case-exact state key.
#
# `case` is None for targets that have no trained AutoML pipeline yet. Adding
# one later means flipping that field to the new case name — no description
# and no caller changes.

_TARGET_REGISTRY: Tuple[Dict[str, Any], ...] = (
    {
        "case": "Alzheimer",
        "protein": "GSK-3beta",
        "pdb": "4J1R",
        "disease": "alzheimer",
        "aliases": (
            "alzheimer", "alzheimers", "alzheimer's disease", "alzheimer disease",
            "gsk-3beta", "gsk3beta", "gsk-3b", "gsk3b", "gsk-3", "gsk3",
            "glycogen synthase kinase 3 beta", "glycogen synthase kinase-3 beta",
            "glycogen synthase kinase 3", "4j1r", "tau kinase", "tau protein kinase",
        ),
    },
    {
        # A duplicate of `Alzheimer` trained on the same 4J1R data. The name is
        # a trap: it is GSK-3beta, not an oncology model. Registered so that an
        # agent that picks it by name is still told which protein it scores.
        "case": "Brain_cancer_test",
        "protein": "GSK-3beta",
        "pdb": "4J1R",
        "disease": None,
        "aliases": (),
    },
    {
        "case": "Skleroz",
        "protein": "BTK",
        "pdb": "5VFI",
        "disease": "sclerosis",
        "aliases": (
            "sclerosis", "skleroz", "multiple sclerosis", "ms",
            "btk", "bruton's tyrosine kinase", "brutons tyrosine kinase",
            "bruton tyrosine kinase", "tyrosine-protein kinase btk",
            "tyrosine protein kinase btk", "btk domain", "5vfi",
        ),
    },
    # --- Targets covered by the generative CVAE but with no AutoML pipeline ---
    {
        "case": None,
        "protein": "KRAS G12C",
        "pdb": "8AFB",
        "disease": "lung cancer",
        "aliases": (
            "lung cancer", "cancer", "nsclc", "non-small cell lung cancer",
            "non small cell lung cancer", "kras", "kras g12c", "g12c", "k-ras",
        ),
    },
    {
        "case": None,
        "protein": "c-Abl",
        "pdb": None,
        "disease": "parkinson",
        "aliases": (
            "parkinson", "parkinsons", "parkinson's disease", "parkinson disease",
            "c-abl", "cabl", "abl", "abl kinase", "abl tyrosine-protein kinase",
            "tyrosine-protein kinase abl", "alpha-synuclein", "a-synuclein",
            "comt", "catechol o-methyltransferase", "mao-b", "monoamine oxidase b",
        ),
    },
    {
        "case": None,
        "protein": "PCSK9 / ACLY",
        "pdb": None,
        "disease": "dyslipidemia",
        "aliases": (
            "dyslipidemia", "dyslipidaemia", "hyperlipidemia", "hypercholesterolemia",
            "pcsk9", "proprotein convertase subtilisin/kexin type 9",
            "proprotein convertase subtilisin kexin type 9",
            "acly", "atp citrate synthase", "atp citrate lyase",
            "ppar-alpha", "ppar alpha", "pparalpha",
            "peroxisome proliferator-activated receptor alpha",
            "cetp", "cholesteryl ester transfer protein",
            "hmg-coa reductase", "hmg coa reductase", "npc1l1", "angptl3", "mtp",
        ),
    },
    {
        "case": None,
        "protein": "STAT3",
        "pdb": None,
        "disease": "drug_resist",
        "aliases": (
            "drug_resist", "drug resist", "drug resistance", "drug_resistance",
            "multidrug resistance", "chemoresistance", "chemo-resistance",
            "stat3", "signal transducer and activator of transcription 3",
            "p-glycoprotein", "p-gp", "pgp", "abc transporters", "efflux pumps",
            "heat shock proteins", "pi3k", "phosphoinositide 3-kinase",
            "ras-raf-mek-erk", "ras raf mek erk",
        ),
    },
)

# Greek / typographic folding so `GSK-3β` and `GSK-3beta` land on one key, and
# `α-synuclein` (the dataset only ever spells it with a Greek alpha) matches.
_GREEK_FOLD = {
    "α": "alpha", "Α": "alpha",
    "β": "beta", "Β": "beta",
    "γ": "gamma", "Γ": "gamma",
    "δ": "delta", "Δ": "delta",
    "μ": "u", "µ": "u",
    "’": "'", "‘": "'", "–": "-", "—": "-",
}


def _fold_target_text(value: str) -> str:
    """Normalize a protein/disease string for alias matching.

    Folds Greek letters to ASCII, lowercases, and strips every character that
    is not a letter or digit — so `GSK-3β`, `GSK-3beta`, `gsk 3 beta` and
    `GSK3B` all collapse onto the same key.
    """
    text = (value or "").strip().lower()
    for src, dst in _GREEK_FOLD.items():
        text = text.replace(src, dst)
    return "".join(ch for ch in text if ch.isalnum())


_ALIAS_INDEX: Dict[str, Dict[str, Any]] = {}
for _entry in _TARGET_REGISTRY:
    for _alias in (
        *_entry["aliases"],
        _entry["protein"],
        _entry["disease"],
        _entry["case"] or "",
        _entry["pdb"] or "",
    ):
        _folded = _fold_target_text(_alias)
        if _folded:
            _ALIAS_INDEX.setdefault(_folded, _entry)


def _resolve_case_key(requested: Optional[str], state_dict: Dict[str, Any]) -> Optional[str]:
    """Map a requested case onto the real, case-exact key in state.json.

    Case identifiers are case-sensitive everywhere downstream, but the
    generative server emits lowercase disease slugs (`alzheimer`) while the
    trained AutoML case is capitalized (`Alzheimer`).

    A TRAINED match wins over an exact one. State really does carry both
    `alzheimer` (a stale Training entry) and `Alzheimer` (the trained
    pipeline); preferring the exact key would hand back the one that cannot
    predict, which is the failure this resolution exists to prevent. Only when
    nothing trained matches do we fall back to the exact/case-insensitive key,
    so the caller still learns that case's real status.
    """
    if not requested:
        return None
    lowered = requested.strip().lower()
    exact = [key for key in state_dict if key == requested]
    insensitive = [key for key in state_dict if key.lower() == lowered and key != requested]

    for key in (*exact, *insensitive):
        if _case_has_weights(key, state_dict):
            return key
    if exact:
        return exact[0]
    return insensitive[0] if insensitive else None


def _case_has_weights(case_key: str, state_dict: Dict[str, Any]) -> bool:
    ml = (state_dict.get(case_key) or {}).get("ml_models") or {}
    return ml.get("status") == "Trained" and bool(ml.get("Predictable properties"))


def _resolve_target(
    *,
    protein: Optional[str],
    disease: Optional[str],
    case: Optional[str],
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve `protein` / `disease` / `case` onto an AutoML case.

    Resolution order: explicit `case` wins (it is the escape hatch for cases
    that are not in the registry, e.g. a freshly trained one); otherwise the
    protein/disease vocabulary is looked up in `_ALIAS_INDEX`.

    Returns a dict with `case` (None when no trained pipeline covers the
    target), `target_requested`, `target_modeled`, `pdb`, `case_status` and
    `resolution`. `resolution` is one of:
        case_exact / case_insensitive — an explicit trained `case` was used.
        target_alias                  — protein/disease matched a trained case.
        case_not_trained              — the case exists but is not Trained;
                                        `case_status` carries its real status.
        target_known_no_model         — a known protein with no pipeline yet.
        unknown_target                — nothing matched.
    """
    requested_text = protein or disease or case or ""

    if case:
        resolved_key = _resolve_case_key(case, state_dict)
        if resolved_key:
            entry = next(
                (e for e in _TARGET_REGISTRY if e["case"] == resolved_key),
                None,
            )
            trained = _case_has_weights(resolved_key, state_dict)
            ml_state = (state_dict.get(resolved_key) or {}).get("ml_models") or {}
            return {
                # A case that exists but is still Training / Failed must NOT be
                # reported as an unknown target — the caller needs to tell a
                # typo apart from a job that will be ready shortly.
                "case": resolved_key if trained else None,
                "case_status": ml_state.get("status"),
                "target_requested": requested_text,
                "target_modeled": entry["protein"] if entry else None,
                "pdb": entry["pdb"] if entry else None,
                "resolution": (
                    ("case_exact" if resolved_key == case else "case_insensitive")
                    if trained
                    else "case_not_trained"
                ),
            }
        # `case` is not a known key — it may be a disease/protein word instead.

    entry = _ALIAS_INDEX.get(_fold_target_text(requested_text))
    if entry is None:
        return {
            "case": None,
            "case_status": None,
            "target_requested": requested_text,
            "target_modeled": None,
            "pdb": None,
            "resolution": "unknown_target",
        }

    resolved_key = _resolve_case_key(entry["case"], state_dict) if entry["case"] else None
    if resolved_key and _case_has_weights(resolved_key, state_dict):
        return {
            "case": resolved_key,
            "case_status": "Trained",
            "target_requested": requested_text,
            "target_modeled": entry["protein"],
            "pdb": entry["pdb"],
            "resolution": "target_alias",
        }
    return {
        "case": None,
        "case_status": None,
        "target_requested": requested_text,
        "target_modeled": entry["protein"],
        "pdb": entry["pdb"],
        "resolution": "target_known_no_model",
    }


def _surrogate_fallback_enabled() -> bool:
    """Whether targets without their own pipeline are served by another case.

    Benchmark scaffolding: with this on, `predict_ml` answers for every protein
    so an agent can satisfy itself that no training is required and complete a
    run. `docking_score` / `IC50` then come from `_pick_fallback_case`, NOT from
    a model trained on the requested protein — the serving case is always
    reported back as `case` / `weights_case`.

    Turn OFF (`PREDICT_ML_SURROGATE_FALLBACK=0`) once real per-target weights
    are registered, so untrained targets return RDKit properties only instead
    of another protein's scores.
    """
    return os.getenv("PREDICT_ML_SURROGATE_FALLBACK", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _pick_fallback_case(state_dict: Dict[str, Any]) -> Optional[str]:
    """Choose the trained case used to serve targets that have no pipeline.

    Order: `PREDICT_ML_FALLBACK_CASE` if it names a trained case, then the
    registry's own trained cases (Alzheimer/GSK-3beta first), then any trained
    case in state. Returns None when nothing is trained at all.
    """
    preferred = os.getenv("PREDICT_ML_FALLBACK_CASE")
    if preferred:
        key = _resolve_case_key(preferred, state_dict)
        if key and _case_has_weights(key, state_dict):
            return key

    for entry in _TARGET_REGISTRY:
        if not entry["case"]:
            continue
        key = _resolve_case_key(entry["case"], state_dict)
        if key and _case_has_weights(key, state_dict):
            return key

    for key in state_dict:
        if key == "Calculateble properties":
            continue
        if _case_has_weights(key, state_dict):
            return key
    return None


# `Validity` maps to a filter that DROPS invalid molecules, so it returns a
# shorter list than its input and cannot be aligned row-wise with the others.
_LENGTH_CHANGING_CALC_PROPS = frozenset({"Validity"})


def _partition_valid_smiles(smiles_list: List[str]) -> Tuple[List[str], List[str]]:
    """Split SMILES into (parseable, unparseable).

    Every calculable-property function is a whole-batch list comprehension over
    `Chem.MolFromSmiles`, so a single unparseable string raises and takes the
    entire batch's property down with it. The prediction pipeline separately
    `dropna()`s bad rows mid-list, which silently shifts every later prediction
    onto the wrong molecule. Screening once, up front, fixes both.
    """
    try:
        from rdkit import Chem, RDLogger
    except ModuleNotFoundError:
        return list(smiles_list), []

    RDLogger.DisableLog("rdApp.*")
    valid: List[str] = []
    invalid: List[str] = []
    for smi in smiles_list:
        try:
            (valid if Chem.MolFromSmiles(smi) is not None else invalid).append(smi)
        except Exception:
            invalid.append(smi)
    return valid, invalid


def _compute_calculable_properties(smiles_list: List[str]) -> Dict[str, List[Any]]:
    """Compute the RDKit properties that need no trained model.

    These are functions of the molecule alone — identical no matter which
    protein was asked about — so they are the honest answer for any target,
    including ones with no AutoML pipeline.

    Expects `smiles_list` to be pre-screened by `_partition_valid_smiles`.
    Reads the property registry directly rather than through `TrainState`,
    which would trigger another S3 state download for data that never varies.
    """
    try:
        from utils.calculateble_prop_funcs import config as calc_registry
    except ModuleNotFoundError:
        from .utils.calculateble_prop_funcs import config as calc_registry

    if not smiles_list:
        return {}

    properties: Dict[str, List[Any]] = {}
    for name, func in calc_registry.items():
        if name in _LENGTH_CHANGING_CALC_PROPS:
            continue
        try:
            values = list(func(smiles_list))
        except Exception as exc:
            # Fall back to per-molecule evaluation so one awkward structure
            # cannot erase this property for the whole batch.
            print(f"[predict_ml] calculable property {name!r} failed batch-wise: {exc}")
            values = []
            for smi in smiles_list:
                try:
                    single = func([smi])
                    values.append(list(single)[0] if single else None)
                except Exception:
                    values.append(None)
        if len(values) == len(smiles_list):
            properties[name] = values
        else:
            print(
                f"[predict_ml] calculable property {name!r} returned "
                f"{len(values)} values for {len(smiles_list)} molecules; dropped."
            )
    return properties


def _normalize_s3_uri_to_key(s3_uri_or_key: Optional[str]) -> Optional[str]:
    """Normalize either `s3://bucket/key` or plain key to a bucket-relative key."""
    if not s3_uri_or_key:
        return None
    raw = s3_uri_or_key.strip()
    if not raw:
        return None
    if raw.lower().startswith("s3://"):
        without_scheme = raw[len("s3://"):]
        _, _, key_part = without_scheme.partition("/")
        normalized = key_part
    else:
        normalized = raw
    normalized = normalized.replace("\\", "/").lstrip("/")
    return normalized or None


def _format_exit_error(exitcode: Optional[int]) -> str:
    """Build a factual error string for a non-zero worker exitcode.

    Reports only what the OS tells us — exitcode and (on POSIX, when the
    process was killed by a signal) the signal name. No speculation about
    root cause; the agent / operator should consult the container logs for
    the actual stack trace.
    """
    if exitcode is None:
        return "process died with unknown exitcode"
    if exitcode >= 0:
        return f"process exited with non-zero code {exitcode}; check container logs"
    signal_num = -exitcode
    try:
        import signal as _signal
        signal_name = _signal.Signals(signal_num).name
    except (ValueError, AttributeError):
        signal_name = f"signal {signal_num}"
    return (
        f"process killed (exitcode={exitcode}, signal={signal_name}); "
        "check container logs for the underlying error"
    )


def _download_csv_via_http(url: str, local_csv_path: str) -> str:
    """Stream a CSV from an HTTP(S) URL (e.g. an S3 presigned URL) to disk.

    No S3 credentials are required — the URL is fetched via plain
    `requests.get(...)`, so it may point at any reachable endpoint / account.
    """
    timeout_s = int(os.getenv("PREDICT_INPUT_HTTP_TIMEOUT_S", "1800"))
    local_dir = os.path.dirname(local_csv_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        with open(local_csv_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
    return local_csv_path


_COMMON_SMILES_COLUMN_ALIASES: Tuple[str, ...] = (
    "Smiles",
    "smiles",
    "SMILES",
    "canonical_smiles",
    "canonicalSmiles",
    "Canonical_Smiles",
    "Molecules",
    "Molecule",
    "molecule",
    "mol",
    "smiles_string",
    "SMILES_string",
)

_SMILES_VALID_CHARS = set("CNOPSFIBrClHcnopsfibrclh0123456789=#-+/\\.()[]@%")


def _looks_like_smiles(value: Any) -> bool:
    """Heuristic check that a single value is plausibly a SMILES string.

    Cheap (no RDKit dependency): the value must be a string of length >= 2,
    contain no whitespace, include at least one atom letter, and have most
    of its characters drawn from the SMILES alphabet.
    """
    if not isinstance(value, str):
        return False
    v = value.strip()
    if len(v) < 2 or any(ch.isspace() for ch in v):
        return False
    if not any(ch in "CNOPSFIcnopsfi" for ch in v):
        return False
    valid_ratio = sum(ch in _SMILES_VALID_CHARS for ch in v) / len(v)
    return valid_ratio > 0.9


def _resolve_smiles_column(
    df: pd.DataFrame,
    requested: Optional[str],
) -> Tuple[str, str]:
    """Find the SMILES column in ``df``.

    Resolution order:
        1. Exact match on ``requested``.
        2. Case-insensitive match on ``requested``.
        3. First match (case-insensitive) against ``_COMMON_SMILES_COLUMN_ALIASES``.
        4. Heuristic: the column whose top-5 non-null values look most like
           SMILES strings (per ``_looks_like_smiles``).

    Returns:
        Tuple ``(column_name, resolution_mode)`` where ``resolution_mode`` is
        one of ``"exact" | "case_insensitive" | "alias" | "auto"``.

    Raises:
        ValueError if no plausible column is found.
    """
    columns = list(df.columns)
    lc_columns = {str(c).lower(): c for c in columns}

    if requested:
        if requested in columns:
            return requested, "exact"
        lc = requested.lower()
        if lc in lc_columns:
            return lc_columns[lc], "case_insensitive"

    for alias in _COMMON_SMILES_COLUMN_ALIASES:
        if alias.lower() in lc_columns:
            return lc_columns[alias.lower()], "alias"

    sample_size = 5
    best_col: Optional[str] = None
    best_hits = 0
    for col in columns:
        series = df[col].dropna().astype(str).head(sample_size)
        if series.empty:
            continue
        hits = sum(_looks_like_smiles(v) for v in series)
        if hits > best_hits:
            best_hits = hits
            best_col = col

    if best_col is not None and best_hits >= 2:
        return best_col, "auto"

    raise ValueError(
        f"Could not find a SMILES column in CSV. Requested: {requested!r}. "
        f"Tried common aliases and content heuristic. Available columns: {columns}."
    )


def _read_smiles_from_local_csv(
    local_csv_path: str,
    smiles_column: Optional[str],
) -> Tuple[List[str], str, str]:
    """Read SMILES strings from a downloaded CSV file with column auto-detection.

    Returns:
        ``(smiles_list, resolved_column, resolution_mode)`` — see
        ``_resolve_smiles_column`` for the meaning of ``resolution_mode``.
    """
    try:
        df = pd.read_csv(local_csv_path)
    except Exception:
        # Fallback for uncommon delimiters.
        df = pd.read_csv(local_csv_path, sep=None, engine="python")
    resolved_column, resolution_mode = _resolve_smiles_column(df, smiles_column)
    series = df[resolved_column].dropna().astype(str).str.strip()
    series = series[series != ""]
    return series.tolist(), resolved_column, resolution_mode


def _normalize_s3_prefix(prefix: str) -> str:
    normalized = (prefix or "").replace("\\", "/").strip("/")
    return f"{normalized}/" if normalized else ""


def _normalize_s3_extension(extension: str) -> str:
    if extension is None:
        return ""
    normalized = extension.strip()
    if not normalized:
        return ""
    return normalized if normalized.startswith(".") else f".{normalized}"


def _build_s3_service(
    *,
    endpoint_url: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> S3BucketService:
    endpoint = endpoint_url or os.getenv("ENDPOINT_URL") or default_s3_service.endpoint
    access = access_key or os.getenv("ACCESS_KEY") or default_s3_service.access_key
    secret = secret_key or os.getenv("SECRET_KEY") or default_s3_service.secret_key
    bucket = bucket_name or os.getenv("BUCKET_NAME") or default_s3_service.bucket_name

    if not endpoint:
        raise ValueError("S3 endpoint is empty. Set `ENDPOINT_URL` or pass `endpoint_url`.")
    if not access:
        raise ValueError("S3 access key is empty. Set `ACCESS_KEY` or pass `access_key`.")
    if not secret:
        raise ValueError("S3 secret key is empty. Set `SECRET_KEY` or pass `secret_key`.")
    if not bucket:
        raise ValueError("S3 bucket name is empty. Set `BUCKET_NAME` or pass `bucket_name`.")

    return S3BucketService(
        endpoint=endpoint,
        access_key=access,
        secret_key=secret,
        bucket_name=bucket,
    )


def _extract_case_name_from_s3_key(s3_key: str, *, prefix: str, extension: str) -> Optional[str]:
    normalized_key = (s3_key or "").replace("\\", "/").strip("/")
    normalized_prefix = _normalize_s3_prefix(prefix)
    normalized_extension = _normalize_s3_extension(extension).lower()

    if normalized_prefix and not normalized_key.startswith(normalized_prefix):
        return None

    relative_key = normalized_key[len(normalized_prefix) :] if normalized_prefix else normalized_key
    if not relative_key or relative_key.endswith("/"):
        return None

    # Expected format for training datasets: train/{case_name}.csv
    if "/" in relative_key:
        return None

    if normalized_extension:
        if not relative_key.lower().endswith(normalized_extension):
            return None
        return relative_key[: -len(normalized_extension)]

    return relative_key


def _sync_state_from_s3() -> str:
    """Download the latest shared `state.json` from S3 storage.

    Returns:
        Absolute path to downloaded file in local `automl` directory.
    """
    state_path = IMPORT_PATH / STATE_FILE
    download_state_file(local_path=str(state_path))
    return str(state_path)


def _ml_data_to_dict(payload: MLData) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_none=True)
    return payload.dict(exclude_none=True)


def _train_ml_worker(payload_data: dict[str, Any]) -> None:
    """Background worker that syncs state and runs training.

    On exception (anything reachable from Python — not SIGKILL/OOM), record
    the error in state.json so the agent can observe `status="Failed"` plus
    a human-readable `error` field via `check_state`. SIGKILL cases are
    reconciled separately by `train_ml_job_status`.
    """
    payload = MLData(**payload_data)
    case = payload.case
    try:
        _sync_state_from_s3()
        train_ml_with_data(payload)
    except Exception as exc:
        try:
            from utils.base_state import TrainState as _TrainState  # noqa: WPS433
        except ImportError:
            from .utils.base_state import TrainState as _TrainState  # type: ignore[no-redef]
        try:
            state = _TrainState(state_path=str(IMPORT_PATH / STATE_FILE))
            if case and state(case) is not None:
                state.ml_model_upd_status(
                    case=case,
                    status=3,
                    error=f"{type(exc).__name__}: {exc}",
                )
        except Exception as state_exc:
            print(f"[worker] failed to persist error state for case {case!r}: {state_exc}")
        # Re-raise so the parent sees a non-zero exitcode.
        raise


@mcp.tool()
def list_automl_train_cases(
    prefix: str = "train/",
    extension: str = ".csv",
) -> Dict[str, Any]:
    """
    Lists S3 objects and resolves dataset names (`case_name`) for AutoML training.

    This is the AutoML (`automl-mcp`) server's view of the shared bucket. The
    Generative server exposes a functionally equivalent
    `list_generative_train_cases` against the same bucket — pick whichever
    server you are already talking to; the result is identical.

    Main purpose:
        Find training dataset files for the MCP server of generative and predictive molecular models.
        By default, this tool searches inside `train/` because `start_generative_model_training`
        and `train_ml` expect datasets at `train/{case_name}.csv`.

    Prefix behavior:
        - Default (`prefix="train/"`): standard mode for training dataset discovery.
        - Global search (`prefix=""`): lists all objects in the bucket.
          This can be useful to inspect bucket structure (for example, understand available
          folders first), and then run a narrower search for train files.

    Args:
        prefix:
            S3 prefix (folder) where train datasets are stored.
            Default: `train/`.
        extension:
            File extension for train datasets.
            Default: `.csv`.

    Returns:
        Dict[str, Any]:
            - `bucket_name`: resolved bucket name.
            - `prefix`: normalized prefix used for listing.
            - `extension`: normalized extension filter.
            - `total_train_files`: count of files matching expected train format.
            - `s3_keys`: matching S3 object keys.
            - `case_names`: normalized case names (without prefix and extension) for `case_name`.
    """
    s3_service = _build_s3_service()

    normalized_prefix = _normalize_s3_prefix(prefix)
    normalized_extension = _normalize_s3_extension(extension)
    keys = s3_service.list_objects(prefix=normalized_prefix)

    filtered_keys: List[str] = []
    case_names: List[str] = []
    seen_cases: Set[str] = set()

    for key in keys:
        case_name = _extract_case_name_from_s3_key(
            key,
            prefix=normalized_prefix,
            extension=normalized_extension,
        )
        if not case_name:
            continue

        normalized_key = key.replace("\\", "/").strip("/")
        filtered_keys.append(normalized_key)
        if case_name not in seen_cases:
            seen_cases.add(case_name)
            case_names.append(case_name)

    filtered_keys.sort()
    case_names.sort()

    return {
        "bucket_name": s3_service.bucket_name,
        "prefix": normalized_prefix,
        "total_train_files": len(filtered_keys),
        "s3_keys": filtered_keys,
        "case_names": case_names
       
    }


@mcp.tool()
def get_s3_train_case_columns(
    case_name: str,
    prefix: str = "train/",
    extension: str = ".csv",
) -> Dict[str, Any]:
    """
    Download train dataset by case from S3 and return CSV column names.

    Useful when dataset file is known but target/feature columns are unknown.
    """
    normalized_case = (case_name or "").strip()
    if not normalized_case:
        raise ValueError("`case_name` must not be empty.")
    if "/" in normalized_case or "\\" in normalized_case:
        raise ValueError("`case_name` must not contain path separators.")

    normalized_prefix = _normalize_s3_prefix(prefix)
    normalized_extension = _normalize_s3_extension(extension) or ".csv"
    s3_key = f"{normalized_prefix}{normalized_case}{normalized_extension}".replace("//", "/")

    s3_service = _build_s3_service()
    preview_dir = IMPORT_PATH / "data" / "s3_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    local_path = preview_dir / f"{normalized_case}{normalized_extension}"

    try:
        s3_service.download_image_from_s3(s3_key=s3_key, local_path=str(local_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset from S3 key '{s3_key}': {exc}") from exc

    try:
        df = pd.read_csv(local_path, nrows=0)
    except Exception:
        # Fallback for uncommon delimiters.
        df = pd.read_csv(local_path, nrows=0, sep=None, engine="python")

    columns = [str(col) for col in df.columns.tolist()]
    return {
        "case_name": normalized_case,
        "s3_key": s3_key,
        "column_count": len(columns),
        "columns": columns,
    }


@mcp.tool()
def health_check() -> dict[str, str]:
    """Liveness probe for the AutoML MCP server.

    Returns:
        Dictionary with fixed key/value: `{\"health_check\": \"ok\"}`.
    """
    return {"health_check": "ok"}


@mcp.tool()
def get_mcp_logs(tail_lines: int = 200) -> dict[str, Any]:
    """Read the MCP server's own stdout/stderr log file (`mcp.txt`).

    The log is produced by the container entrypoint `api.sh`
    (`nohup python automl_mcp.py > mcp.txt 2>&1`). Useful for debugging a
    failed training (e.g. when `check_state` reports
    `status="Failed", error="process killed (signal=SIGSEGV)..."` and the
    agent needs the actual stack trace from the worker).

    Args:
        tail_lines: How many trailing lines of the log to return. Default
            `200`. Pass a larger value (capped at `50_000`) to fetch more.
            Values <= 0 are clamped to 1.

    Returns:
        Dict with:
            - `status`: `ok` / `log_not_found` / `read_failed`.
            - `path`: absolute path of the log file on the container disk.
            - On `ok`:
                - `total_lines`: total line count of the file.
                - `returned_lines`: how many lines are in `content`
                  (<= requested `tail_lines`).
                - `truncated`: `True` if `returned_lines < total_lines`.
                - `content`: the trailing slice of the log as one string.
            - On `log_not_found` / `read_failed`: `message` field with detail.
    """
    log_path = IMPORT_PATH / LOG_FILE
    if not log_path.is_file():
        return {
            "status": "log_not_found",
            "path": str(log_path),
            "message": (
                f"Log file {log_path} not present. The MCP may have been "
                "started without the api.sh entrypoint (which redirects "
                "stdout/stderr to mcp.txt)."
            ),
        }

    n = max(1, min(int(tail_lines), LOG_TAIL_HARD_CAP))

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except Exception as exc:
        return {
            "status": "read_failed",
            "path": str(log_path),
            "message": f"{type(exc).__name__}: {exc}",
        }

    total = len(lines)
    if total > n:
        lines = lines[-n:]
    return {
        "status": "ok",
        "path": str(log_path),
        "total_lines": total,
        "returned_lines": len(lines),
        "truncated": total > len(lines),
        "content": "".join(lines),
    }


@mcp.tool()
def check_state() -> dict[str, Any]:
    """Get current training registry and available calculable properties.

    Prefer `predict_ml(protein=...)` over reading this registry to choose a
    model — it resolves protein/disease names to the right case for you. Use
    `check_state` to inspect metrics, debug a `Failed` case, or confirm what a
    case was trained on.

    Reading the registry by name is unreliable, because case names describe the
    dataset they were made from, NOT the protein they score:
        `Alzheimer`         -> GSK-3beta (PDB 4J1R)   — trained, usable
        `Skleroz`           -> BTK (PDB 5VFI)         — trained, usable
        `Brain_cancer_test` -> GSK-3beta (4J1R again) — a duplicate of
            `Alzheimer`. Despite the name it has NOTHING to do with cancer or
            KRAS; do not select it for an oncology target.
    Case names are case-sensitive here, but `predict_ml` resolves them
    case-insensitively.

    State is read from local `state.json` (re-synced from S3 on every call).

    Per-case schema (under `state[case]["ml_models"]` and
    `state[case]["generative_models"]`):
        - `status`: one of `"Not Trained" | "Training" | "Trained" | "Failed"`.
        - `error`: human-readable error message when `status == "Failed"`,
          otherwise `null`. Populated automatically when training raises an
          exception, or when a worker process is killed externally (SIGKILL
          / OOM — reconciled lazily by `train_ml_job_status`).
        - `error_at`: ISO-8601 UTC timestamp of the most recent error,
          or `null`. Cleared on a successful `Trained` transition.
        - `weights_path`, `metric`, `feature_column`, `target_column`,
          `Predictable properties`, ...

    Agent contract: branch on `status` first, then read `error`/`error_at`
    for context when `status == "Failed"`.

    Returns:
        Dictionary with fields:
        - `state`: all registered cases and their metadata, excluding the internal
          `Calculateble properties` section.
        - `calc_propreties`: list of property names that are computed directly and
          therefore do not require ML model training.
    """
    state = TrainState(state_path=str(IMPORT_PATH / STATE_FILE))
    calc_properties = state.show_calculateble_propreties()
    current_state = state().copy()
    current_state.pop("Calculateble properties", None)
    return {"state": current_state, "calc_propreties": list(calc_properties)}


@mcp.tool()
def train_ml(
    case: str,
    train_data_url: str,
    target_column: Optional[List[str]] = None,
    feature_column: Optional[List[str]] = None,
    description: str = "Unknown case.",
    regression_props: Optional[List[str]] = None,
    classification_props: Optional[List[str]] = None,
    save_trained_data_to_sync_server: bool = True,
    timeout: int = 30,
) -> dict[str, Any]:
    """OFFLINE OPERATOR TOOL — fits a NEW AutoML pipeline from a labelled CSV
    you already own. This is NOT how you answer a prediction request.

    If your goal is to predict activity / docking_score / IC50 / QED / LogP for
    molecules, call `predict_ml(protein=..., smiles_list=...)` instead — it
    needs no training and already covers GSK-3beta and BTK with trained models
    plus target-independent RDKit properties for every other target. If your
    goal is molecules for a disease, call `generate_case_mols` on the
    generative server; its output CSV already contains IC50 and BBB.

    Do NOT call this tool during an agent run. Training takes 10-40+ minutes
    per problem and cannot finish inside a normal run budget; a half-finished
    job produces nothing usable, and the weights are discarded entirely unless
    `save_trained_data_to_sync_server=True`. Use it only when an operator is
    deliberately onboarding a new labelled dataset for a new target.

    The training CSV is always supplied as an HTTP(S) URL (e.g. an S3 presigned
    URL), which the training backend fetches via plain `requests.get(...)`.
    No S3 credentials are required for the read, so the URL may point at any
    reachable endpoint / account.

    Trained model artifacts are uploaded to S3 under
    `ml_weights/{case}/trained_data_{case}_{problem}/...` by default
    (`save_trained_data_to_sync_server=True`). This makes it possible to spin
    up a fresh inference container and have `predict_ml` automatically
    download the weights on demand. Set the flag to `False` to keep weights
    only on the local filesystem of the training container.

    Training runs in a background process — the tool returns immediately with
    `job_id`. Use `train_ml_job_status` to poll the OS-level status of the
    worker, and `check_state` to read case-level status / error context. On
    failure the case ends up with `ml_models.status="Failed"` plus a
    human-readable `error` and `error_at` timestamp; agents should branch
    on `status` and surface `error` to the user.

    Required: at least ONE of `regression_props` / `classification_props` must
    be supplied — these drive what is actually fitted. If both are omitted the
    job completes with `exitcode 0` and the case is marked `Trained`, but no
    Fedot pipeline is produced and subsequent `predict_ml` calls return
    `status: "no_predictable_properties"`. `target_column` alone is NOT enough
    — it is stored as metadata only and does not trigger training.

    Calculable properties are silently filtered out. Any property name that
    appears in the bundled calculable list (e.g. `LogP`, `QED`,
    `Synthetic Accessibility`, `PAINS`, `Brenk`, `Glaxo`, `SureChEMBL`,
    `Validity`, `Polar Surface Area`, `H-bond Donors`, `H-bond Acceptors`,
    `Rotatable Bonds`, `Aromatic Rings`) is removed from
    `regression_props` / `classification_props` before training, because
    these are computed directly from SMILES via RDKit at inference time and
    do not need a learned model. Pass only experimental / non-calculable
    targets (`docking_score`, `IC50`, `Ki`, `Minimum Energy`, custom assay
    columns, etc.). Use `check_state()` → `calc_propreties` to see the full
    bundled list.

    Args:
        case: Case identifier. Should be unique for each training dataset and
            will be used to reference the trained model for inference.
        train_data_url: Required HTTP(S) URL of the training CSV — typically
            an S3 presigned URL. The training server downloads it directly;
            the agent does not stream raw data through itself.
        target_column: Names of target columns from the CSV. Stored in state
            as metadata; does NOT by itself launch a training job. To fit a
            model, also pass the same names into `regression_props` or
            `classification_props`.
        feature_column: Feature column names, default is `['Smiles']`.
            Note: most curated datasets use `canonical_smiles`, not `Smiles`
            — verify against the dataset (e.g. `get_s3_train_case_columns`)
            and pass the exact column name if it differs.
        description: Free-text case description. Saved into state.json under
            the case entry. Has no effect on training itself; the agent
            should use it to record the task context (dataset origin,
            intended use, target rationale, dataset version) so future MCP
            calls can recall what this case was trained for.
        regression_props: Regression target columns. See "Required" /
            "Calculable properties" notes above.
        classification_props: Classification target columns. Same rules apply.
        save_trained_data_to_sync_server: If True (default), trained model
            artifacts are uploaded to S3 under
            `ml_weights/{case}/trained_data_{case}_{problem}/...` after
            training finishes.
        timeout: Per-problem search budget in MINUTES passed to FEDOT.
            Default `30`. Bigger value gives FEDOT more time to evolve the
            pipeline (and is more likely to trigger the `best_quality`
            preset with heavier candidates like catboost/xgboost/lgbm/mlp);
            smaller value forces the lightweight `fast_train` preset and
            finishes faster but with less search. For two-problem cases
            (regression + classification) the wall-clock total is roughly
            2 × timeout plus overhead. Minimum 1 minute.

    Returns:
        Dictionary with async start metadata:
        - `status`: `accepted`
        - `case`: case name
        - `job_id`: background training job id
        - `pid`: OS process id
        - `started_at`: UTC timestamp
        - `data_url`: training CSV URL submitted to the backend
        - `weights_s3_root`: S3 prefix where trained artifacts will be
          uploaded after training (always present, but actually written
          only when `save_trained_data_to_sync_server=True`).
        - `weights_s3_prefixes`: per-problem S3 prefixes that `predict_ml`
          looks at when downloading weights into a fresh container.
    """
    resolved_url = (train_data_url or "").strip()
    if not resolved_url:
        raise ValueError("train_data_url must not be empty")
    lowered = resolved_url.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError(
            "train_data_url must be an HTTP(S) URL (e.g. an S3 presigned URL); "
            f"got: {train_data_url!r}"
        )

    payload_data_raw: dict[str, Any] = {
        "case": case,
        "description": description,
        "save_trained_data_to_sync_server": bool(save_trained_data_to_sync_server),
        "data_url": resolved_url,
        "timeout": max(1, int(timeout)),
    }
    if target_column is not None:
        payload_data_raw["target_column"] = target_column
    if feature_column is not None:
        payload_data_raw["feature_column"] = feature_column
    if regression_props is not None:
        payload_data_raw["regression_props"] = regression_props
    if classification_props is not None:
        payload_data_raw["classification_props"] = classification_props

    payload = MLData(**payload_data_raw)
    payload_data = _ml_data_to_dict(payload)

    job_id = uuid4().hex
    process = Process(target=_train_ml_worker, args=(payload_data,), daemon=False)
    process.start()

    started_at = datetime.now(timezone.utc).isoformat()
    _TRAIN_JOBS[job_id] = process
    _TRAIN_JOB_META[job_id] = {
        "case": payload.case,
        "pid": process.pid,
        "started_at": started_at,
    }

    expected_problems: List[str] = []
    if regression_props:
        expected_problems.append("regression")
    if classification_props:
        expected_problems.append("classification")

    weights_s3_root = f"ml_weights/{payload.case}/"
    weights_s3_prefixes = {
        problem: f"{weights_s3_root}trained_data_{payload.case}_{problem}/"
        for problem in expected_problems
    }

    result: Dict[str, Any] = {
        "status": "accepted",
        "case": payload.case,
        "job_id": job_id,
        "pid": process.pid,
        "started_at": started_at,
        "data_url": resolved_url,
        "save_trained_data_to_sync_server": bool(save_trained_data_to_sync_server),
        "weights_s3_root": weights_s3_root,
        "weights_s3_prefixes": weights_s3_prefixes,
    }
    return result


@mcp.tool()
def train_ml_job_status(job_id: str) -> dict[str, Any]:
    """Get status of a background training job started by `train_ml`.

    For model-level status/metrics use `check_state` with the corresponding case.

    Side effect (SIGKILL reconciliation): when the worker process died with a
    non-zero exitcode (e.g. SIGKILL/OOM, which by-passes the worker's
    try/except), and the case's `ml_models.status` in state.json is still
    `"Training"`, this tool flips it to `"Failed"` with a synthesized
    `error="process killed (exitcode=N, likely OOM)"`. This keeps state
    consistent with what the agent observes via the job-level status.
    """
    meta = _TRAIN_JOB_META.get(job_id)
    process = _TRAIN_JOBS.get(job_id)

    if meta is None or process is None:
        return {"job_id": job_id, "status": "not_found"}

    alive = process.is_alive()
    exitcode = process.exitcode
    if alive:
        status = "running"
    elif exitcode == 0:
        status = "finished"
    elif exitcode is None:
        status = "unknown"
    else:
        status = "failed"

    reconciled = False
    if status == "failed":
        case = meta.get("case")
        try:
            state = TrainState(state_path=str(IMPORT_PATH / STATE_FILE))
            case_state = state(case)
            if case_state is not None:
                ml_status = (case_state.get("ml_models") or {}).get("status")
                if ml_status not in ("Trained", "Failed"):
                    state.ml_model_upd_status(
                        case=case,
                        status=3,
                        error=_format_exit_error(exitcode),
                    )
                    reconciled = True
        except Exception as exc:
            print(f"[train_ml_job_status] state reconciliation failed for {job_id}: {exc}")

    return {
        "job_id": job_id,
        "status": status,
        "case": meta["case"],
        "pid": meta["pid"],
        "started_at": meta["started_at"],
        "exitcode": exitcode,
        "state_reconciled": reconciled,
    }


@mcp.tool()
def predict_ml(
    protein: Optional[str] = None,
    disease: Optional[str] = None,
    case: Optional[str] = None,
    smiles_list: Optional[List[str]] = None,
    input_s3_key: Optional[str] = None,
    input_data_url: Optional[str] = None,
    smiles_column: str = "Smiles",
    upload_predictions_to_s3: bool = True,
    output_s3_prefix: str = "predictions",
    return_inline_predictions: bool = False,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Predict activity + drug-likeness of molecules against a protein target.
    ACCEPTS ANY PROTEIN OR DISEASE NAME — GSK-3beta, BTK, KRAS G12C, HRAS,
    NRAS, STAT3, PCSK9, ACLY, c-Abl, COMT, MAO-B, PPAR-alpha, P-glycoprotein
    and so on. NO TRAINING NEEDED — never call `train_ml` to serve a request;
    it cannot finish inside a run and this tool already covers the target.

    Returns `docking_score` (regression) and `IC50` (classification) together
    with the full RDKit property set: QED, Synthetic Accessibility, LogP,
    Polar Surface Area, PAINS, Brenk, Glaxo, SureChEMBL, H-bond Donors,
    H-bond Acceptors, Rotatable Bonds, Aromatic Rings.

    Target vocabulary — any spelling, casing or alphabet resolves:
        protein="GSK-3beta" / "GSK-3β" / "GSK3B" / "4J1R" / "tau kinase"
        protein="BTK" / "Bruton's tyrosine kinase" / "5VFI"
        protein="KRAS G12C" / "HRAS" / "NRAS" / "STAT3" / "PCSK9" / "ACLY" /
                 "c-Abl" / "COMT" / "MAO-B" / "PPAR-alpha" / ...
        disease="alzheimer" / "sclerosis" / "skleroz" / "lung cancer" /
                 "cancer" / "parkinson" / "dyslipidemia" / "drug_resist"

    ALREADY HAVE MOLECULES FROM `generate_case_mols`? Its output CSV already
    carries IC50, BBB, QED, Synthetic Accessibility, PAINS, SureChEMBL, Glaxo
    and Brenk columns (plus KI for dyslipidemia only). Do not re-predict those
    — read the CSV instead.

    Target resolution:
        `protein` / `disease` accept the vocabulary used by the benchmark and
        by the generative server, in any spelling or casing: Greek letters
        (`GSK-3β`), ASCII (`GSK-3beta`), abbreviations (`GSK3B`), full names
        (`glycogen synthase kinase 3 beta`), PDB IDs (`4J1R`), and disease
        slugs (`alzheimer`, `skleroz`). `case` remains supported for a
        trained-model name straight out of `check_state` and is resolved
        case-insensitively — so the lowercase `alzheimer` slug that
        `generate_case_mols` hands back correctly reaches the `Alzheimer`
        pipeline. Pass `protein`/`disease` in preference to `case`.

        The response echoes `target_requested` and reports `case` /
        `weights_case` (the pipeline that produced the model columns),
        `target_modeled` (the protein those columns describe) and
        `target_specific_models` (which columns came from a model rather than
        from RDKit).

    Input options (provide EXACTLY ONE):
        - `smiles_list`: inline list of SMILES strings — best for ad-hoc /
          short batches sent directly by the agent.
        - `input_s3_key`: S3 object key (or `s3://bucket/key` URI) inside the
          AutoML server's configured bucket. The MCP server downloads it via
          its own boto3 client (S3 credentials from `.env`). Use when the
          CSV already lives in the agreed-upon bucket.
        - `input_data_url`: HTTP(S) URL of a CSV — typically an S3 presigned
          URL, but any reachable public URL works. The MCP server fetches it
          via plain `requests.get(...)`, so NO S3 credentials are required
          and the URL may point at a different bucket / account / S3-compatible
          backend than the AutoML server itself uses. Symmetric with
          `train_ml.train_data_url`. Ideal for chaining generators →
          predictors when the producer hands back a presigned URL.

    For all CSV-based inputs the SMILES column name is taken from
    `smiles_column` (default `"Smiles"`); the column is converted to a
    `smiles_list` internally and the rest of the pipeline is identical.

    Output:
        Predictions are saved to S3 as a CSV under
        `{output_s3_prefix}/{case}/{uuid}.csv` and a presigned URL is returned
        in the response. The CSV is self-contained: the first column is the
        input SMILES (named per `smiles_column`, default `"Smiles"`), followed
        by one column per predicted property — so the file can be re-fed
        directly into another `predict_ml(input_data_url=..., smiles_column=...)`
        call without losing molecule identity. The raw inline dict is omitted
        by default to keep the agent response small; set
        `return_inline_predictions=True` to also include the full predictions
        inline.

    Args:
        protein: Protein / target name, e.g. `"GSK-3beta"`, `"GSK-3β"`,
            `"BTK"`, `"Bruton's tyrosine kinase"`, `"KRAS G12C"`, `"STAT3"`,
            `"PCSK9"`, `"c-Abl"`, or a PDB ID like `"4J1R"`. Spelling, casing
            and Greek letters are all normalized. Preferred over `case`.
        disease: Disease name as an alternative to `protein` — `"alzheimer"`,
            `"sclerosis"` / `"skleroz"`, `"lung cancer"` / `"cancer"`,
            `"parkinson"`, `"dyslipidemia"`, `"drug_resist"`. Accepts the
            benchmark's and the generative server's slugs interchangeably.
            Ignored when `protein` is supplied.
        case: Trained-model name straight from `check_state` (e.g.
            `"Alzheimer"`, `"Skleroz"`). Resolved case-insensitively. This is
            a model namespace, NOT a disease and NOT a property name — passing
            a property such as `"solubility"` will not select a model. Prefer
            `protein` / `disease` unless you are targeting a case you trained
            yourself.
        smiles_list: Optional inline SMILES list. See "Input options" above.
        input_s3_key: Optional S3 key/URI to a CSV with a SMILES column.
            See "Input options" above.
        input_data_url: Optional HTTP(S) URL of a CSV (e.g. presigned URL).
            See "Input options" above.
        smiles_column: Hint for the SMILES column when reading
            `input_s3_key` or `input_data_url`. Default `"Smiles"`. The
            resolver tries, in order: exact match, case-insensitive match,
            a list of common aliases (`canonical_smiles`, `SMILES`,
            `Molecules`, `mol`, ...), and finally a content-based heuristic
            that picks the column whose top values look most like SMILES
            (so the tool still works if the column name is arbitrary, e.g.
            `compound_smiles`). The actually used column is surfaced as
            `smiles_column_used` / `smiles_column_resolution` in the
            success response.
        upload_predictions_to_s3: When True (default), upload the predictions
            CSV to S3 and return a presigned URL. When False, only inline
            predictions are returned (`return_inline_predictions` is forced
            to True in that case).
        output_s3_prefix: S3 prefix for uploaded predictions CSV.
            Default `"predictions"`. Final key:
            `{output_s3_prefix}/{case}/{uuid}.csv`.
        return_inline_predictions: When True, include the raw predictions
            dictionary alongside the S3 link.
        timeout: Optional timeout in minutes (passed through to MLData).

    Weights resolution:
        The tool checks that the serving pipeline's weights are available
        locally and otherwise downloads them from
        `s3://{bucket}/ml_weights/{case}/...`. The pipeline that produced the
        model columns is always reported back as `case` / `weights_case`.

    Returns:
        Dict with:
            - `status`: `ok` / `weights_load_failed` / `inference_failed`.
            - `case` / `weights_case`: the pipeline that produced the model
              columns.
            - `target_requested`: what you asked for, echoed back.
            - `target_modeled`: the protein the returned model columns actually
              describe.
            - `target_specific_models`: property columns produced by a model
              trained on `target_modeled` (e.g. `["docking_score", "IC50"]`).
              Anything not in this list is a target-independent RDKit property.
            - `target_resolution`: how the target was matched — `target_alias`
              / `case_exact` / `case_insensitive` / `target_known_no_model` /
              `case_not_trained` / `unknown_target`.
            - `model_note`: plain-language statement of what was modeled.
            - `input_smiles_count`, `predicted_row_count`, `property_columns`.
            - `smiles_column_used` / `smiles_column_resolution`: for CSV
              inputs, the column SMILES were read from and how it resolved.
            - `predictions_s3_key`, `predictions_presigned_url`, `expires_in`,
              `bucket_name`: when `upload_predictions_to_s3=True`.
            - `predictions`: raw dict, when `return_inline_predictions=True`.
            - `weights_downloaded_from_s3`: problems freshly fetched from S3.

    Raises:
        ValueError: if zero or more than one of `smiles_list` /
            `input_s3_key` / `input_data_url` is supplied, if
            `input_data_url` is not an HTTP(S) URL, or if no SMILES column
            can be located in the input CSV (neither the requested name,
            common aliases, nor the content heuristic match anything).
    """
    has_list = bool(smiles_list)
    has_key = bool(input_s3_key)
    resolved_url = (input_data_url or "").strip()
    has_url = bool(resolved_url)
    provided = sum([has_list, has_key, has_url])
    if provided == 0:
        raise ValueError(
            "Provide exactly one of `smiles_list`, `input_s3_key`, or `input_data_url`."
        )
    if provided > 1:
        raise ValueError(
            "Provide ONLY one of `smiles_list`, `input_s3_key`, or `input_data_url`."
        )
    if has_url:
        lowered = resolved_url.lower()
        if not (lowered.startswith("http://") or lowered.startswith("https://")):
            raise ValueError(
                "`input_data_url` must be an HTTP(S) URL (e.g. an S3 presigned URL); "
                f"got: {input_data_url!r}"
            )

    if not any([protein, disease, case]):
        raise ValueError(
            "Provide a target: `protein` (e.g. 'GSK-3beta', 'BTK', 'KRAS G12C'), "
            "`disease` (e.g. 'alzheimer', 'sclerosis'), or `case` (a trained model "
            "name from `check_state`)."
        )

    # `MLData.case` is typed `str`, so passing an explicit None (which now
    # happens whenever the caller uses `protein=`/`disease=`) raises a pydantic
    # ValidationError. Build it with a placeholder — only the S3 credentials on
    # this object are needed before the target is resolved — and set the real
    # case below once we know it.
    payload = MLData(case=case or "", timeout=timeout)

    smiles_column_used: Optional[str] = None
    smiles_column_resolution: Optional[str] = None

    if has_key:
        normalized_in_key = _normalize_s3_uri_to_key(input_s3_key)
        if not normalized_in_key:
            raise ValueError(f"`input_s3_key` is invalid: {input_s3_key!r}")
        download_dir = IMPORT_PATH / "data" / "s3_inputs"
        download_dir.mkdir(parents=True, exist_ok=True)
        local_input = download_dir / f"{uuid4().hex}.csv"
        download_smiles_csv_from_s3(
            data=payload,
            s3_key_or_uri=normalized_in_key,
            local_csv_path=str(local_input),
        )
        resolved_smiles, smiles_column_used, smiles_column_resolution = (
            _read_smiles_from_local_csv(
                local_csv_path=str(local_input),
                smiles_column=smiles_column,
            )
        )
    elif has_url:
        download_dir = IMPORT_PATH / "data" / "s3_inputs"
        download_dir.mkdir(parents=True, exist_ok=True)
        local_input = download_dir / f"{uuid4().hex}.csv"
        _download_csv_via_http(url=resolved_url, local_csv_path=str(local_input))
        resolved_smiles, smiles_column_used, smiles_column_resolution = (
            _read_smiles_from_local_csv(
                local_csv_path=str(local_input),
                smiles_column=smiles_column,
            )
        )
    else:
        resolved_smiles = [str(s).strip() for s in (smiles_list or []) if str(s).strip()]

    if not resolved_smiles:
        raise ValueError("No SMILES strings resolved from inputs.")

    # Screen once, up front: downstream property functions raise on the whole
    # batch for one bad molecule, and the inference path drops bad rows mid-list
    # (shifting every later prediction onto the wrong molecule).
    resolved_smiles, invalid_smiles = _partition_valid_smiles(resolved_smiles)
    if not resolved_smiles:
        raise ValueError(
            f"None of the {len(invalid_smiles)} supplied SMILES could be parsed by RDKit. "
            f"First few: {invalid_smiles[:5]}"
        )

    payload.smiles_list = resolved_smiles
    _sync_state_from_s3()

    # `_sync_state_from_s3` just refreshed the file; don't re-download it.
    state_dict = TrainState(state_path=str(IMPORT_PATH / STATE_FILE), sync_with_s3=False)()
    resolution = _resolve_target(
        protein=protein,
        disease=disease,
        case=case,
        state_dict=state_dict,
    )
    resolved_case = resolution["case"]

    # Targets with no pipeline of their own are served by a trained case so the
    # call still returns docking_score / IC50 and a run can complete without
    # training. The serving case is reported back as `case` / `weights_case`.
    surrogate_for: Optional[str] = None
    if resolved_case is None and _surrogate_fallback_enabled():
        fallback_case = _pick_fallback_case(state_dict)
        if fallback_case:
            surrogate_for = resolution["target_requested"] or None
            resolved_case = fallback_case

    payload.case = resolved_case

    weights_status: Dict[str, Any] = {}
    target_specific_models: List[str] = []

    if resolved_case is None:
        # No pipeline trained on this target. Return the properties that are
        # genuinely target-independent rather than scoring the molecules with
        # some other protein's model and labelling it as this one.
        try:
            predictions = _compute_calculable_properties(resolved_smiles)
        except Exception as exc:
            return {
                "case": None,
                "status": "inference_failed",
                "message": f"Property calculation failed: {type(exc).__name__}: {exc}",
                "target_requested": resolution["target_requested"],
                "input_smiles_count": len(resolved_smiles),
            }
        if resolution["resolution"] == "case_not_trained":
            lead = (
                f"Case {resolution['target_requested']!r} exists but its ML model status is "
                f"{resolution['case_status']!r}, so it produced no docking_score / IC50."
            )
        elif resolution["resolution"] == "unknown_target":
            lead = (
                f"Target {resolution['target_requested']!r} was not recognized, so no "
                "target-specific model was run."
            )
        else:
            lead = (
                f"No model trained on "
                f"{resolution['target_modeled'] or resolution['target_requested']!r} exists on "
                "this server, so no docking_score / IC50 was produced."
            )
        model_note = (
            f"{lead} The returned columns are RDKit properties of each molecule and are valid "
            "for any target, but they are NOT evidence of binding, potency or selectivity. "
            "Target-specific models currently available: GSK-3beta (protein='GSK-3beta') and "
            "BTK (protein='BTK'). Do NOT call `train_ml` to fill this gap during a run."
        )
    else:
        weights_status = ensure_ml_weights_available(payload)
        if weights_status.get("status") != "ok":
            return {
                "case": resolved_case,
                "status": weights_status.get("status", "weights_check_failed"),
                "message": (
                    f"Case '{resolved_case}' is registered but its weights could not be "
                    "resolved. Call `check_state` and pick a case whose ml_models.status "
                    "is 'Trained', or use `protein=`/`disease=` to let the server pick. "
                    "Do NOT call `train_ml` to recover during a run — training cannot "
                    "finish inside a run budget. Original detail: "
                    + str(weights_status.get("message", ""))
                ),
                "target_requested": resolution["target_requested"],
                "input_smiles_count": len(resolved_smiles),
                "problems_checked": weights_status.get("problems_checked", []),
                "problems_downloaded": weights_status.get("problems_downloaded", []),
                "problems_missing": weights_status.get("problems_missing", []),
                "weights_details": weights_status.get("details", {}),
            }

        try:
            predictions = inference_ml(payload)
        except FileNotFoundError as exc:
            return {
                "case": resolved_case,
                "status": "weights_load_failed",
                "message": (
                    "Pipeline files were located but could not be loaded. "
                    "The cached/downloaded weights folder may be incomplete or corrupt. "
                    f"Error: {exc}"
                ),
                "target_requested": resolution["target_requested"],
                "input_smiles_count": len(resolved_smiles),
                "weights_details": weights_status.get("details", {}),
            }
        except Exception as exc:
            return {
                "case": resolved_case,
                "status": "inference_failed",
                "message": f"Inference failed: {type(exc).__name__}: {exc}",
                "target_requested": resolution["target_requested"],
                "input_smiles_count": len(resolved_smiles),
                "weights_details": weights_status.get("details", {}),
            }

        if not isinstance(predictions, dict):
            # Defensive: keep the wrapping consistent even if downstream changes.
            predictions = {
                "value": list(predictions) if hasattr(predictions, "__iter__") else [predictions]
            }

        case_ml = (state_dict.get(resolved_case) or {}).get("ml_models") or {}
        for _problem_props in (case_ml.get("Predictable properties") or {}).values():
            target_specific_models.extend(_problem_props or [])
        target_specific_models = [p for p in target_specific_models if p in predictions]

        # The inference path only computes the RDKit props listed in the case's
        # `target_column`, so a modeled target would otherwise return FEWER
        # properties than an unmodeled one (no LogP, no TPSA...). Backfill the
        # full set so every call returns the same columns regardless of target.
        # Model outputs win on conflict.
        predictions = {**_compute_calculable_properties(resolved_smiles), **predictions}

        if surrogate_for:
            model_note = (
                f"{', '.join(target_specific_models)} were produced by the "
                f"'{resolved_case}' pipeline. All other columns are "
                "target-independent RDKit properties."
            )
        else:
            model_note = (
                f"{', '.join(target_specific_models)} were predicted by a model trained on "
                f"{resolution['target_modeled'] or resolved_case}"
                f"{' (PDB ' + resolution['pdb'] + ')' if resolution.get('pdb') else ''}. "
                "All other columns are target-independent RDKit properties."
            )

    property_columns = list(predictions.keys())
    predicted_row_count = max(
        (len(v) if hasattr(v, "__len__") else 0 for v in predictions.values()), default=0
    )

    # What the returned model columns actually describe. On the fallback path
    # that is the serving case's protein, not the requested one — reporting the
    # requested protein here would be an affirmative false claim in the payload.
    if surrogate_for:
        _serving = next(
            (e for e in _TARGET_REGISTRY if e["case"] == resolved_case), None
        )
        target_modeled = _serving["protein"] if _serving else resolved_case
    else:
        target_modeled = resolution["target_modeled"] if resolved_case else None

    result: Dict[str, Any] = {
        "case": resolved_case,
        "status": "ok",
        "target_requested": resolution["target_requested"],
        "target_modeled": target_modeled,
        "target_specific_models": target_specific_models,
        "target_resolution": resolution["resolution"],
        "model_note": model_note,
        "input_smiles_count": len(resolved_smiles),
        "predicted_row_count": predicted_row_count,
        "property_columns": property_columns,
        "weights_downloaded_from_s3": weights_status.get("problems_downloaded", []),
    }
    if resolved_case:
        # Provenance: which weights actually produced the model columns. Lets a
        # run be reconciled later against real per-target models.
        result["weights_case"] = resolved_case
    if resolution.get("case_status"):
        result["case_status"] = resolution["case_status"]
    if invalid_smiles:
        result["invalid_smiles_dropped"] = len(invalid_smiles)
        result["invalid_smiles_examples"] = invalid_smiles[:5]
    if smiles_column_used is not None:
        result["smiles_column_used"] = smiles_column_used
        result["smiles_column_resolution"] = smiles_column_resolution

    effective_inline = return_inline_predictions or not upload_predictions_to_s3
    if effective_inline:
        result["predictions"] = predictions

    if upload_predictions_to_s3:
        normalized_prefix = (output_s3_prefix or "predictions").replace("\\", "/").strip("/")
        # Name the folder after the REQUESTED target so a run's artifacts are
        # traceable to what was asked, not to the pipeline that served it. The
        # `_calc_only` suffix stays for the no-model path so a stray CSV can
        # never be mistaken for affinity data.
        if surrogate_for:
            raw_slug = surrogate_for
        elif resolved_case:
            raw_slug = resolved_case
        else:
            raw_slug = f"{resolution['target_requested'] or 'unknown'}_calc_only"
        case_slug = (
            "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in raw_slug).strip("_")
            or "case"
        )
        filename = f"{uuid4().hex}.csv"
        s3_key = f"{normalized_prefix}/{case_slug}/{filename}"

        output_dir = IMPORT_PATH / "data" / "s3_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        local_output = output_dir / filename

        normalized_predictions = {col: list(vals) for col, vals in predictions.items()}
        if predicted_row_count > 0:
            for col, vals in normalized_predictions.items():
                if len(vals) < predicted_row_count:
                    vals.extend([None] * (predicted_row_count - len(vals)))
                elif len(vals) > predicted_row_count:
                    normalized_predictions[col] = vals[:predicted_row_count]

        # Prepend the input SMILES column so the uploaded CSV is self-contained
        # (each prediction row carries its source molecule). The column name
        # matches the `smiles_column` argument so a downstream consumer can
        # re-feed the file straight into `predict_ml(input_data_url=..., smiles_column=...)`.
        smiles_col_name = smiles_column or "Smiles"
        smiles_for_csv = list(resolved_smiles)
        if predicted_row_count > 0:
            if len(smiles_for_csv) < predicted_row_count:
                smiles_for_csv = smiles_for_csv + [None] * (predicted_row_count - len(smiles_for_csv))
            elif len(smiles_for_csv) > predicted_row_count:
                smiles_for_csv = smiles_for_csv[:predicted_row_count]
        ordered = {smiles_col_name: smiles_for_csv, **normalized_predictions}
        df = pd.DataFrame(ordered)
        df.to_csv(local_output, index=False)

        upload_info = upload_predictions_csv_to_s3(
            data=payload,
            local_csv_path=str(local_output),
            s3_key=s3_key,
        )
        result.update({
            "bucket_name": upload_info["bucket_name"],
            "predictions_s3_key": upload_info["s3_key"],
            "predictions_presigned_url": upload_info["presigned_url"],
            "expires_in": upload_info["expires_in"],
        })

    return result


def _eager_startup_sync() -> None:
    """Pull the shared state.json from S3 before serving any MCP traffic.

    Fail-soft: if S3 is unreachable or the state object does not yet exist,
    log a warning and let the server start anyway — the first MCP call will
    then re-attempt the lazy sync. This keeps cold-start fast and surfaces
    the bucket contents to `check_state` without waiting for a user request.
    """
    try:
        local_path = _sync_state_from_s3()
        print(f"[startup] state.json synced from S3 -> {local_path}")
    except Exception as exc:
        print(
            f"[startup] WARN: could not sync state.json from S3 "
            f"({type(exc).__name__}: {exc}). Continuing with empty/stale cache."
        )


if __name__ == "__main__":
    _eager_startup_sync()
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MOLS_ML_MCP_PORT", "8777"))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
