from typing import Any, Dict, List, Optional
from pathlib import Path
from fastapi import Body
import os
import sys
import types
import requests
from pydantic import BaseModel
from inference import predict_smiles
from autotrain.utils.base_state import TrainState
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
from GAN.gan_lstm_refactoring.gen import generate
from generate.config import parsing
from train_data.utils.config_generate import configurate_parser
from train_data.utils.s3_utils import S3BucketService
from train_data.generate import generator_for_agent_multi_prop as multi_generator
import pandas as pd
from utils.validation import check_chem_valid, eval_P_S_G, eval_qed, eval_sa, check_brenk
import lightgbm
from utils.ic_50_models.alzheimer.predict_ic50_clf import eval_ic_50_alzheimer
from utils.ic_50_models.skleroz_ic50_clf.scripts.predict_ic50_btk_clf import eval_ic_50_sklrz
from utils.ic_50_models.kras_ic50_prediction.predict_ic50_clf import eval_ic_50_cancer
from utils.ic_50_models.citrate_classif_inference.inference_citrate_clf import predict as parkenson_predict_ic50
from utils.ic_50_models.drug_resis_classif_inference.inference_drug_clf import predict as drug_res_predict_ic50
from utils.ic_50_models.tyrosine_classif_inference.inference_tyrosine_clf import predict as dyslip_predict_ic50
from utils.ki_models.tyrosine_regression_inference.tyrosine_inference_regr import predict as dyslip_predict_ki
from utils.inference_BB_clf.BB_inference import predict as eval_bbb
from autotrain.auto_train import main, main_generate
###Docking
from autodock_vina_python3.src.docking_score import docking_list
from utils.check_novelty import check_novelty_chembl
import pickle
from GAN.gan_lstm_refactoring.train_gan import auto_train
import GAN.gan_lstm_refactoring.scripts.model as GAN_PICKLE_MODEL
import GAN.gan_lstm_refactoring.scripts.utils as GAN_PICKLE_UTILS
import GAN.gan_lstm_refactoring.scripts.layers as GAN_PICKLE_LAYERS
import GAN.gan_lstm_refactoring.scripts.tokenizer as GAN_PICKLE_TOKENIZER


def _register_gan_pickle_aliases():
    # Backward compatibility for checkpoints pickled with legacy `scripts.*` paths.
    scripts_pkg = sys.modules.get("scripts")
    if scripts_pkg is None:
        scripts_pkg = types.ModuleType("scripts")
        scripts_pkg.__path__ = []
        sys.modules["scripts"] = scripts_pkg

    sys.modules["GAN.gan_lstm_refactoring.scripts.model"] = GAN_PICKLE_MODEL
    sys.modules["GAN.gan_lstm_refactoring.scripts.utils"] = GAN_PICKLE_UTILS
    sys.modules["GAN.gan_lstm_refactoring.scripts.layers"] = GAN_PICKLE_LAYERS
    sys.modules["GAN.gan_lstm_refactoring.scripts.tokenizer"] = GAN_PICKLE_TOKENIZER
    sys.modules["scripts.model"] = GAN_PICKLE_MODEL
    sys.modules["scripts.utils"] = GAN_PICKLE_UTILS
    sys.modules["scripts.layers"] = GAN_PICKLE_LAYERS
    sys.modules["scripts.tokenizer"] = GAN_PICKLE_TOKENIZER

    scripts_pkg.model = GAN_PICKLE_MODEL
    scripts_pkg.utils = GAN_PICKLE_UTILS
    scripts_pkg.layers = GAN_PICKLE_LAYERS
    scripts_pkg.tokenizer = GAN_PICKLE_TOKENIZER


_register_gan_pickle_aliases()


def _normalize_prop_values(values, expected_size: int) -> list:
    if values is None:
        return [None] * expected_size
    if isinstance(values, pd.Series):
        values = values.tolist()
    elif hasattr(values, "tolist") and not isinstance(values, list):
        values = values.tolist()
    elif isinstance(values, tuple):
        values = list(values)
    elif not isinstance(values, list):
        values = [values] * expected_size

    if expected_size == 0:
        return []
    if len(values) == expected_size:
        return values
    if len(values) == 1:
        return values * expected_size
    if len(values) > expected_size:
        return values[:expected_size]
    return values + [None] * (expected_size - len(values))


_GENERIC_GAN_CASE_TOKENS = {"", "rnmd", "rndm", "default", "gan_default", "none"}


def _build_s3_service_for_gan(
    endpoint_url: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> Optional[S3BucketService]:
    """Build an S3 client from explicit values or env vars. Returns None if
    any required credential is missing — callers can then degrade gracefully
    instead of crashing the FastAPI request."""
    endpoint = endpoint_url or os.getenv("ENDPOINT_URL")
    access = access_key or os.getenv("ACCESS_KEY")
    secret = secret_key or os.getenv("SECRET_KEY")
    bucket = bucket_name or os.getenv("BUCKET_NAME")
    if not all([endpoint, access, secret, bucket]):
        return None
    return S3BucketService(
        endpoint=endpoint,
        access_key=access,
        secret_key=secret,
        bucket_name=bucket,
    )


def upload_gan_weights_folder_to_s3(
    case: str,
    local_folder: str,
    s3_service: Optional[S3BucketService] = None,
) -> Dict[str, Any]:
    """Recursively upload `local_folder` into `gan_weights/{case}/{folder_name}/...`
    in the configured S3 bucket. Returns a small status dict (uploaded files,
    bucket, prefix) — never raises on missing credentials, just reports `no_s3`.
    """
    folder_path = Path(local_folder)
    if not folder_path.is_dir():
        return {
            "status": "no_local_folder",
            "message": f"Local weights folder not found: {folder_path}",
            "local_folder": str(folder_path),
        }
    service = s3_service if s3_service is not None else _build_s3_service_for_gan()
    if service is None:
        return {
            "status": "no_s3",
            "message": "S3 credentials not set; skipping weights upload.",
            "local_folder": str(folder_path),
        }
    case_slug = (case or "").strip().strip("/")
    if not case_slug:
        return {
            "status": "no_case",
            "message": "Case name is empty; skipping weights upload.",
        }
    base_prefix = f"gan_weights/{case_slug}/{folder_path.name}".strip("/")
    uploaded = 0
    for file_path in folder_path.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(folder_path).as_posix()
        s3_key = f"{base_prefix}/{rel_path}".strip("/")
        if "/" in s3_key:
            prefix, source_file_name = s3_key.rsplit("/", 1)
        else:
            prefix, source_file_name = "", s3_key
        service.upload_file_object(
            prefix=prefix,
            source_file_name=source_file_name,
            file_path=str(file_path),
        )
        uploaded += 1
    return {
        "status": "uploaded",
        "bucket_name": service.bucket_name,
        "s3_prefix": base_prefix + "/",
        "files_uploaded": uploaded,
        "local_folder": str(folder_path),
    }


def download_gan_weights_folder_from_s3(
    s3_prefix: str,
    local_target_dir: str,
    s3_service: Optional[S3BucketService] = None,
) -> int:
    """Download every object under `s3_prefix` into `local_target_dir`.

    Returns:
        Number of files downloaded. Zero means the prefix is empty / does not
        exist or S3 credentials are missing.
    """
    service = s3_service if s3_service is not None else _build_s3_service_for_gan()
    if service is None:
        return 0
    normalized_prefix = s3_prefix.replace("\\", "/").lstrip("/")
    if not normalized_prefix.endswith("/"):
        normalized_prefix = normalized_prefix + "/"
    keys = service.list_objects(prefix=normalized_prefix)
    if not keys:
        return 0
    target = Path(local_target_dir)
    target.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for key in keys:
        if key.startswith(normalized_prefix):
            relative = key[len(normalized_prefix):]
        else:
            relative = key
        relative = relative.lstrip("/")
        if not relative:
            continue
        local_file = target / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)
        service.download_image_from_s3(s3_key=key, local_path=str(local_file))
        downloaded += 1
    return downloaded


def ensure_gan_weights_available(case: str) -> Dict[str, Any]:
    """Verify that GAN weights for `case` are available locally; download from
    S3 if missing.

    Returns:
        Dict with keys:
        - `status`: one of
            * `ok`              — `gan_weights.pkl` is present locally
              (cached or just downloaded from S3).
            * `case_not_found`  — case is not registered in state.json.
            * `case_not_trained`— case is registered but has no `weights_path`
              recorded (training never finished or never ran).
            * `weights_not_found` — `weights_path` is set, but the file is
              missing locally and no S3 backup exists at the expected prefix.
        - `case`, `message`, `local_path`, `s3_prefix`, `source`,
          `files_downloaded` — populated where relevant.
    """
    case_name = (case or "").strip()
    if not case_name:
        return {
            "status": "case_not_found",
            "case": case,
            "message": "Empty `case` name.",
        }

    state = TrainState(state_path='autotrain/utils/state.json')
    case_state = state(case_name)
    if case_state is None:
        return {
            "status": "case_not_found",
            "case": case_name,
            "message": (
                f"Case '{case_name}' is not registered in state.json. "
                "Train a GAN first via `start_generative_model_training`."
            ),
        }

    gen_state = state(case_name, "gen")
    weights_path = gen_state.get("weights_path") if gen_state else None
    if not weights_path:
        return {
            "status": "case_not_trained",
            "case": case_name,
            "message": (
                f"Case '{case_name}' has no GAN weights recorded. "
                "Train via `start_generative_model_training`."
            ),
        }

    weights_dir = Path(weights_path)
    weights_file = weights_dir / "gan_weights.pkl"
    if weights_file.is_file():
        return {
            "status": "ok",
            "case": case_name,
            "source": "local_cache",
            "local_path": str(weights_file),
            "weights_dir": str(weights_dir),
        }

    folder_name = weights_dir.name
    s3_prefix = f"gan_weights/{case_name}/{folder_name}/"
    try:
        count = download_gan_weights_folder_from_s3(s3_prefix, str(weights_dir))
    except Exception as exc:
        return {
            "status": "weights_not_found",
            "case": case_name,
            "message": (
                f"Failed to download GAN weights from S3 prefix '{s3_prefix}': "
                f"{type(exc).__name__}: {exc}"
            ),
            "s3_prefix": s3_prefix,
            "weights_dir": str(weights_dir),
        }

    if count > 0 and weights_file.is_file():
        return {
            "status": "ok",
            "case": case_name,
            "source": "s3",
            "local_path": str(weights_file),
            "weights_dir": str(weights_dir),
            "s3_prefix": s3_prefix,
            "files_downloaded": count,
        }

    return {
        "status": "weights_not_found",
        "case": case_name,
        "message": (
            f"GAN weights for case '{case_name}' are not available locally and "
            f"no S3 backup at `{s3_prefix}` (in the configured bucket). "
            "Retrain via `start_generative_model_training` "
            "(default `save_trained_data_to_sync_server=True`)."
        ),
        "s3_prefix": s3_prefix,
        "weights_dir": str(weights_dir),
    }


class GenData(BaseModel):
        numb_mol: int =1
        model:str = None
        cuda:bool=True
        mean_:float=0
        std_:float=1
        case_ : str = 'RNMD'
        url:str = os.getenv('ML_MOLS_MODEL_APP_URL')

class TrainData(BaseModel):
        data:dict = None
        case:str = None
        data_path:str = None
        target_column:list = None
        #smiles_list: list = None
        timeout:int = 30 #30 min
        feature_column:list = ['Smiles']
        path_to_save:str = 'automl/trained_data'
        description:str = 'Unknown case.'
        url:str = os.getenv('ML_MOLS_MODEL_APP_URL')
        n_samples:int = 1000
        fine_tune:bool = True
        new_vocab:bool = False
        epochs:int = 10
        batchsize:int = 2048
        # regression_props:list= None
        # classification_props:list = None

class TrainDataS3(BaseModel):
        case:str = None
        endpoint_url:str = os.getenv("ENDPOINT_URL")
        access_key:str = os.getenv("ACCESS_KEY")
        secret_key:str = os.getenv("SECRET_KEY")
        bucket_name:str = os.getenv("BUCKET_NAME")
        s3_key:str = None
        # Full HTTP/HTTPS URL of the training CSV (e.g. an S3 presigned URL).
        # When set, takes precedence over `s3_key`: the CSV is fetched via
        # `requests.get(...)` and no S3 credentials are required to read it.
        data_url:str = None
        data_path:str = None
        feature_column:list = ['Smiles']
        # Backward compatibility with common typo/casing in clients.
        Future_column:list = None
        future_column:list = None
        fine_tune:bool = True
        epochs:int = 10
        # When True, after `auto_train` finishes the trained GAN folder
        # (autotrain/GAN_weights/train_GAN_{case}/) is recursively uploaded to
        # `s3://{bucket}/gan_weights/{case}/train_GAN_{case}/...` so a fresh
        # inference container can fetch it on demand.
        save_trained_data_to_sync_server:bool = True
        # Backward compatibility with previous payload names.
        s3_bucket:str = None
        s3_endpoint_url:str = None

class Molecules(BaseModel):
    mol_list:List[str]

class Docking_config(BaseModel):
    mol_list:List[str]
    receptor_case : str = 'Alzhmr'

def condition_enchance():
     pass


def _resolve_feature_columns(data) -> list:
    feature_column = data.feature_column
    if data.Future_column is not None and (feature_column is None or feature_column == ['Smiles']):
        feature_column = data.Future_column
    if data.future_column is not None and (feature_column is None or feature_column == ['Smiles']):
        feature_column = data.future_column

    if isinstance(feature_column, str):
        feature_column = [feature_column]
    if not feature_column:
        raise ValueError("`feature_column` (or `Future_column`) is required.")
    return feature_column


_SMILES_COLUMN_ALIASES = {
    "smiles",
    "smile",
    "canonical_smiles",
    "canonicalsmiles",
    "smiles_canonical",
    "molecule_smiles",
    "moleculesmiles",
    "molecules",
    "molecule",
    "mol",
    "mol_smiles",
    "structure",
}


def _normalize_column_name(column_name: str) -> str:
    return "".join(ch for ch in str(column_name).strip().lower() if ch.isalnum())


def _detect_smiles_feature_column(df: pd.DataFrame) -> str:
    """
    Detect SMILES column in a downloaded dataset.

    Detection strategy:
    1) Alias lookup by common column names.
    2) Fallback heuristic based on RDKit-valid SMILES ratio in sample values.
    """
    if df is None or df.empty:
        return None

    columns = list(df.columns)
    if not columns:
        return None

    # 1) Alias-based match (fast path).
    alias_map = {_normalize_column_name(alias) for alias in _SMILES_COLUMN_ALIASES}
    for col in columns:
        normalized = _normalize_column_name(col)
        if normalized in alias_map:
            return col
        if normalized.endswith("smiles") or normalized.startswith("smiles"):
            return col

    # 2) Heuristic by chemical validity for string-like columns.
    best_column = None
    best_score = 0.0
    best_valid_count = -1
    min_rows_for_eval = 10
    sample_size = 200

    for col in columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        values = series.dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue

        sample = values.head(sample_size).tolist()
        if len(sample) < min_rows_for_eval:
            continue

        valid_count = len(check_chem_valid(sample))
        score = valid_count / len(sample)

        if score > best_score or (score == best_score and valid_count > best_valid_count):
            best_column = col
            best_score = score
            best_valid_count = valid_count

    # Conservative threshold to avoid selecting non-chemical text columns.
    if best_column is not None and best_score >= 0.60:
        return best_column
    return None


def _resolve_or_detect_feature_column(df: pd.DataFrame, requested_feature_column: list) -> list:
    """
    Resolve feature column from request; if not found, auto-detect SMILES column.
    """
    requested = requested_feature_column[0] if requested_feature_column else None

    if requested in df.columns:
        return [requested]

    if requested is not None:
        requested_norm = str(requested).strip().lower()
        for col in df.columns:
            if str(col).strip().lower() == requested_norm:
                return [col]

    detected = _detect_smiles_feature_column(df)
    if detected is not None:
        return [detected]

    raise ValueError(
        f"Feature column '{requested}' not found and auto-detection failed. "
        f"Available columns: {df.columns.tolist()}"
    )

def case_trainer(data:TrainData=Body()):
    # #####FOR TEST####
    # if data.numb_mol>100:
    #     data.numb_mol=100
    # ##################
    #Docking score
    is_evalute_docking = False
    """Main API func for train generative models.

    Args:
        numb_mol (int): Number of molecules to generating.
        model (str, optional): What model need to use.Choose from [lstm,CVAE,TVAE]. Defaults to 'lstm'.
        cuda (bool, optional): Choose cuda usage option. Defaults to False.
        case_ (str): Choose what disease u want to generate molecules for. 

    Returns:
        _type_: _description_
    """
    state = TrainState(state_path='autotrain/utils/state.json')
    try:
        if data.data is not None:
                    df = pd.DataFrame(data.data)
                    data.data_path = f"autotrain/data/{data.case}"
                    if not os.path.isdir(data.data_path):
                        os.mkdir(data.data_path)
                    data.data_path = data.data_path + '/data.csv'
                    df = df.dropna()
                    df = df[df[data.feature_column[0]].str.len()<200]
                    df.to_csv(data.data_path)
                    state.gen_model_upd_data(case=data.case,data_path=data.data_path)
        #CASE = 'CYK'
        # train_data = 'docked_data_for_train/data_cyk_short.csv'
        # conditions = ['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
        test_mode = False
        
    
        if data.fine_tune==True:
            load_weights = 'autotrain/many_prop_CVAE/Alzheimer_1_prop/weights'
            load_weights_fields = 'autotrain/many_prop_CVAE/Alzheimer_1_prop/weights'
            data.new_vocab = False
        else:
            load_weights=None
            load_weights_fields = None
            data.new_vocab = True
        # if state(CASE) is None:#Check if case exist
        #     state.add_new_case(CASE,rewrite=False)
        
        if state(data.case,'ml') is None:
            print(f"{data.case} is not exist! Train ML model before")
            state.gen_model_upd_status(case=data.case,status=3)
            return 0

        use_cond2dec = False
        main(epochs=data.epochs,
            conditions = state(data.case,'ml')['target_column'],
            case=data.case, 
            server_dir = f'autotrain/train_{data.case}',
            data_path_with_conds = data.data_path,
            test_mode=test_mode,
            state=state,
            url=data.url,
            n_samples = data.n_samples,
            load_weights=load_weights,
            load_weights_fields = load_weights_fields,
            use_cond2dec=use_cond2dec,
            new_vocab= data.new_vocab,
            ml_model_url=os.getenv('ML_MOLS_MODEL_APP_URL'))
    except Exception as e:
        print(e)
        state.gen_model_upd_status(case=data.case,error=str(e))

def gan_case_trainer(data:TrainData=Body()):
    # #####FOR TEST####
    # if data.numb_mol>100:
    #     data.numb_mol=100
    # ##################
    #Docking score
    is_evalute_docking = False
    """Main API func for train generative models.

    Args:
        numb_mol (int): Number of molecules to generating.
        model (str, optional): What model need to use.Choose from [lstm,CVAE,TVAE]. Defaults to 'lstm'.
        cuda (bool, optional): Choose cuda usage option. Defaults to False.
        case_ (str): Choose what disease u want to generate molecules for. 

    Returns:
        _type_: _description_
    """
    state = TrainState(state_path='autotrain/utils/state.json')
    try:
        if data.data is not None:
                    df = pd.DataFrame(data.data)
                    data.data_path = f"autotrain/data/{data.case}"
                    if not os.path.isdir(data.data_path):
                        os.mkdir(data.data_path)
                    data.data_path = data.data_path + '/data.csv'
                    df = df.dropna()
                    #df = df[df[data.feature_column[0]].str.len()<200]
                    df.to_csv(data.data_path)
                    state.gen_model_upd_data(case=data.case,data_path=data.data_path,feature_column=data.feature_column)
        #CASE = 'CYK'
        # train_data = 'docked_data_for_train/data_cyk_short.csv'
        # conditions = ['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
        test_mode = False
        
    
        # if data.fine_tune==True:
        #     load_weights = 'GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl'
        #     load_weights_fields = 'GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl'
        #     data.new_vocab = False
        # else:
        #     load_weights=None
        #     load_weights_fields = None
        #     data.new_vocab = True
        # if state(CASE) is None:#Check if case exist
        #     state.add_new_case(CASE,rewrite=False)
        
        if state(data.case,'ml') is None:
            print(f"{data.case} is not exist! Train ML model before")
            state.gen_model_upd_status(case=data.case,status=3)
            return 0

        auto_train(data.case,
                   path_ds=data.data_path,
                   fine_tune=data.fine_tune,
                   state=state,
                   feature_column=data.feature_column,
                   steps=data.epochs)

    except Exception as e:
        print(e)
        state.gen_model_upd_status(case=data.case,error=str(e))

def gan_case_trainer_s3(data:TrainDataS3=Body()):
    # #####FOR TEST####
    # if data.numb_mol>100:
    #     data.numb_mol=100
    # ##################
    #Docking score
    is_evalute_docking = False
    """GAN trainer that fetches the train dataset either from an S3 object
    (`data.s3_key`) or from an arbitrary HTTP(S) URL (`data.data_url`, e.g.
    an S3 presigned URL), then fine-tunes the GAN and optionally uploads the
    trained weights back to S3."""
    state = TrainState(state_path='autotrain/utils/state.json')
    try:
        if data.case is None:
            raise ValueError("`case` is required.")
        data.case = data.case.strip()
        if not data.case:
            raise ValueError("`case` must not be empty.")
        data_url = (data.data_url or "").strip() or None
        s3_key = (data.s3_key or "").strip() or None
        if not data_url and not s3_key:
            raise ValueError(
                "Either `data_url` (HTTP(S) URL of the train CSV, e.g. a presigned URL) "
                "or `s3_key` is required."
            )
        feature_column = _resolve_feature_columns(data)

        # Ensure case exists in state before any updates/status writes.
        if state(data.case) is None:
            state.add_new_case(case_name=data.case, rewrite=False)

        if not data.data_path:
            data.data_path = f"autotrain/data/{data.case}/data.csv"
        data.data_path = data.data_path.replace("\\", "/")
        local_dir = os.path.dirname(data.data_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        # Resolve S3 credentials. They're needed for the S3-key download path
        # AND for uploading trained weights afterwards. When `data_url` is set
        # and `save_trained_data_to_sync_server` is False, S3 creds are not
        # mandatory — we degrade gracefully in that case.
        endpoint_url = data.endpoint_url or data.s3_endpoint_url or os.getenv("ENDPOINT_URL")
        access_key = data.access_key or os.getenv("ACCESS_KEY")
        secret_key = data.secret_key or os.getenv("SECRET_KEY")
        bucket_name = data.bucket_name or data.s3_bucket or os.getenv("BUCKET_NAME")

        s3_service: Optional[S3BucketService] = None
        if all([endpoint_url, access_key, secret_key, bucket_name]):
            s3_service = S3BucketService(
                endpoint=endpoint_url,
                access_key=access_key,
                secret_key=secret_key,
                bucket_name=bucket_name,
            )

        # ---- Dataset download ----
        if data_url:
            # HTTP/HTTPS path — fetch the CSV directly. Works for S3 presigned
            # URLs, arbitrary public URLs, etc. No S3 client involved.
            print(f"Downloading training CSV via HTTP -> {data.data_path}")
            timeout_s = int(os.getenv("TRAIN_DATA_HTTP_TIMEOUT_S", "1800"))
            with requests.get(data_url, stream=True, timeout=timeout_s) as resp:
                resp.raise_for_status()
                with open(data.data_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            fh.write(chunk)
            print(f"Downloaded {os.path.getsize(data.data_path)} bytes from URL")
        else:
            if s3_service is None:
                missing = [
                    name for name, value in (
                        ("ENDPOINT_URL", endpoint_url),
                        ("ACCESS_KEY", access_key),
                        ("SECRET_KEY", secret_key),
                        ("BUCKET_NAME", bucket_name),
                    ) if not value
                ]
                raise ValueError(
                    "Cannot download training CSV by `s3_key`: missing S3 credentials. "
                    f"Provide them via env or payload (missing: {missing}). "
                    "Alternatively pass `data_url` with a presigned HTTP URL."
                )
            s3_service.download_image_from_s3(s3_key=s3_key, local_path=data.data_path)

        # Keep behavior close to gan_case_trainer: sanitize and save local csv before training.
        df = pd.read_csv(data.data_path).dropna()
        requested_feature_name = None
        if isinstance(data.feature_column, list) and data.feature_column:
            requested_feature_name = data.feature_column[0]
        elif isinstance(data.feature_column, str):
            requested_feature_name = data.feature_column
        feature_column = _resolve_or_detect_feature_column(df, feature_column)
        if feature_column[0] != requested_feature_name:
            print(
                f"Auto-detected feature column for case '{data.case}': "
                f"'{feature_column[0]}' (requested: '{requested_feature_name}')"
            )
        df.to_csv(data.data_path, index=False)
        state.gen_model_upd_data(
            case=data.case,
            data_path=data.data_path,
            feature_column=feature_column
        )

        # if state(data.case,'ml') is None:
        #     print(f"{data.case} is not exist! Train ML model before")
        #     state.gen_model_upd_status(case=data.case,status=3)
        #     return 0

        auto_train(
            data.case,
            path_ds=data.data_path,
            fine_tune=data.fine_tune,
            state=state,
            feature_column=feature_column,
            steps=data.epochs
        )

        if getattr(data, "save_trained_data_to_sync_server", True):
            local_weights_folder = f"autotrain/GAN_weights/train_GAN_{data.case}"
            try:
                if s3_service is None:
                    print(
                        "GAN weights upload to S3 skipped: S3 credentials are not "
                        "configured. Training completed; weights are kept locally."
                    )
                else:
                    upload_status = upload_gan_weights_folder_to_s3(
                        case=data.case,
                        local_folder=local_weights_folder,
                        s3_service=s3_service,
                    )
                    print(f"GAN weights upload to S3: {upload_status}")
            except Exception as upload_exc:
                # Upload failure must not invalidate a successful training run.
                print(f"GAN weights upload to S3 failed: {upload_exc}")

    except Exception as e:
        print(e)
        if getattr(data, "case", None) and state(data.case) is not None:
            state.gen_model_upd_status(case=data.case,error=str(e))

def gan_auto_generator(data:GenData=Body()):
    state = TrainState(state_path='autotrain/utils/state.json')

    raw_case = (data.case_ or "").strip()
    is_case_request = raw_case.lower() not in _GENERIC_GAN_CASE_TOKENS

    weights_candidates: List[str] = []
    weights_status: Dict[str, Any] = {"status": "ok", "case": raw_case or None}

    if is_case_request:
        # Strict case mode: weights MUST be the case-specific ones; if missing
        # locally, try to download them from S3. Never silently fall back to
        # generic GAN — the caller asked for a specific trained case.
        weights_status = ensure_gan_weights_available(raw_case)
        if weights_status.get("status") != "ok":
            return weights_status
        weights_candidates.append(weights_status["local_path"])
    else:
        # Generic mode (no case requested): use the bundled fallback weights
        # that ship with the image / are pulled from HF at build time.
        gen_state = state(raw_case, 'gen') if raw_case else None
        if gen_state is not None and gen_state.get('weights_path'):
            weights_candidates.append(os.path.join(gen_state['weights_path'], 'gan_weights.pkl'))
        weights_candidates.extend([
            os.path.join('autotrain', 'many_prop_CVAE', 'weights_8p_alzhmr', 'gan_weights.pkl'),
            os.path.join('GAN', 'gan_lstm_refactoring', 'weights', 'v4_gan_mol_124_0.0003_8k.pkl')
        ])
        weights_candidates = list(dict.fromkeys(weights_candidates))

    gan_mol = None
    load_errors: List[str] = []
    loaded_from: Optional[str] = None
    for weights_path in weights_candidates:
        if not os.path.isfile(weights_path):
            continue
        try:
            with open(weights_path, "rb") as f:
                gan_mol = pickle.load(f)
            loaded_from = weights_path
            print(f'Loaded GAN weights from: {weights_path}')
            break
        except Exception as e:
            load_errors.append(f'{weights_path}: {e}')

    if gan_mol is None:
        return {
            "status": "weights_load_failed",
            "case": raw_case or None,
            "message": (
                "Could not load any GAN weights for this generation request. "
                f"Tried: {weights_candidates}. Load errors: {load_errors}."
            ),
            "weights_candidates": weights_candidates,
            "load_errors": load_errors,
            "weights_status": weights_status,
        }
    
    gan_mol.eval()
    calc_props = state()["Calculateble properties"]
    samples = gan_mol.generate_n(data.numb_mol)
    valid_mols = calc_props['Validity'](samples)
    if not isinstance(valid_mols, list):
        valid_mols = list(valid_mols) if valid_mols is not None else []

    valid_count = len(valid_mols)
    generated_count = len(samples) if samples is not None else 0
    unique_count = len(set(valid_mols))
    duplicates = 1 - (unique_count / valid_count) if valid_count else 0.0
    validity = (valid_count / generated_count) if generated_count else 0.0

    props_for_calc = [name for name in state.show_calculateble_propreties() if name != "Validity"]
    result = {'Smiles': valid_mols}
    for key in props_for_calc:
        raw_values = calc_props[key](valid_mols) if valid_count else []
        result[key] = _normalize_prop_values(raw_values, valid_count)

    result['Validity'] = [validity] * valid_count
    result["Duplicates"] = [duplicates] * valid_count
    # print(os.getenv('ML_MOLS_MODEL_APP_URL'))
    # try:
    #     if state(data.case_,'ml')['status'] == 'Trained':
    #         ml_props = predict_smiles(valid_mols,data.case_,url=os.getenv('ML_MOLS_MODEL_APP_URL'))
    #         for key,value in ml_props.items():
    #             props[key]=value
    # except:
    #     ml_props = predict_smiles(valid_mols,'Base',url=os.getenv('ML_MOLS_MODEL_APP_URL'))
    #     for key,value in ml_props.items():
    #         props[key]=value
    return result
    #return samples


def auto_generator(data:TrainData=Body()):
     
    state = TrainState(state_path='autotrain/utils/state.json')
    if state(data.case,'gen')["status"] == "Trained":
        use_cond2dec = False
        gen_dict = main_generate(epochs=data.epochs,
                conditions = state(data.case,'ml')['target_column'],
                case=data.case, 
                server_dir = f'autotrain/train_{data.case}',
                test_mode=False,
                state=state,
                url=data.url,
                ml_model_url= data.url,
                n_samples = data.n_samples,
                load_weights=state(data.case,'gen')['weights_path'],
                load_weights_fields = state(data.case,'gen')['weights_path'],
                use_cond2dec=use_cond2dec,
                new_vocab= data.new_vocab,
                batchsize=data.batchsize)
        return gen_dict
    else:
        print('Case is not trained!')
        return 0

def case_generator(data:GenData=Body()):
    #####FOR TEST####
    if data.numb_mol>100:
        data.numb_mol=100
    ##################
    #Docking score
    is_evalute_docking = False
    """Main API func for generation any chem case.

    Args:
        numb_mol (int): Number of molecules to generating.
        model (str, optional): What model need to use.Choose from [lstm,CVAE,TVAE]. Defaults to 'lstm'.
        cuda (bool, optional): Choose cuda usage option. Defaults to False.
        case_ (str): Choose what disease u want to generate molecules for. 

    Returns:
        _type_: _description_
    """
    dis_case = cases[data.case_]
    print(data)
    init_mol_numb = data.numb_mol
    #data = data
    mol_list = []
    mol_list = _gen_n(data)
    mol_list = [i for i in mol_list if '.' not in i]
    mol_list = check_chem_valid(mol_list)
    mol_list,diversity = check_novelty_chembl(mol_list,train_data_path=dis_case['train_data_path'])
    
    #Check if generated not enouth molecules 
    while len(mol_list)<init_mol_numb:

        #init_mol_numb = data.numb_mol-len(mol_list)
        data.numb_mol = init_mol_numb-len(mol_list)
        mol_list_temp = _gen_n(data)
        mol_list = [i for i in mol_list if '.' not in i]
        mol_list += check_chem_valid(mol_list_temp)
        mol_list,diversity = check_novelty_chembl(mol_list,train_data_path=dis_case['train_data_path'])
    
    path = dis_case['docking_path'] #Choose path to docking file for current case

    #Calculate metrics
    df = pd.DataFrame(data=mol_list,columns=['Molecules'])

    #Docking Score
    if is_evalute_docking:
        d_s = docking_list(smiles=mol_list,path_receptor_pdb=path)
        df['Docking score'] = d_s

    if dis_case['anti_docking_path'] is not None:
         for i in dis_case['anti_docking_path'].keys():
              anti_path = dis_case['anti_docking_path'][i]
              d_s_anti_target = docking_list(smiles=mol_list,path_receptor_pdb=anti_path)
              col_name = f'Anti Docking score for {i}'
              df[col_name] = d_s_anti_target

    df['QED'] = df['Molecules'].apply(eval_qed)
    df['Synthetic Accessibility'] = df['Molecules'].apply(eval_sa) # SA
    df['PAINS'] = df['Molecules'].apply(eval_P_S_G,type_n='PAINS')
    df['SureChEMBL']= df['Molecules'].apply(eval_P_S_G,type_n='SureChEMBL')
    df['Glaxo'] = df['Molecules'].apply(eval_P_S_G,type_n='Glaxo')
    #df['Diversity'] = diversity
    df['Brenk'] = df['Molecules'].apply(check_brenk)
    df['BBB'] = eval_bbb(list(df['Molecules']))

    #Choose property IC50 OR KI for cases
    if data.case_ == 'Dslpdm':
        ki_values = dis_case['KI'](mol_list)
        df['KI'] = ki_values
    if dis_case['IC50'] is not None:
        ic50_ki_values = dis_case['IC50'](mol_list)
        df['IC50'] = ic50_ki_values
    df = df.round(2)
    return {i:df[i].to_list() for i in df.columns}
    

def _gen_n(data):
        """Function for generating molecules for choosen case
        """
       
        mol_list = []
        dis_case = cases[data.case_]
        if data.case_ == 'Alzhmr':
            df = dis_case['generative_model'](opt=dis_case['opt'],
                n_samples=data.numb_mol,
                path_to_save='',
                cuda=data.cuda,
                save=False,
                spec_conds = [[-9,-12],[0.8,1],[0,2.99],[0,0],[0,0],[0,0],[0,0],[0,0]],
                mean_=data.mean_,std_=data.std_)
            mol_list += df
        elif data.case_ == 'TBLET':
            df = dis_case['generative_model'](opt=dis_case['opt'],
                n_samples=data.numb_mol,
                path_to_save='',
                cuda=data.cuda,
                save=False,
                spec_conds = [[-7,-11],[0.7,1],[0,2.99],[0,0],[0,0],[0,0],[0,0],[1,1]],
                mean_=data.mean_,std_=data.std_)
            mol_list += df
        elif data.case_ == 'RNDM':
             df = dis_case['generative_model'](data.numb_mol)
             mol_list += df
        else:
            df = dis_case['generative_model'](opt=dis_case['opt'],
                n_samples=data.numb_mol,
                path_to_save='',
                cuda=data.cuda,
                save=False,
                spec_conds = [[-7,-12],[0.6,1],[0,2.99],[0,0],[0,0],[0,0],[0,0],[1,1]],
                mean_=data.mean_,std_=data.std_)
            mol_list += df
        return mol_list

docking_paths = {'Alzhmr' : 'autodock_vina_python3/data/4j1r.pdb',
                 'Sklrz':'autodock_vina_python3/data/target_BTK.pdb'}

opt = parsing()
parser = configurate_parser(load_weights="autotrain/many_prop_CVAE/weights_8p_alzhmr",#weights_33k_trained
                            load_weights_fields = "autotrain/many_prop_CVAE/weights_8p_alzhmr",
                            cuda=False,
                            save_folder_name='alzh_gen_mols',
                            new_vocab = False,
                            import_path = import_path,
                            cond_dim=8
                                )
opt_Alz_multi = parser.parse_args()
opt_sklrz = parser.parse_args()
opt_sklrz.load_weights = "autotrain/many_prop_CVAE/weights_8p_sklrz"
opt_sklrz.load_weights_fields = "autotrain/many_prop_CVAE/weights_8p_sklrz"

opt_cnsr = parser.parse_args()
opt_cnsr.load_weights = 'autotrain/many_prop_CVAE/weights_8p_cnsr'
opt_cnsr.load_weights_fields = 'autotrain/many_prop_CVAE/weights_8p_cnsr'


opt_tablet = parser.parse_args()
opt_tablet.load_weights = 'autotrain/many_prop_CVAE/weights_8p_tablet'
opt_tablet.load_weights_fields = 'autotrain/many_prop_CVAE/weights_8p_tablet'

opt_park = parser.parse_args()
opt_park.load_weights = 'autotrain/many_prop_CVAE/weights_parkinson'
opt_park.load_weights_fields = 'autotrain/many_prop_CVAE/weights_parkinson'

opt_dislip = parser.parse_args()
opt_dislip.load_weights = 'autotrain/many_prop_CVAE/weights_dislip'
opt_dislip.load_weights_fields = 'autotrain/many_prop_CVAE/weights_dislip'

# Case information
cases = {'Alzhmr' : 
         {'docking_path' : 'autodock_vina_python3/data/4j1r.pdb',
        'generative_model':multi_generator,
        'opt':opt_Alz_multi,
        'IC50':eval_ic_50_alzheimer,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'docked_data_for_train/data_4j1r.csv'
                     },
        #TODO update next cases
        'Sklrz':
        {'docking_path' :'autodock_vina_python3/data/skleroz/target_BTK.pdb',
         'generative_model':multi_generator,
         'opt':opt_sklrz,
         'IC50':eval_ic_50_sklrz,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'docked_data_for_train/data_5vfi.csv'#{'BMX':'autodock_vina_python3/data/skleroz/BMX_8x2a_protein.pdb'}
         },

         'Prkns':
        {'docking_path' :'autodock_vina_python3/data/parkinson/tyrosine_protein_kinase_ABL.pdb',
         'generative_model':multi_generator,
         'opt':opt_park,
         'IC50':parkenson_predict_ic50,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'docked_data_for_train/data_ABL.csv'
         },

         'Cnsr':
        {'docking_path' :'autodock_vina_python3/data/Canser/8afb_protein.pdb',
         'generative_model':multi_generator,
         'opt':opt_cnsr,
         'IC50':eval_ic_50_cancer,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'docked_data_for_train/data_8afb.csv'#{'NRAS':'autodock_vina_python3/data/Canser/NRAS_3con_protein.pdb',
                              #'HRAS':'autodock_vina_python3/data/Canser/HRAS_3k8y_protein.pdb'}
         },

         'Dslpdm':
        {'docking_path' :'autodock_vina_python3/data/dislipidemia/ATP_citrate_synthase.pdb',
         'generative_model':multi_generator,
         'opt':opt_dislip,
         'IC50':dyslip_predict_ic50,
         'KI':dyslip_predict_ki,
         'anti_docking_path':None,
         'train_data_path':'docked_data_for_train/data_ATP.csv'
         },

         'TBLET':
        {'docking_path' :'autodock_vina_python3/data/Signal_Transducer_and_Activator_of_Transcription_3.pdb',
         'generative_model':multi_generator,
         'opt':opt_tablet,
         'IC50':drug_res_predict_ic50,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'docked_data_for_train/data_stat3.csv'
         },
         'RNDM':
         {'docking_path' :'autodock_vina_python3/data/Signal_Transducer_and_Activator_of_Transcription_3.pdb',
         'generative_model':generate,
         'opt':opt,
         'IC50':None,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'docked_data_for_train/data_stat3.csv'
         },
        }

