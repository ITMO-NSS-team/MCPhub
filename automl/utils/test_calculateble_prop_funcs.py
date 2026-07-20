"""Guards for the calculable-property registry.

`config` is hand-wired dict -> function, and it has been mis-wired before:
`"H-bond Acceptors"` pointed at `aromatic_rings` and `"Aromatic Rings"` at
`h_bound_acceptors`, so both columns silently carried each other's values in
every generated/predicted CSV. The functions were correct; only the mapping was
swapped, which is why nothing raised.

The registry is duplicated in three places (this one, plus
`SMILES_generative_models/autotrain/utils/` and
`SMILES_generative_models/train_data/utils/`). `test_all_registry_copies_agree`
below checks them against each other, so a fix applied to only one copy fails
here rather than silently making the two MCP servers disagree about what a
column name means.

Run:  python -m pytest automl/utils/test_calculateble_prop_funcs.py
"""

import contextlib
import importlib.util
import os
from pathlib import Path

import pytest

from calculateble_prop_funcs import config

# (smiles, name, aromatic_rings, h_bond_acceptors, h_bond_donors, rotatable_bonds)
# Values are RDKit ground truth; picked so the two swapped properties differ,
# which a symmetric molecule would not catch.
KNOWN = [
    ("c1ccccc1", "benzene", 1, 0, 0, 0),
    ("CCO", "ethanol", 0, 1, 1, 0),
    ("CC(=O)Oc1ccccc1C(=O)O", "aspirin", 1, 3, 1, 2),
    ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "caffeine", 2, 6, 0, 0),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATIVE_ROOT = _REPO_ROOT / "SMILES_generative_models"

_REGISTRY_COPIES = (
    _GENERATIVE_ROOT / "autotrain" / "utils" / "calculateble_prop_funcs.py",
    _GENERATIVE_ROOT / "train_data" / "utils" / "calculateble_prop_funcs.py",
)


@contextlib.contextmanager
def _cwd(path):
    """Run inside `path`.

    Both generative copies read the PAINS/Glaxo/SureChEMBL alert table via the
    hardcoded relative path 'autotrain/utils/alert_collections.csv', so those
    three properties only resolve when the process runs from the
    SMILES_generative_models root — which is what api.sh arranges (`cd
    "$SCRIPT_DIR"`). Reproduce that here instead of skipping the comparison.
    (The automl copy derives the path from __file__ and is CWD-independent.)
    """
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@pytest.mark.parametrize("smiles,name,aromatic,acceptors,donors,rotatable", KNOWN)
def test_registry_keys_map_to_matching_property(
    smiles, name, aromatic, acceptors, donors, rotatable
):
    """Each key must return the property its NAME promises."""
    assert config["Aromatic Rings"]([smiles]) == [aromatic], f"Aromatic Rings wrong for {name}"
    assert config["H-bond Acceptors"]([smiles]) == [acceptors], f"H-bond Acceptors wrong for {name}"
    assert config["H-bond Donors"]([smiles]) == [donors], f"H-bond Donors wrong for {name}"
    assert config["Rotatable Bonds"]([smiles]) == [rotatable], f"Rotatable Bonds wrong for {name}"


def test_aromatic_rings_and_acceptors_are_not_swapped():
    """The exact regression: benzene has rings but no acceptors, ethanol the reverse.

    If the two keys are swapped again, benzene reports 0 rings / 1 acceptor and
    this fails — whereas a molecule where both counts coincide would not.
    """
    smiles = ["c1ccccc1", "CCO"]
    assert config["Aromatic Rings"](smiles) == [1, 0]
    assert config["H-bond Acceptors"](smiles) == [0, 1]


@pytest.mark.parametrize("copy_path", _REGISTRY_COPIES, ids=lambda p: p.parts[-3])
def test_all_registry_copies_agree(copy_path):
    """Every duplicated registry must produce identical values for every key.

    The two MCP servers ship as separate images off separate copies of this
    file; if one is fixed and another is not, the same column name means two
    different properties depending on which server produced the CSV.
    """
    if not copy_path.is_file():
        pytest.skip(f"registry copy not present: {copy_path}")

    spec = importlib.util.spec_from_file_location(f"copy_{copy_path.parts[-3]}", copy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    smiles = [s for s, *_ in KNOWN]
    assert set(module.config) == set(config), f"{copy_path} has a different key set"

    with _cwd(_GENERATIVE_ROOT):
        other = {key: list(func(smiles)) for key, func in module.config.items()}

    for key in config:
        assert other[key] == list(config[key](smiles)), (
            f"{copy_path} disagrees with automl on '{key}' — "
            "the two MCP servers would emit different values under one column name"
        )
