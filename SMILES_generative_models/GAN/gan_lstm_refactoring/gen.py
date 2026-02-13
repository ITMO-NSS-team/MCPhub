import sys
import os
from pathlib import Path

repo_root = Path(__file__).resolve()
for parent in [repo_root] + list(repo_root.parents):
    if (parent / "pyproject.toml").exists():
        repo_root = parent
        break
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import pickle as pi
import pandas as pd
import random
import GAN.gan_lstm_refactoring.scripts.model as KOSTIL_FOR_PICKL
sys.modules['scripts.model'] = KOSTIL_FOR_PICKL
sys.modules['scripts.utils'] = KOSTIL_FOR_PICKL
sys.modules['scripts.layers'] = KOSTIL_FOR_PICKL
sys.modules['scripts.tokenizer'] = KOSTIL_FOR_PICKL


def generate(n):
    gan_mol = pi.load(open('GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl', 'rb'))
    
    # generate smiles molecules
    smiles_list = gan_mol.generate_n(n)
    return smiles_list
if __name__=='__main__':
    print(generate(4))
    
