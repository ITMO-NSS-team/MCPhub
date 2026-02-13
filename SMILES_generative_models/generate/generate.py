import sys
import os
import shutil
from pathlib import Path

import_path = os.path.dirname(os.path.abspath(__file__))
repo_root = Path(import_path).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# sys.path.append(import_path)
# sys.path.append(os.getcwd())
import lightgbm

from ic50_classifire_model.read_ic50 import Ic50
from inference import generate_alzh
from Process import *
import argparse
from Models import get_model
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
#from pipeline.classifier import Classifier
from config import configurate_parser

def generator(opt,
              n_samples=10000,
              path_to_save:str='alzh_gen_mols',
              spec_conds=[1],
              save:bool=True,
              cuda:bool=True,
              path_to_val_model:str='Sber/ic50_classifire_model/kinase_inhib.pkl',
              weights_path:str = "Sber/generate_alzh/weights",
              index = '0',
              mean_=0,std_=1,
              
              ):
    '''
    The generator function generates the specified number of molecules.
        n_samples - number of molecules to be generated.
        path_to_save - suffix to the file path to save the molecules.
    It is necessary to give a name to the file for generation.
        save - whether to save the generated molecules? True/False
        spec_conds - None for random assignment of physical properties/
    list for specifying molecules of interest. Example: [1,1,0].
    '''
    
    opt = opt
    print(opt.load_weights)
    opt.device = 'cuda' if cuda else 'cpu'
    opt.path_script = import_path+f'/{path_to_save}/'
    if not os.path.isdir(opt.path_script):
        os.mkdir(opt.path_script )
    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    opt.classifire = Ic50
    df = generate_alzh(opt=opt,
             model=model,
             SRC=SRC,
             TRG=TRG,
             n_samples=n_samples,
             spec_conds=spec_conds,
             save=save,
             shift_path=index,
             mean_=mean_,std_=std_,
             path_to_val_model=path_to_val_model)
    return df


def shut_copy(src,trg):
    shutil.copy(src+'/SRC.pkl',trg+'/SRC.pkl')
    shutil.copy(src+'/TRG.pkl',trg+'/TRG.pkl')
    shutil.copy(src+'/toklen_list.csv',trg+'/toklen_list.csv')

if __name__ == '__main__':
    
    parser = configurate_parser(load_weights="Sber_Alzheimer/train_cVAE_sber_altz_docking/weights",
                            load_weights_fields = "Sber_Alzheimer/train_cVAE_sber_altz_docking/weights",
                            cuda=False,
                            save_folder_name='alzh_gen_mols',
                            new_vocab = False
                                )
    opt_Alz = parser.parse_args()
    generator(opt=opt_Alz,n_samples=10,spec_conds=[0])
