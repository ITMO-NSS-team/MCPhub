from typing import List 
from fastapi import Body
import os
import sys
import types
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
        data_path:str = None
        feature_column:list = ['Smiles']
        # Backward compatibility with common typo/casing in clients.
        Future_column:list = None
        future_column:list = None
        fine_tune:bool = True
        epochs:int = 10
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
    """Copy of GAN trainer that downloads train dataset from S3 before fit."""
    state = TrainState(state_path='autotrain/utils/state.json')
    try:
        if data.case is None:
            raise ValueError("`case` is required.")
        data.case = data.case.strip()
        if not data.case:
            raise ValueError("`case` must not be empty.")
        if not data.s3_key:
            raise ValueError("`s3_key` is required.")
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

        endpoint_url = data.endpoint_url or data.s3_endpoint_url or os.getenv("ENDPOINT_URL")
        access_key = data.access_key or os.getenv("ACCESS_KEY")
        secret_key = data.secret_key or os.getenv("SECRET_KEY")
        bucket_name = data.bucket_name or data.s3_bucket or os.getenv("BUCKET_NAME")

        if not endpoint_url:
            raise ValueError("S3 endpoint is empty. Set `endpoint_url` or env `ENDPOINT_URL`.")
        if not access_key:
            raise ValueError("S3 access key is empty. Set `access_key` or env `ACCESS_KEY`.")
        if not secret_key:
            raise ValueError("S3 secret key is empty. Set `secret_key` or env `SECRET_KEY`.")
        if not bucket_name:
            raise ValueError("S3 bucket is empty. Set `bucket_name` or env `BUCKET_NAME`.")

        s3_service = S3BucketService(
            endpoint=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name
        )
        s3_service.download_image_from_s3(s3_key=data.s3_key, local_path=data.data_path)

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

    except Exception as e:
        print(e)
        if getattr(data, "case", None) and state(data.case) is not None:
            state.gen_model_upd_status(case=data.case,error=str(e))

def gan_auto_generator(data:GenData=Body()):
    state = TrainState(state_path='autotrain/utils/state.json')

    gen_state = state(data.case_, 'gen')
    weights_candidates = []
    if gen_state is not None and gen_state.get('weights_path'):
        weights_candidates.append(os.path.join(gen_state['weights_path'], 'gan_weights.pkl'))
    weights_candidates.extend([
        os.path.join('autotrain', 'many_prop_CVAE', 'weights_8p_alzhmr', 'gan_weights.pkl'),
        os.path.join('GAN', 'gan_lstm_refactoring', 'weights', 'v4_gan_mol_124_0.0003_8k.pkl')
    ])
    weights_candidates = list(dict.fromkeys(weights_candidates))

    gan_mol = None
    load_errors = []
    for weights_path in weights_candidates:
        if not os.path.isfile(weights_path):
            continue
        try:
            with open(weights_path, "rb") as f:
                gan_mol = pickle.load(f)
            print(f'Loaded GAN weights from: {weights_path}')
            break
        except Exception as e:
            load_errors.append(f'{weights_path}: {e}')

    if gan_mol is None:
        raise FileNotFoundError(
            f'Could not load GAN weights. Checked: {weights_candidates}. '
            f'Load errors: {load_errors}'
        )
    
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

