import json
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel
import os
from autotrain.utils.calculateble_prop_funcs import config
from autotrain.utils.state_s3 import download_state_file, upload_state_file


VALID_STATUSES = ("Not Trained", "Training", "Trained", "Failed")


class BaseState(BaseModel):
    status:str = None
    weights_path:str = {"regression" : None, "classification" : None}
    arch_type:str = None
    case_name: str = None
    base_state:dict = {"description" : None,
                       "generative_models": {"data_path":None,"feature_column":None,"target_column":None,"problem":None,'status':status,'weights_path':None,'arch_type':arch_type,"metric":None,"error":None,"error_at":None},
                        "ml_models":{"data_path":None,"feature_column":None,"target_column":None,'status':status,'weights_path' : weights_path,"metric":None,
                            "Predictable properties" : {}, "error": None, "error_at": None
                                  }
                        }
class TrainState:
    """A class representing a state with information about the available trained 
    models for a multi-agent system. The main essence of its operation is the ability to load
      and save a dictionary with a state, using it to analyze the availability of 
      trained (training) generative or predictive models.

    Returns:
        _type_: _description_
    """
    defult_parameters = BaseState()

    def __init__(self, state_path: str = None, sync_with_s3: bool = True):
        # The S3 key for state.json is a hardcoded contract shared with the
        # AutoML MCP server (see autotrain/utils/state_s3.STATE_S3_KEY). It
        # is NOT configurable per-instance / per-env: any drift between the
        # two servers' keys would split the state into two independent
        # files, silently dropping cases registered on the "other" server.
        self.state_path = state_path
        self.sync_with_s3 = sync_with_s3

        if self.state_path is None:
            self.state_path = r'autotrain/utils/state.json'

        state_dir = os.path.dirname(self.state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

        if self.sync_with_s3:
            self.sync_state_from_s3(raise_on_error=False)

        if os.path.isfile(self.state_path):
            self.current_state = self.__load_state()
        else:
            self.current_state = {"Calculateble properties" : config}
            self.__save_state()

    def __call__(self, case:str = None, 
                 model:str = None,
                   *args,
                     **kwargs):
        """Return full state if take no arguments.

        Args:
            case (str, optional): Task name. If defined - then return one state of task, named ("case"). Defaults to None.
            model (str, optional): Model state. May be choose from "ml" or "gen" to return only state for generative or ML models.
             Must be specified with argument "case". Defaults to 'ml'.

        Returns:
            dict: State with model information.
        """
        if case is None:
            return self.current_state
        elif case in self.current_state.keys():
            if model=='ml':
                return self.current_state[case]["ml_models"]
            elif model=='gen':
                return self.current_state[case]["generative_models"]
            else: 
                return self.current_state[case]
        else:
            print('Case do not exist in current State!')
            return None
    
    def add_new_case(self,
                     case_name:str,
                     rewrite=False,
                     description : str = 'Unknown case',
                     **kwargs):
        """Add new case object to state dict with case name and defult parameters. After adding update and save state.

        Args:
            case_name (str): User specified case name.
            rewrite (bool, optional): Indicates whether or not to overwrite an already existing state with this state. Defaults to False.

        Returns:
            None:
        """
        if case_name in self.current_state.keys() and not rewrite:
            print(f'Case already exist! Change name for new case, or use exist case state named {case_name}!')
            print(f"Now using case named '{case_name}' - Case Description: {self.current_state[case_name]['description']}")
            return None
        
        self.current_state[case_name] = self.defult_parameters.base_state
        self.current_state[case_name]['description'] = description
        self.__save_state()

    def gen_model_upd_data(self,
                           case:str,
                           data_path:str=None,
                           feature_column:List[str] = None,
                           target_column:List[str] = None
                           ):
        """Necessary to update the parameters of a generative model state with the specified name ("case").
          The path to the training data, the type of the problem, and the columns
            with attributes and target parameters in the dataset can be specified.
            All or only any one parameters specified in the function call.

        Args:
            case (str): Case name.
            data_path (str, optional): Path to training data. Defaults to None.
            feature_column (str, optional): Name of training data feature column. Defaults to None.
            target_column (str, optional): Name of training data target column. Defaults to None.
            problem (str, optional): Problem ("regression" or "classification"). Defaults to None.
        """
        if data_path is not None:
            self.current_state[case]["generative_models"]["data_path"] = data_path
        if feature_column is not None:
            self.current_state[case]["generative_models"]["feature_column"] = feature_column
        if target_column is not None:
            self.current_state[case]["generative_models"]["target_column"] = target_column
        self.__save_state()

    def ml_model_upd_data(self,
                        case:str,
                        data_path:str=None,
                        feature_column:List[str] = None,
                        target_column:List[str] = None,
                        predictable_properties:dict = None):
        """Necessary to update the parameters of a ML model state with the specified name ("case").
          The path to the training data, the type of the problem, and the columns
            with attributes and target parameters in the dataset can be specified.
            All or only any one parameters specified in the function call.

        Args:
            case (str): Case name.
            data_path (str, optional): Path to training data. Defaults to None.
            feature_column (str, optional): Name of training data feature column. Defaults to None.
            target_column (str, optional): Name of training data target column. Defaults to None.
            predictable_properties (dict, optional): May be give as dict with problem as keys, and propreties as values exmpl:\
                  predictable_properties = {"regression":['LogP','QED'], "classification":['IC50','Num rings']} . Defaults to None.
        """
        if data_path is not None:
            self.current_state[case]["ml_models"]["data_path"] = data_path
        if feature_column is not None:
            self.current_state[case]["ml_models"]["feature_column"] = feature_column
        if target_column is not None:
            self.current_state[case]["ml_models"]["target_column"] = target_column
        if predictable_properties is not None:
            self.current_state[case]["ml_models"]['Predictable properties'] = predictable_properties
        self.__validate_properties(case)
        print(f"Data for ML models training has been updated! \
               \n Current predictable properties and tasks are {self.current_state[case]['ml_models']['Predictable properties']}")
        self.__save_state()

    def ml_model_upd_status(self,
                            case:str,
                            model_weight_path:str = None,
                            metric = None,
                            status:int = None,
                            problem:str = 'regression',
                            error: Optional[str] = None):
        """Update ML training status, weights path, metrics, or error.

        Args:
            case: Case name.
            model_weight_path: Path to trained model weights save folder.
            metric: Value of metric after model training.
            status: 0=Not Trained, 1=Training, 2=Trained, 3=Failed.
            problem: 'regression' or 'classification'.
            error: Optional human-readable error message. When provided,
                writes `error` + `error_at` and forces `status="Failed"` if
                no explicit `status` was supplied. On a successful Trained
                transition, error fields are cleared.
        """
        entry = self.current_state[case]["ml_models"]
        entry.setdefault("error", None)
        entry.setdefault("error_at", None)

        if error is not None:
            entry["error"] = str(error)
            entry["error_at"] = datetime.now(timezone.utc).isoformat()
            if status is None:
                entry["status"] = "Failed"

        if status is not None:
            entry["status"] = VALID_STATUSES[status]
            if VALID_STATUSES[status] == "Trained":
                entry["error"] = None
                entry["error_at"] = None
        elif entry["status"] == "Trained":
            print(f'ML model for task "{case}" already trained!')

        if model_weight_path is not None:
            if not os.path.isdir(model_weight_path):
                os.mkdir(model_weight_path)
        if entry["weights_path"][problem] is None and model_weight_path is not None:
            entry["weights_path"][problem] = model_weight_path
        if metric is not None:
            entry["metric"] = metric
        self.__save_state()

    def gen_model_upd_status(self,
                             case:str,
                             model_weight_path:str = None,
                             metric = None,
                             status:int = None,
                             error: Optional[str] = None):
        """Update generative training status, weights path, metrics, or error.

        Args:
            case: Case name.
            model_weight_path: Path to trained GAN weights folder.
            metric: Value of metric after training.
            status: 0=Not Trained, 1=Training, 2=Trained, 3=Failed.
            error: Optional human-readable error message. When provided,
                writes `error` + `error_at` and forces `status="Failed"`.
                Status is kept clean as an enum; `error` lives next to it
                instead of overwriting it (legacy behavior).
        """
        entry = self.current_state[case]["generative_models"]
        entry.setdefault("error", None)
        entry.setdefault("error_at", None)

        if error is not None:
            entry["error"] = str(error)
            entry["error_at"] = datetime.now(timezone.utc).isoformat()
            entry["status"] = "Failed"
        elif status is not None:
            entry["status"] = VALID_STATUSES[status]
            if VALID_STATUSES[status] == "Trained":
                entry["error"] = None
                entry["error_at"] = None
        else:
            # Legacy auto-progression for callers that pass neither status nor error
            # (auto_train.py / auto_generate.py rely on this).
            if entry["status"] is None:
                entry["status"] = "Training"
            elif entry["status"] == "Training":
                entry["status"] = "Trained"
                entry["error"] = None
                entry["error_at"] = None
            else:
                print(f'Generative model for task "{case}" already trained!')

        if model_weight_path is not None:
            if not os.path.isdir(model_weight_path):
                os.mkdir(model_weight_path)
        if model_weight_path is not None:
            entry["weights_path"] = model_weight_path
        if metric is not None:
            entry["metric"] = metric
        self.__save_state()
    
    def show_calculateble_propreties(self):
        return self.current_state["Calculateble properties"].keys()

    @staticmethod
    def load_state(path:str = r'autotrain/utils/state.json'):
        with open(path) as state_file:
            state = json.load(state_file)
        state["Calculateble properties"] = config
        return state
    
    def save(self,path:str = r'autotrain/utils/state.json'):
        self.state_path = path
        self.__save_state()

    def sync_state_from_s3(self, raise_on_error: bool = True) -> bool:
        if not self.sync_with_s3:
            return False
        try:
            download_state_file(local_path=self.state_path)
            return True
        except Exception as exc:
            if raise_on_error:
                raise
            print(f"Failed to download state from S3: {exc}")
            return False

    def sync_state_to_s3(self, raise_on_error: bool = False) -> bool:
        if not self.sync_with_s3:
            return False
        try:
            upload_state_file(local_path=self.state_path)
            return True
        except Exception as exc:
            if raise_on_error:
                raise
            print(f"Failed to upload state to S3: {exc}")
            return False

    def __save_state(self):
        saving_dict = self.current_state.copy()
        del saving_dict["Calculateble properties"]
        with open(self.state_path, 'w') as state_file:
            json.dump(saving_dict, state_file)
        self.sync_state_to_s3(raise_on_error=False)

    def __load_state(self):
        with open(self.state_path) as state_file:
            state = json.load(state_file)
        state["Calculateble properties"] = config
        return state
    
    def __validate_properties(self, case: str):
            if (
                self.current_state[case]["ml_models"]["feature_column"] is not None
                and self.current_state[case]["ml_models"]["Predictable properties"] != {}
            ):
                temp_predictable_properties = {}
                for task in self.current_state[case]["ml_models"]["Predictable properties"].keys():
                    if self.current_state[case]["ml_models"]["Predictable properties"][task] is not None:
                        temp_predictable_properties[task] = [
                            proper
                            for proper in self.current_state[case]["ml_models"]["Predictable properties"][task]
                            if proper not in self.show_calculateble_propreties() and proper is not None
                        ]
                        if len(temp_predictable_properties[task]) == 0:
                            del temp_predictable_properties[task]
                self.current_state[case]["ml_models"]["Predictable properties"] = temp_predictable_properties

if __name__=='__main__':
    task = 'Brain_cancer_test'
    feature_column=['canonical_smiles']
    target_column=['docking_score','canonical_smiles','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
    regression_props = ['LogP','docking_score',"Synthetic Accessibility",'QED']
    classification_props = ['QED','QED']

    state = TrainState()
    state.add_new_case(task,rewrite=True)
    state.ml_model_upd_data(case=task,
                            feature_column=feature_column,
                            target_column=target_column,
                            predictable_properties={"regression":regression_props, "classification":classification_props})
    print(state(task,'ml')['Predictable properties'])
    state.gen_model_upd_data(case=task,data_path=r'D:\Projects\ChemCoScientist\Project\ChemCoScientist\automl\data\data_4j1r.csv')
    state.gen_model_upd_status(case=task)
    state.gen_model_upd_status(case=task)
    state.gen_model_upd_status(case=task)
    state.ml_model_upd_status(case=task)
    state.ml_model_upd_status(case=task)

    print(state.show_calculateble_propreties())
    print()
