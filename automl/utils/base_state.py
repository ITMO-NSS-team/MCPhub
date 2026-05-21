import copy
import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel

try:
    from utils.calculateble_prop_funcs import config
    from utils.state_s3 import download_state_file, upload_state_file
except ModuleNotFoundError:
    from .calculateble_prop_funcs import config
    from .state_s3 import download_state_file, upload_state_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STATE_PATH = os.path.join(BASE_DIR, "state.json")


VALID_STATUSES = ("Not Trained", "Training", "Trained", "Failed")


class BaseState(BaseModel):
    status: str = None
    weights_path: str = {"regression": None, "classification": None}
    arch_type: str = None
    case_name: str = None
    base_state: dict = {
        "description": None,
        "generative_models": {
            "data_path": None,
            "feature_column": None,
            "target_column": None,
            "problem": None,
            "status": status,
            "weights_path": None,
            "arch_type": arch_type,
            "metric": None,
            "error": None,
            "error_at": None,
        },
        "ml_models": {
            "data_path": None,
            "feature_column": None,
            "target_column": None,
            "status": status,
            "weights_path": weights_path,
            "metric": None,
            "Predictable properties": {},
            "error": None,
            "error_at": None,
        },
    }


class TrainState:
    """State manager for trained ML/generative cases with optional S3 sync."""

    defult_parameters = BaseState()

    def __init__(self, state_path: str = None, sync_with_s3: bool = True):
        # The S3 key for state.json is a hardcoded contract shared with the
        # generative MCP server (see utils/state_s3.STATE_S3_KEY). It is NOT
        # configurable per-instance / per-env: any drift between the two
        # servers' keys would split the state into two independent files,
        # silently dropping cases registered on the "other" server.
        self.state_path = state_path or DEFAULT_STATE_PATH
        self.sync_with_s3 = sync_with_s3

        state_dir = os.path.dirname(self.state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

        if self.sync_with_s3:
            self.sync_state_from_s3(raise_on_error=False)

        if os.path.isfile(self.state_path):
            self.current_state = self.__load_state()
        else:
            self.current_state = {"Calculateble properties": config}
            self.__save_state()

    def __call__(self, case: str = None, model: str = None, *args, **kwargs):
        if case is None:
            return self.current_state
        if case not in self.current_state:
            print("Case do not exist in current State!")
            return None
        if model == "ml":
            return self.current_state[case]["ml_models"]
        if model == "gen":
            return self.current_state[case]["generative_models"]
        return self.current_state[case]

    def add_new_case(
        self,
        case_name: str,
        rewrite: bool = False,
        description: str = "Unknown case",
        **kwargs,
    ):
        if case_name in self.current_state and not rewrite:
            print(f"Case already exist! Change name for new case, or use exist case state named {case_name}!")
            print(f"Now using case named '{case_name}' - Case Description: {self.current_state[case_name]['description']}")
            return None

        self.current_state[case_name] = copy.deepcopy(self.defult_parameters.base_state)
        self.current_state[case_name]["description"] = description
        self.__save_state()

    def gen_model_upd_data(
        self,
        case: str,
        data_path: str = None,
        feature_column: List[str] = None,
        target_column: List[str] = None,
    ):
        if data_path is not None:
            self.current_state[case]["generative_models"]["data_path"] = data_path
        if feature_column is not None:
            self.current_state[case]["generative_models"]["feature_column"] = feature_column
        if target_column is not None:
            self.current_state[case]["generative_models"]["target_column"] = target_column
        self.__save_state()

    def ml_model_upd_data(
        self,
        case: str,
        data_path: str = None,
        feature_column: List[str] = None,
        target_column: List[str] = None,
        predictable_properties: dict = None,
    ):
        if data_path is not None:
            self.current_state[case]["ml_models"]["data_path"] = data_path
        if feature_column is not None:
            self.current_state[case]["ml_models"]["feature_column"] = feature_column
        if target_column is not None:
            self.current_state[case]["ml_models"]["target_column"] = target_column
        if predictable_properties is not None:
            self.current_state[case]["ml_models"]["Predictable properties"] = predictable_properties
        self.__validate_properties(case)
        print(
            "Data for ML models training has been updated! "
            f"\n Current predictable properties and tasks are {self.current_state[case]['ml_models']['Predictable properties']}"
        )
        self.__save_state()

    def ml_model_upd_status(
        self,
        case: str,
        model_weight_path: str = None,
        metric=None,
        status: int = None,
        problem: str = "regression",
        error: Optional[str] = None,
    ):
        """Update ML training status, weights path, metrics, or error.

        Args:
            status: 0=Not Trained, 1=Training, 2=Trained, 3=Failed.
            error: Optional human-readable error message. When provided,
                `error` + `error_at` are written; `status` is also forced to
                'Failed' if no explicit `status` was supplied. On a successful
                status transition (Trained), `error`/`error_at` are cleared.
        """
        entry = self.current_state[case]["ml_models"]
        entry.setdefault("error", None)
        entry.setdefault("error_at", None)

        if error is not None:
            entry["error"] = str(error)
            entry["error_at"] = datetime.now(timezone.utc).isoformat()
            if status is None:
                entry["status"] = "Failed"

        if status is not None and entry["status"] != "Trained":
            entry["status"] = VALID_STATUSES[status]
            if VALID_STATUSES[status] == "Trained":
                entry["error"] = None
                entry["error_at"] = None
        elif entry["status"] == "Trained":
            print(f'ML model for task "{case}" already trained!')

        if model_weight_path is not None and not os.path.isdir(model_weight_path):
            os.mkdir(model_weight_path)
        if entry["weights_path"][problem] is None and model_weight_path is not None:
            entry["weights_path"][problem] = model_weight_path
        if metric is not None:
            entry["metric"] = metric
        self.__save_state()

    def gen_model_upd_status(
        self,
        case: str,
        model_weight_path: str = None,
        metric=None,
        error: Optional[str] = None,
    ):
        """Update generative training status, weights path, metrics, or error.

        Args:
            error: Optional human-readable error message. When provided,
                writes `error` + `error_at` and forces `status="Failed"`. On
                a successful Training -> Trained transition, error fields are
                cleared.
        """
        entry = self.current_state[case]["generative_models"]
        entry.setdefault("error", None)
        entry.setdefault("error_at", None)

        if error is not None:
            entry["error"] = str(error)
            entry["error_at"] = datetime.now(timezone.utc).isoformat()
            entry["status"] = "Failed"
        elif entry["status"] is None:
            entry["status"] = "Training"
        elif entry["status"] == "Training":
            entry["status"] = "Trained"
            entry["error"] = None
            entry["error_at"] = None
        else:
            print(f'Generative model for task "{case}" already trained!')

        if model_weight_path is not None and not os.path.isdir(model_weight_path):
            os.mkdir(model_weight_path)
        if entry["weights_path"] is not None and model_weight_path is not None:
            entry["weights_path"] = model_weight_path
        if metric is not None:
            entry["metric"] = metric
        self.__save_state()

    def show_calculateble_propreties(self):
        return self.current_state["Calculateble properties"].keys()

    @staticmethod
    def load_state(path: str = DEFAULT_STATE_PATH):
        with open(path) as state_file:
            state = json.load(state_file)
        state["Calculateble properties"] = config
        return state

    def save(self, path: str = DEFAULT_STATE_PATH):
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
        with open(self.state_path, "w") as state_file:
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
