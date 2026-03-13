import os

import pandas as pd
from fastapi import Body
from pydantic import BaseModel

from utils.automl_main import run_predict_automl_from_list, run_train_automl
from utils.base_state import TrainState

try:
    from s3_utils import S3BucketService, s3_service as default_s3_service
except ModuleNotFoundError:
    from .s3_utils import S3BucketService, s3_service as default_s3_service

class MLData(BaseModel):
        """
        Represents a container for machine learning data, handling loading, processing, and storage.
        
                 Class Attributes:
                 - data: The loaded dataset.
                 - case: Specifies whether the data represents a regression or classification task.
                 - data_path: Path to the dataset file.
                 - target_column: Name of the target variable column.
                 - smiles_list: List of SMILES strings if applicable to the dataset.
                 - timeout: Timeout value for data loading operations.
                 - feature_column: Name of the feature column.
                 - path_to_save: Path to save the processed data.
                 - description: A textual description of the dataset.
                 - regression_props: List of regression properties.
                 - classification_props: List of classification properties.
        """

        data:dict = None
        case:str = None
        data_path:str = None
        target_column:list = None
        smiles_list: list = None
        timeout:int = 30 #30 min
        feature_column:list = ['Smiles']
        path_to_save:str = 'train_model_data/trained_data'
        description:str = 'Unknown case.'
        regression_props:list= None
        classification_props:list = None
        save_trained_data_to_sync_server:bool = False
        endpoint_url:str = os.getenv("ENDPOINT_URL")
        access_key:str = os.getenv("ACCESS_KEY")
        secret_key:str = os.getenv("SECRET_KEY")
        bucket_name:str = os.getenv("BUCKET_NAME")
        s3_key:str = None
        # Backward compatibility with previous payload names.
        s3_bucket:str = None
        s3_endpoint_url:str = None


def _build_s3_service(data: MLData) -> S3BucketService:
    endpoint_url = data.endpoint_url or data.s3_endpoint_url or os.getenv("ENDPOINT_URL") or default_s3_service.endpoint
    access_key = data.access_key or os.getenv("ACCESS_KEY") or default_s3_service.access_key
    secret_key = data.secret_key or os.getenv("SECRET_KEY") or default_s3_service.secret_key
    bucket_name = data.bucket_name or data.s3_bucket or os.getenv("BUCKET_NAME") or default_s3_service.bucket_name

    if not endpoint_url:
        raise ValueError("S3 endpoint is empty. Set `endpoint_url` or env `ENDPOINT_URL`.")
    if not access_key:
        raise ValueError("S3 access key is empty. Set `access_key` or env `ACCESS_KEY`.")
    if not secret_key:
        raise ValueError("S3 secret key is empty. Set `secret_key` or env `SECRET_KEY`.")
    if not bucket_name:
        raise ValueError("S3 bucket is empty. Set `bucket_name` or env `BUCKET_NAME`.")

    return S3BucketService(
        endpoint=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
    )


def _download_dataset_from_s3(data: MLData) -> str:
    """Download training CSV from S3 and return normalized local path."""
    if not data.case:
        raise ValueError("`case` is required.")
    s3_service = _build_s3_service(data)

    if not data.data_path:
        data.data_path = f"data/{data.case}/data.csv"
    data.data_path = data.data_path.replace("\\", "/")
    local_dir = os.path.dirname(data.data_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    s3_service.download_image_from_s3(s3_key=f"/train/{data.case}.csv", local_path=data.data_path)
    return data.data_path


def train_ml_with_data(data:MLData=Body()):
    """
    Trains a machine learning model using provided data.
    
    This method prepares data for model training by saving it to a CSV file and 
    configuring the training environment with details like data location, 
    feature columns, and target columns. It then initiates the AutoML training process. 
    This allows the system to learn from specific datasets provided by the user.
    
    Args:
        data (MLData): The input data containing case details, data (as a pandas DataFrame), 
                       feature columns, target columns, and prediction properties.
    
    Returns:
        None
    """
    state = TrainState()
    state.add_new_case(case_name=data.case,
                        rewrite=True,
                        description=data.description)
    if data.case:
            _download_dataset_from_s3(data)
            df = pd.read_csv(data.data_path).dropna()
            if data.feature_column[0] not in df.columns:
                raise ValueError(
                    f"Feature column '{data.feature_column[0]}' not found in downloaded file. "
                    f"Available columns: {df.columns.tolist()}"
                )
            #df = df[df[data.feature_column[0]].astype(str).str.len() < 200]
            df.to_csv(data.data_path, index=False)
    elif data.data is not None:
            df = pd.DataFrame(data.data)
            data.data_path = f"data/{data.case}/data.csv"
            os.makedirs(os.path.dirname(data.data_path), exist_ok=True)
            df = df.dropna()
            #df = df[df[data.feature_column[0]].astype(str).str.len()<200]
            df.to_csv(data.data_path, index=False) 
                    
    state.ml_model_upd_data(case=data.case,
                            data_path=data.data_path,
                            feature_column=data.feature_column,
                            target_column=data.target_column,
                                predictable_properties={"regression":data.regression_props, "classification":data.classification_props})
    run_train_automl(case=data.case,
                        path_to_save=data.path_to_save,
                        timeout=data.timeout,
                        save_trained_data_to_sync_server=data.save_trained_data_to_sync_server)

def inference_ml(data:MLData=Body()):
    """
    Runs a prediction using an automated machine learning model.
    
    Args:
        data: An MLData object containing the case identifier and a list of SMILES strings representing chemical structures.
    
    Returns:
        The prediction results generated by the AutoML model, based on the provided chemical structures and case.
    """
    resutls = run_predict_automl_from_list(data.case,data=data.smiles_list)
    return resutls


