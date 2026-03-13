from logging import debug
from fastapi import FastAPI,Body
import os
import sys
print(sys.path)
sys.path.append(os.getcwd())
import uvicorn
import json
import_path = os.path.dirname(os.path.abspath(__file__))
import socket
import yaml
from api_utils import *
from utils.base_state import TrainState

try:
    from utils.state_s3 import download_state_file
except ModuleNotFoundError:
    from .utils.state_s3 import download_state_file

with open(os.path.join(import_path, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

is_public_API = config['is_public_API']


if __name__=='__main__':
    
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('10.254.254.254', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP
    
    print("Getting IP")
    ip = str(get_ip())
    print(f"Current IP: {ip}")
    print("Starting...")

    def sync_state_from_s3():
        state_path = os.path.join(import_path, "state.json")
        download_state_file(local_path=state_path)
        return state_path

    app = FastAPI(debug=True)
    @app.get("/")
    def health_check():
        return {"health_check": "ok"}

    # API operations
    @app.get("/check_state")
    def check_state():
        sync_state_from_s3()
        state = TrainState(state_path=os.path.join(import_path, "state.json"))
        calc_properies = state.show_calculateble_propreties()
        current_state = state().copy()
        
        del current_state["Calculateble properties"]
        return {'state': current_state,
                'calc_propreties':list(calc_properies)}

    @app.post("/train_ml")
    def train_ml_api(data:MLData=Body()):
        sync_state_from_s3()
        train_ml_with_data(data)

    @app.post("/predict_ml")
    def predict_ml_api(data:MLData=Body()):
        print(data)
        sync_state_from_s3()
        return inference_ml(data)
    

    if is_public_API:
        uvicorn_ip = ip
    else:
        uvicorn_ip = '127.0.0.1' 
    uvicorn.run(app,host=uvicorn_ip,port=81,log_level='info')
