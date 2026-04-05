import torch
from fastapi import FastAPI, UploadFile, File
from openchemie import OpenChemIE
from molscribe import MolScribe
from huggingface_hub import hf_hub_download
import tempfile
import logging
import os
from utils import _clean_reaction_entities, _clean_molecule_data
import numpy as np
import cv2
import base64
from fastapi.responses import JSONResponse
from docking.docking_tools import run_docking

logging.basicConfig(level=logging.INFO)

molscribe_model = None
openchemie_model = None
app = FastAPI()


@app.get("/")
def index():
    return {"response": "Chemical tools"}


@app.on_event("startup")
def load_models():
    global molscribe_model
    global openchemie_model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    logging.info(f"Using {device} device")
    molscribe_ckpt_path = hf_hub_download(repo_id='yujieq/MolScribe', filename='swin_base_char_aux_1m.pth')
    molscribe_model = MolScribe(molscribe_ckpt_path, device=device)
    openchemie_model = OpenChemIE(device=device)


@app.post("/extract_reactions_from_pdf/")
async def extract_reactions_from_pdf(pdf_file: UploadFile = File(...)):
    """Extracts reactions from a PDF file."""
    try:
        if not pdf_file.filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Invalid file type",
                    "data": None,
                    "error": "File must be a PDF"
                }
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf_file.read())
            tmp_path = tmp.name
        
        figure_results = openchemie_model.extract_reactions_from_figures_in_pdf(tmp_path)
        for figure in figure_results:
            for reaction in figure.get('reactions', []):
                _clean_reaction_entities(reaction.get('reactants', []))
                _clean_reaction_entities(reaction.get('conditions', []))
                _clean_reaction_entities(reaction.get('products', []))
            figure.pop('figure', None)
        figure_results = [figure for figure in figure_results if len(figure.get('reactions', [])) > 0]
        
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Extracted {len(figure_results)} figure(s) with reactions",
                "data": figure_results,
                "error": None
            }
        )
    except Exception as e:
        logging.error(f"Error extracting reactions from PDF: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Failed to extract reactions from PDF",
                "data": None,
                "error": str(e)
            }
        )


@app.post("/extract_reactions_from_figure/")
async def extract_reactions_from_figure(image: UploadFile = File(...)):
    """Extracts reactions from an image file."""
    try:
        contents = await image.read()
        image_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Failed to decode image",
                    "data": None,
                    "error": "Invalid image format or corrupted file"
                }
            )
        
        figure_results = openchemie_model.extract_reactions_from_figures([image_array])
        for figure in figure_results:
            for reaction in figure.get('reactions', []):
                _clean_reaction_entities(reaction.get('reactants', []))
                _clean_reaction_entities(reaction.get('conditions', []))
                _clean_reaction_entities(reaction.get('products', []))
            figure.pop('figure', None)
        figure_results = [figure for figure in figure_results if len(figure.get('reactions', [])) > 0]
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Extracted {len(figure_results)} figure(s) with reactions",
                "data": figure_results,
                "error": None
            }
        )
    except Exception as e:
        logging.error(f"Error extracting reactions from figure: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Failed to extract reactions from figure",
                "data": None,
                "error": str(e)
            }
        )


@app.post("/extract_molecules_from_pdf/")
async def extract_molecules_from_pdf(pdf_file: UploadFile = File(...)):
    """Extracts molecules with identifiers from a PDF file."""
    try:
        if not pdf_file.filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Invalid file type",
                    "data": None,
                    "error": "File must be a PDF"
                }
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf_file.read())
            tmp_path = tmp.name
        
        figure_results = openchemie_model.extract_molecule_corefs_from_figures_in_pdf(tmp_path)
        for figure in figure_results:
            figure['bboxes'] = [
                _clean_molecule_data(bbox, False)
                for bbox in figure.get('bboxes', [])
            ]
        
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        total_molecules = sum(len(figure.get('bboxes', [])) for figure in figure_results)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Extracted {total_molecules} molecule(s) from {len(figure_results)} figure(s)",
                "data": figure_results,
                "error": None
            }
        )
    except Exception as e:
        logging.error(f"Error extracting molecules from PDF: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Failed to extract molecules from PDF",
                "data": None,
                "error": str(e)
            }
        )


@app.post("/extract_molecules_from_figure/")
async def extract_molecules_from_figure(image: UploadFile = File(...)):
    """Extracts molecules with identifiers from an image file."""
    try:
        contents = await image.read()
        image_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Failed to decode image",
                    "data": None,
                    "error": "Invalid image format or corrupted file"
                }
            )
        
        figure_results = openchemie_model.extract_molecule_corefs_from_figures([image_array])
        for figure in figure_results:
            figure['bboxes'] = [
                _clean_molecule_data(bbox, False)
                for bbox in figure.get('bboxes', [])
            ]
        
        total_molecules = sum(len(figure.get('bboxes', [])) for figure in figure_results)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Extracted {total_molecules} molecule(s) from {len(figure_results)} figure(s)",
                "data": figure_results,
                "error": None
            }
        )
    except Exception as e:
        logging.error(f"Error extracting molecules from figure: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Failed to extract molecules from figure",
                "data": None,
                "error": str(e)
            }
        )
    
    
@app.post("/convert_image_to_smiles/")
async def convert_image_to_smiles(image_file: UploadFile = File(...)):
    """Convert image to SMILES."""
    try:
        if not image_file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Invalid file type",
                    "data": None,
                    "error": "File must be an image"
                }
            )
        
        contents = await image_file.read()
        image_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        if image_array is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Failed to decode image",
                    "data": None,
                    "error": "Invalid image format or corrupted file"
                }
            )
        
        output = molscribe_model.predict_image(image_array)
        smiles = output.get("smiles", "")
        
        if not smiles:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "No SMILES string could be extracted from the image",
                    "data": None,
                    "error": "Could not recognize molecule in image"
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Successfully converted image to SMILES",
                "data": smiles,
                "error": None
            }
        )
    except Exception as e:
        logging.error(f"Error converting image to SMILES: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Failed to convert image to SMILES",
                "data": None,
                "error": str(e)
            }
        )


@app.post("/docking/")
async def docking(smiles: str, pdb_id: str):
    """Docking."""
    try:
        result = run_docking(smiles, pdb_id)
        
        data = {
            "affinity": result["affinity"],
            "visualization": result.get("visualization")
        }
    
        response_data = {
            "success": result["success"],
            "message": result["message"],
            "data": data,
            "error": result.get("error")
        }
        
        status_code = 200 if result["success"] else 400
        return JSONResponse(status_code=status_code, content=response_data)
    
    except Exception as e:
        logging.error(f"Error in docking endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Docking process failed",
                "data": None,
                "error": str(e)
            }
        )
    
    
    
    