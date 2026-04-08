# forward/schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel

ForwardBackend = Literal["wldn5", "graph2smiles", "augmented_transformer"]


class ForwardRequestDTO(BaseModel):
    backend: ForwardBackend
    model_name: Optional[str] = "pistachio"
    # as in ASKCOS: list of SMILES strings (usually a batch of examples)
    smiles: List[str]
    # string fields as in the controller
    reagents: str = ""
    solvent: str = ""


class ForwardProductDTO(BaseModel):
    smiles: str
    score: float                             # probability/score of the model


class ForwardResponseDTO(BaseModel):
    inputs: List[str]                        # everything that was fed to the input (reactants+reagents+solvent)
    backend: ForwardBackend
    model_name: str
    predictions: List[ForwardProductDTO]
