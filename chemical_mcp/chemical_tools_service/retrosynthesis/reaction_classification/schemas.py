from typing import List
from pydantic import BaseModel, Field


class ReactionClassificationRequestDTO(BaseModel):
    """
    Request from the user to classify the reaction.
    """
    smiles: List[str] = Field(..., description="List of reaction SMILES, e.g. ['A.B>>C']")
    num_results: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of classes/predictions per reaction"
    )


class ReactionClassificationHitDTO(BaseModel):
    """
    One classification result from ASKCOS.
    """
    rank: int
    reaction_num: str
    reaction_name: str
    reaction_classnum: str
    reaction_classname: str
    reaction_superclassnum: str
    reaction_superclassname: str
    prediction_certainty: float


class ReactionClassificationResponseDTO(BaseModel):
    """
    Full response from our service — 1:1 with the ASKCOS format.
    """
    status_code: int
    message: str
    result: List[ReactionClassificationHitDTO]
