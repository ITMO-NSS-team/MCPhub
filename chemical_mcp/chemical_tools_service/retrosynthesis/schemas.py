from typing import List, Optional, Literal
from pydantic import BaseModel, AnyUrl, Field

class TreeSearchRequest(BaseModel):
    smiles: str = Field(default="CCCC", description="SMILES of the target molecule")


class MoleculeDTO(BaseModel):
    smiles: str
    terminal: Optional[bool] = None
    buy_link: Optional[AnyUrl] = None  # if the links are dirty, you can replace it with Optional[str]
    stoichiometry: int = 1            # default 1, if collapse_reagents did not add the field


class ReactionTemplateDTO(BaseModel):
    reaction_smarts: str
    template_rank: Optional[int] = None
    num_examples: Optional[int] = None


class RetroStepDTO(BaseModel):
    reaction_smiles: str
    mapped_smiles: Optional[str] = None

    plausibility: Optional[float] = None
    precursor_rank: Optional[int] = None
    precursor_score: Optional[float] = None
    model_score: Optional[float] = None

    template: Optional[ReactionTemplateDTO] = None

    reactants: List[MoleculeDTO]
    products: List[MoleculeDTO]


class RetroRouteDTO(BaseModel):
    id: str

    depth: Optional[int] = None
    precursor_cost: Optional[float] = None
    score: Optional[float] = None

    min_step_plausibility: Optional[float] = None
    avg_step_plausibility: Optional[float] = None

    steps: List[RetroStepDTO]


class RetroResponseDTO(BaseModel):
    target: Optional[str] = None
    routes: List[RetroRouteDTO]
