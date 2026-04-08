from typing import Dict, Any
from pydantic import ValidationError

from .schemas import (
    ReactionClassificationRequestDTO,
    ReactionClassificationResponseDTO,
)


class ReactionClassificationDependencies:
    """
    Utilities for /api/reaction-classification/*
    """

    @classmethod
    def make_body(cls, req: ReactionClassificationRequestDTO) -> Dict[str, Any]:
        """
        Body request to ASKCOS.
        Simply pass the request DTO as is.
        """
        return req.dict()

    @classmethod
    def parse_response(cls, askcos_json: Dict[str, Any]) -> ReactionClassificationResponseDTO:
        """
        Transform JSON from ASKCOS to a validated DTO.
        Format is 1:1, so push it into Pydantic.
        """
        try:
            return ReactionClassificationResponseDTO(**askcos_json)
        except ValidationError as e:
            raise ValueError(f"Failed to validate ReactionClassificationResponseDTO: {e}") from e
