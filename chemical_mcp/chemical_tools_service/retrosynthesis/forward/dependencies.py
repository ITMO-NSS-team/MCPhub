from typing import Dict, Any
from pydantic import ValidationError
from forward.schemas import (
    ForwardRequestDTO,
    ForwardResponseDTO,
    ForwardProductDTO,
    ForwardBackend,
)


class ForwardDependencies:
    """
    Utilities for working with /api/forward/controller/call-sync(-without-token).
    """

    DEFAULT_MODEL_BY_BACKEND: Dict[ForwardBackend, str] = {
        "wldn5": "pistachio",
        "graph2smiles": "uspto-stereo",
        "augmented_transformer": "uspto-stereo",
    }

    @classmethod
    def make_controller_body(cls, req: ForwardRequestDTO) -> Dict[str, Any]:
        """
        Body request to ASKCOS.
        """
        model_name = req.model_name or cls.DEFAULT_MODEL_BY_BACKEND[req.backend]

        body: Dict[str, Any] = {
            "backend": req.backend,
            "model_name": model_name,
            "smiles": req.smiles,
            "reagents": req.reagents or "",
            "solvent": req.solvent or "",
        }

        return body

    @classmethod
    def process_forward_result(
        cls,
        askcos_response: Dict[str, Any],
        req: ForwardRequestDTO,
    ) -> ForwardResponseDTO:
        """
        Parser strictly for the response:

        {
          "status_code": 0,
          "message": "string",
          "result": [
            [
              {
                "rank": 0,
                "outcome": "string",
                "score": 0,
                "prob": 0,
                "mol_wt": 0
              }
            ]
          ]
        }

        Map:
        - outcome -> smiles
        - prob    -> score
        """

        status_code = askcos_response.get("status_code")
        if status_code is None:
            raise ValueError("Missing 'status_code' in ASKCOS forward response")

        result = askcos_response.get("result") or []
        if not isinstance(result, list) or not result:
            raise ValueError("ASKCOS forward response has empty or invalid 'result' field")

        first_batch = result[0]
        if not isinstance(first_batch, list):
            raise ValueError("ASKCOS forward response 'result[0]' is not a list")

        predictions = []
        for item in first_batch:
            if not isinstance(item, dict):
                continue
            outcome = item.get("outcome")
            prob = item.get("prob")
            if not outcome:
                continue
            try:
                score_f = float(prob)
            except (TypeError, ValueError):
                score_f = 0.0
            predictions.append(
                ForwardProductDTO(
                    smiles=outcome,
                    score=score_f,
                )
            )

        inputs = list(req.smiles)
        if req.reagents:
            inputs.append(req.reagents)
        if req.solvent:
            inputs.append(req.solvent)

        model_name = req.model_name or cls.DEFAULT_MODEL_BY_BACKEND[req.backend]

        dto = ForwardResponseDTO(
            inputs=inputs,
            backend=req.backend,
            model_name=model_name,
            predictions=predictions,
        )
        return dto
