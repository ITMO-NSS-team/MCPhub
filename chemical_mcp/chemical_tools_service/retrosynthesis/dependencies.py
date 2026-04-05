from typing import Dict, Any, List, Optional
from collections import Counter


class RetroSyntehsisDependencies:

    
    @classmethod
    async def process_retrosynthesis_result(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform the raw response from ASKCOS RetroStar to a compact format for the frontend.

        Expected input:
            either the entire response from /api/tree-search/retro-star/call-sync-without-token
            (with the "result" -> "uds" fields),
            or already a uds-chunk (with the "node_dict", "uuid2smiles", "pathways" fields).

        Output:
        {
        "target": "<SMILES of the target molecule>",
        "routes": [
            {
            "id": "route_1",
            "depth": 1,
            "precursor_cost": 1,
            "score": null,  # if it exists in pathways_properties
            "min_step_plausibility": 0.7,
            "avg_step_plausibility": 0.8,
            "steps": [
                {
                "reaction_smiles": "C=CCC>>CCCC",
                "mapped_smiles": "...",
                "plausibility": 0.75,
                "precursor_rank": 1,
                "precursor_score": -0.02,
                "model_score": 0.03,
                "template": {
                    "reaction_smarts": "...",
                    "template_rank": 5,
                    "num_examples": 4038
                },
                "reactants": [
                    {
                    "smiles": "C=CCC",
                    "terminal": true,
                    "buy_link": "https://..."
                    }
                ],
                "products": [
                    {
                    "smiles": "CCCC",
                    "terminal": False,
                    "buy_link": "https://..."
                    }
                ]
                },
                ...
            ]
            },
            ...
        ]
        }
        """
        # --- 1. Extract uds ---
        if "uds" in response:
            uds = response
        elif "result" in response and "uds" in response["result"]:
            uds = response["result"]["uds"]
        else:
            raise ValueError("The key 'uds' not found in the root or in response['result'].")

        node_dict: Dict[str, Dict[str, Any]] = uds.get("node_dict", {})
        uuid2smiles: Dict[str, str] = uds.get("uuid2smiles", {})
        pathways: List[List[Dict[str, str]]] = uds.get("pathways", [])
        pathways_props: List[Dict[str, Any]] = uds.get("pathways_properties", [])

        # uuid -> node (chemical/reaction)
        uuid2node: Dict[str, Dict[str, Any]] = {
            u: node_dict.get(smiles, {}) for u, smiles in uuid2smiles.items()
        }

        # --- 2. Determine the target (target molecule) ---
        target_smiles: Optional[str] = None
        if pathways:
            first_edge = pathways[0][0]
            target_uuid = first_edge["source"]
            target_smiles = uuid2smiles.get(target_uuid)

        # --- 3. Helper: information about the chemical substance (for reactants/products) ---
        def chem_info(smiles: str) -> Dict[str, Any]:
            nd = node_dict.get(smiles, {})
            props_list = nd.get("properties") or []
            buy_link = None
            for p in props_list:
                if isinstance(p, dict) and "link" in p:
                    buy_link = p["link"]
                    break
            return {
                "smiles": smiles,
                "terminal": nd.get("terminal"),
                "buy_link": buy_link,
            }

        routes_ui: List[Dict[str, Any]] = []

        # --- 4. Expand each path into a list of steps ---
        for idx, path_edges in enumerate(pathways):
            pw_prop = pathways_props[idx] if idx < len(pathways_props) else {}
            depth = pw_prop.get("depth")
            precursor_cost = pw_prop.get("precursor_cost")
            route_score = pw_prop.get("score")

            reaction_uuids: List[str] = []
            for edge in path_edges:
                tgt = edge["target"]
                node = uuid2node.get(tgt)
                if node and node.get("type") == "reaction" and tgt not in reaction_uuids:
                    reaction_uuids.append(tgt)

            steps: List[Dict[str, Any]] = []
            step_plaus = []

            for r_uuid in reaction_uuids:
                r_node = uuid2node.get(r_uuid, {})
                r_props = r_node.get("reaction_properties") or {}

                canonical_rxn = r_props.get("canonical_reaction_smiles") or r_node.get("smiles")
                mapped_smiles = r_props.get("mapped_smiles")

                rxn_plaus = r_props.get("plausibility") or r_node.get("plausibility")
                if rxn_plaus is not None:
                    step_plaus.append(float(rxn_plaus))

                precursor_rank = r_node.get("precursor_rank")
                precursor_score = r_node.get("precursor_score")
                model_score = r_node.get("rxn_score_from_model")

                template_info = None
                model_meta = r_node.get("model_metadata") or []
                if model_meta:
                    src = (model_meta[0] or {}).get("source", {})
                    tmpl = src.get("template")
                    if tmpl:
                        template_info = {
                            "reaction_smarts": tmpl.get("reaction_smarts"),
                            "template_rank": tmpl.get("template_rank"),
                            "num_examples": tmpl.get("num_examples"),
                        }

                reactants_ui: List[Dict[str, Any]] = []
                products_ui: List[Dict[str, Any]] = []

                if canonical_rxn and ">>" in canonical_rxn:
                    reac_str, prod_str = canonical_rxn.split(">>")
                    reac_smiles = [s for s in reac_str.split(".") if s]
                    prod_smiles = [s for s in prod_str.split(".") if s]

                    reactants_ui = [chem_info(sm) for sm in reac_smiles]
                    products_ui = [chem_info(sm) for sm in prod_smiles]
                    
                    reactants_ui = cls.collapse_reagents(reactants_ui)
                    products_ui = cls.collapse_reagents(products_ui)
                    

                step_data = {
                    "reaction_smiles": canonical_rxn,
                    "mapped_smiles": mapped_smiles,
                    "plausibility": rxn_plaus,
                    "precursor_rank": precursor_rank,
                    "precursor_score": precursor_score,
                    "model_score": model_score,
                    "template": template_info,
                    "reactants": reactants_ui,
                    "products": products_ui,
                }
                steps.append(step_data)

            if step_plaus:
                min_pl = min(step_plaus)
                avg_pl = sum(step_plaus) / len(step_plaus)
            else:
                min_pl = None
                avg_pl = None

            route_ui = {
                "id": f"route_{idx + 1}",
                "depth": depth,
                "precursor_cost": precursor_cost,
                "score": route_score,
                "min_step_plausibility": min_pl,
                "avg_step_plausibility": avg_pl,
                "steps": steps,
            }
            routes_ui.append(route_ui)

        routes_ui.sort(
            key=lambda r: (
                -(r["avg_step_plausibility"] or 0.0),
                r["depth"] if r["depth"] is not None else 999,
            )
        )

        return {
            "target": target_smiles,
            "routes": routes_ui,
        }


    @classmethod
    def collapse_reagents(cls, reactants_ui):
        """
        Accepts a list of the following form:
        [
        {"smiles": "CC[Mg]Br", "terminal": true, "buy_link": "..."},
        {"smiles": "CC[Mg]Br", "terminal": true, "buy_link": "..."},
        ...
        ]
        and returns a list of the following form:
        [
        {"smiles": "CC[Mg]Br", "terminal": True, "buy_link": "...", "stoichiometry": 2}
        ]
        """
        counter = Counter()
        extra_data = {}

        for r in reactants_ui:
            key = (r["smiles"], r.get("terminal"), r.get("buy_link"))
            counter[key] += 1
            # remember any additional fields (if they appear)
            extra_data.setdefault(key, r)

        result = []
        for key, n in counter.items():
            smiles, terminal, buy_link = key
            base = dict(extra_data[key])  # copy the original dict
            base["stoichiometry"] = n
            result.append(base)

        return result
