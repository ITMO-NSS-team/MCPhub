FAST_TREE_SEARCH_BODY = {
    "description": "",
    "tags": "",
    "expand_one_options": {
        "template_count": 30,
        "max_cum_template_prob": 0.98,
        "forbidden_molecules": [],
        "known_bad_reactions": [],
        "retro_backend_options": [
            {
                "retro_backend": "template_relevance",
                "retro_model_name": "reaxys",
                "max_num_templates": 300,
                "max_cum_prob": 0.98,
                "attribute_filter": [],
                "threshold": 0.35,
                "top_k": 5
            }
        ],
        "use_fast_filter": True,
        "filter_threshold": 0.8,
        "retro_rerank_backend": "relevance_heuristic",
        "atom_map_backend": "rxnmapper",
        "cluster_precursors": False,
        "cluster_setting": {
            "feature": "original",
            "cluster_method": "hdbscan",
            "fp_type": "morgan",
            "fp_length": 512,
            "fp_radius": 1,
            "classification_threshold": 0.2
        },
        "extract_template": False,
        "return_reacting_atoms": True,
        "selectivity_check": False
    },
    "build_tree_options": {
        "expansion_time": 5,
        "max_branching": 10,
        "max_depth": 4,
        "exploration_weight": 1,
        "return_first": False,
        "max_trees": 200,
        "buyable_logic": "and",
        "max_ppg_logic": "none",
        "max_ppg": 0,
        "max_scscore_logic": "none",
        "max_scscore": 0,
        "chemical_property_logic": "none",
        "max_chemprop_c": 0,
        "max_chemprop_n": 0,
        "max_chemprop_o": 0,
        "max_chemprop_h": 0,
        "chemical_popularity_logic": "none",
        "min_chempop_reactants": 5,
        "min_chempop_products": 5,
        "custom_buyables": [],
        "use_value_network": True
    },
    "enumerate_paths_options": {
        "path_format": "json",
        "json_format": "nodelink",
        "sorting_metric": "plausibility",
        "validate_paths": True,
        "score_trees": False,
        "cluster_trees": False,
        "cluster_method": "hdbscan",
        "min_samples": 5,
        "min_cluster_size": 5,
        "paths_only": False,
        "max_paths": 50
    },
    "run_async": False,
}

BALANCED_TREE_SEARCH_BODY = {   
    "description": "",
    "tags": "",
    "expand_one_options": {
        "template_count": 100,
        "max_cum_template_prob": 0.995,
        "forbidden_molecules": [],
        "known_bad_reactions": [],
        "retro_backend_options": [
            {
                "retro_backend": "template_relevance",
                "retro_model_name": "reaxys",
                "max_num_templates": 1000,
                "max_cum_prob": 0.995,
                "attribute_filter": [],
                "threshold": 0.3,
                "top_k": 15
            }
        ],
        "use_fast_filter": True,
        "filter_threshold": 0.75,
        "retro_rerank_backend": "relevance_heuristic",
        "atom_map_backend": "rxnmapper",
        "cluster_precursors": False,
        "cluster_setting": {
            "feature": "original",
            "cluster_method": "hdbscan",
            "fp_type": "morgan",
            "fp_length": 512,
            "fp_radius": 1,
            "classification_threshold": 0.2
        },
        "extract_template": False,
        "return_reacting_atoms": True,
        "selectivity_check": False
    },
    "build_tree_options": {
        "expansion_time": 15,
        "max_branching": 25,
        "max_depth": 5,
        "exploration_weight": 1,
        "return_first": False,
        "max_trees": 600,
        "buyable_logic": "and",
        "max_ppg_logic": "none",
        "max_ppg": 0,
        "max_scscore_logic": "none",
        "max_scscore": 0,
        "chemical_property_logic": "none",
        "max_chemprop_c": 0,
        "max_chemprop_n": 0,
        "max_chemprop_o": 0,
        "max_chemprop_h": 0,
        "chemical_popularity_logic": "none",
        "min_chempop_reactants": 5,
        "min_chempop_products": 5,
        "custom_buyables": [],
        "use_value_network": True
    },
    "enumerate_paths_options": {
        "path_format": "json",
        "json_format": "nodelink",
        "sorting_metric": "plausibility",
        "validate_paths": True,
        "score_trees": False,
        "cluster_trees": False,
        "cluster_method": "hdbscan",
        "min_samples": 5,
        "min_cluster_size": 5,
        "paths_only": False,
        "max_paths": 200
    },
    "run_async": False,
}


DEEP_TREE_SEARCH_BODY = {
    "description": "",
    "tags": "",
    "expand_one_options": {
        "template_count": 200,
        "max_cum_template_prob": 0.999,
        "forbidden_molecules": [],
        "known_bad_reactions": [],
        "retro_backend_options": [
            {
                "retro_backend": "template_relevance",
                "retro_model_name": "reaxys",
                "max_num_templates": 2000,
                "max_cum_prob": 0.999,
                "attribute_filter": [],
                "threshold": 0.2,
                "top_k": 30
            }
        ],
        "use_fast_filter": True,
        "filter_threshold": 0.65,
        "retro_rerank_backend": "relevance_heuristic",
        "atom_map_backend": "rxnmapper",
        "cluster_precursors": False,
        "cluster_setting": {
            "feature": "original",
            "cluster_method": "hdbscan",
            "fp_type": "morgan",
            "fp_length": 512,
            "fp_radius": 1,
            "classification_threshold": 0.2
        },
        "extract_template": False,
        "return_reacting_atoms": True,
        "selectivity_check": False
    },
    "build_tree_options": {
        "expansion_time": 45,
        "max_branching": 40,
        "max_depth": 7,
        "exploration_weight": 1,
        "return_first": False,
        "max_trees": 1500,
        "buyable_logic": "and",
        "max_ppg_logic": "none",
        "max_ppg": 0,
        "max_scscore_logic": "none",
        "max_scscore": 0,
        "chemical_property_logic": "none",
        "max_chemprop_c": 0,
        "max_chemprop_n": 0,
        "max_chemprop_o": 0,
        "max_chemprop_h": 0,
        "chemical_popularity_logic": "none",
        "min_chempop_reactants": 5,
        "min_chempop_products": 5,
        "custom_buyables": [],
        "use_value_network": True
    },
    "enumerate_paths_options": {
        "path_format": "json",
        "json_format": "nodelink",
        "sorting_metric": "plausibility",
        "validate_paths": True,
        "score_trees": False,
        "cluster_trees": False,
        "cluster_method": "hdbscan",
        "min_samples": 5,
        "min_cluster_size": 5,
        "paths_only": False,
        "max_paths": 500
    },
    "run_async": False,
}
