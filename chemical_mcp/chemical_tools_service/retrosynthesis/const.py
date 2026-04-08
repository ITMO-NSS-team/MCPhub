from typing import ClassVar


class AskcosEndpoints:
    """Collection of ASKCOS API endpoint paths used across the service."""

    FORWARD: ClassVar[str] = "/api/forward/controller/call-sync"
    TREE_SEARCH: ClassVar[str] = "/api/tree-search/retro-star/call-sync-without-token"
    REACTION_CLASSIFICATION: ClassVar[str] = "/api/reaction-classification/call-sync"

