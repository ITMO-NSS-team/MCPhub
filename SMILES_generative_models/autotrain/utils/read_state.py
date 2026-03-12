"""Legacy compatibility wrapper for old imports.

Use `autotrain.utils.base_state` directly in new code.
"""

from autotrain.utils.base_state import BaseState, TrainState

__all__ = ["BaseState", "TrainState"]
