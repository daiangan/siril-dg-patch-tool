from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PatchAction:
    """Represents a single reversible patch operation."""
    x: int
    y: int
    w: int
    h: int
    before_data: np.ndarray
    after_data: np.ndarray
    mask: np.ndarray
    algorithm: str
    opacity: float


class HistoryManager:
    """
    Manages an Undo/Redo stack of localized patch operations.
    Stores only the ROI sub-arrays to minimize RAM footprint on large astro images.
    """

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth
        self.undo_stack: list[PatchAction] = []
        self.redo_stack: list[PatchAction] = []

    def record_patch(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        before_data: np.ndarray,
        after_data: np.ndarray,
        mask: np.ndarray,
        algorithm: str,
        opacity: float,
    ) -> PatchAction:
        """Records an applied patch to the undo stack and clears redo history."""
        action = PatchAction(
            x=x,
            y=y,
            w=w,
            h=h,
            before_data=before_data.copy(),
            after_data=after_data.copy(),
            mask=mask.copy(),
            algorithm=algorithm,
            opacity=opacity,
        )
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack.pop(0)

        self.redo_stack.clear()
        return action

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo(self, image_buffer) -> Optional[PatchAction]:
        """Reverts the last patch by restoring before_data in image_buffer."""
        if not self.can_undo():
            return None

        action = self.undo_stack.pop()
        image_buffer.set_roi(action.x, action.y, action.w, action.h, action.before_data)
        self.redo_stack.append(action)
        return action

    def redo(self, image_buffer) -> Optional[PatchAction]:
        """Re-applies the reverted patch by restoring after_data in image_buffer."""
        if not self.can_redo():
            return None

        action = self.redo_stack.pop()
        image_buffer.set_roi(action.x, action.y, action.w, action.h, action.after_data)
        self.undo_stack.append(action)
        return action

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
