#!/usr/bin/env python3
# Script: DG_Patch_Tool
# Title: DG_Patch_Tool
# Description: External patch and heal tool for starless image cleanup in Siril
# Author: Daian Gan
"""DG_Patch_Tool — standalone script for Siril.
Version: 1.0.0

Author: Daian Gan
Website: https://daiangan.com
"""

import sys
import importlib.abc
import importlib.util

# Ensure required packages in Siril's Python environment
try:
    import sirilpy
    if hasattr(sirilpy, "ensure_installed"):
        sirilpy.ensure_installed("PyQt6", "opencv-contrib-python", "numpy", "scipy")
except Exception:
    pass

_MODULE_SOURCES = {}
_PACKAGE_NAMES = ['dg_patch_tool', 'dg_patch_tool.core', 'dg_patch_tool.ui']


class _EmbeddedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _MODULE_SOURCES:
            return None
        return importlib.util.spec_from_loader(
            fullname, self, is_package=fullname in _PACKAGE_NAMES
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = _MODULE_SOURCES[module.__name__]
        exec(compile(source, "<" + module.__name__ + ">", "exec"), module.__dict__)


sys.meta_path.insert(0, _EmbeddedFinder())

# ============================================================================
# Module: dg_patch_tool
# ============================================================================
_MODULE_SOURCES["dg_patch_tool"] = r'''
"""
DG_Patch_Tool
Patch and heal cleanup tool for Siril astrophotography starless images.
Author: Daian Gan
"""

__version__ = "1.0.0"
__author__ = "Daian Gan"
'''

# ============================================================================
# Module: dg_patch_tool.app
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.app"] = r'''
import sys
import numpy as np
from PyQt6.QtWidgets import QApplication

from dg_patch_tool.siril_bridge import SirilBridge
from dg_patch_tool.ui.main_window import MainWindow


_STARTUP_BANNER_LINES = [
    "================================================",
    "DG Patch Tool",
    "Author: Daian Gan",
    "Email:  daian@ganmedia.com",
    "Web:    https://daiangan.com",
    "================================================",
]


def run(image_array: np.ndarray = None):
    """
    Main application entry point.
    If image_array is provided, uses it directly.
    Otherwise attempts to fetch active image from Siril via SirilBridge.
    If no active Siril session is detected, loads a realistic synthetic starless image for testing.
    """
    bridge = SirilBridge()

    # Output startup banner to Siril console log
    for line in _STARTUP_BANNER_LINES:
        bridge.log(line)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setStyle("Fusion")

    if image_array is None:
        if bridge.is_connected():
            image_array = bridge.get_image_pixeldata()

    if image_array is None:
        # Fallback to synthetic starless test image
        image_array = bridge.generate_synthetic_starless()

    window = MainWindow(raw_image=image_array, siril_bridge=bridge)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
'''

# ============================================================================
# Module: dg_patch_tool.core
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.core"] = r'''
from .image_buffer import ImageBuffer
from .history import HistoryManager, PatchAction
from .blend_engine import BlendEngine
from .safety_check import check_linear_data

__all__ = ["ImageBuffer", "HistoryManager", "PatchAction", "BlendEngine", "check_linear_data"]
'''

# ============================================================================
# Module: dg_patch_tool.core.blend_engine
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.core.blend_engine"] = r'''
import cv2
import numpy as np


class BlendEngine:
    """
    High-performance blending engine tailored for astrophotography cleanup.
    Supports:
    1. Poisson Mixed Clone (cv2.MIXED_CLONE) - default
    2. Poisson Normal Clone (cv2.NORMAL_CLONE)
    3. ShiftMap Exemplar Inpaint (cv2.xphoto.inpaint) - Auto-heal
    4. Laplacian Pyramid Multi-band Blending
    5. Real-time alpha compositing with feathering
    """

    MODE_POISSON_NORMAL = "Poisson (Normal Clone)"
    MODE_POISSON_MIXED = "Poisson (Mixed Clone)"
    MODE_LAPLACIAN = "Laplacian Pyramid"
    MODE_SHIFTMAP = "Auto-Heal (ShiftMap)"

    ALL_MODES = [
        MODE_POISSON_NORMAL,
        MODE_POISSON_MIXED,
        MODE_LAPLACIAN,
        MODE_SHIFTMAP,
    ]

    @staticmethod
    def feather_mask(mask: np.ndarray, feather_radius: int) -> np.ndarray:
        """
        Applies a smooth Gaussian feather to a binary mask.
        Returns float32 array in [0.0, 1.0].
        """
        if mask is None:
            return None

        m_float = (mask > 0).astype(np.float32)
        if feather_radius <= 0:
            return m_float

        ksize = int(feather_radius * 2 + 1)
        # Ensure ksize is odd
        if ksize % 2 == 0:
            ksize += 1
        sigma = max(0.5, feather_radius / 2.0)
        feathered = cv2.GaussianBlur(m_float, (ksize, ksize), sigma)
        return np.clip(feathered, 0.0, 1.0)

    @classmethod
    def blend(
        cls,
        dst_roi: np.ndarray,
        src_roi: np.ndarray,
        mask: np.ndarray,
        mode: str = MODE_POISSON_NORMAL,
        feather_radius: int = 3,
    ) -> np.ndarray:
        """
        Blends src_roi into dst_roi according to mask and mode.
        Returns blended array with the exact same shape and dtype as dst_roi.
        """
        if mode == cls.MODE_SHIFTMAP:
            return cls.inpaint_shiftmap(dst_roi, mask)
        elif mode == cls.MODE_LAPLACIAN:
            return cls.blend_laplacian_pyramid(dst_roi, src_roi, mask, feather_radius)
        elif mode == cls.MODE_POISSON_NORMAL:
            return cls.blend_poisson(dst_roi, src_roi, mask, cv2.NORMAL_CLONE)
        else:
            # Default to MIXED_CLONE
            return cls.blend_poisson(dst_roi, src_roi, mask, cv2.MIXED_CLONE)

    @classmethod
    def apply_opacity(
        cls,
        dst_roi: np.ndarray,
        patched_roi: np.ndarray,
        feathered_mask: np.ndarray,
        opacity: float,
    ) -> np.ndarray:
        """
        Real-time alpha compositing:
        result = dst + (patched - dst) * feathered_mask * opacity
        Runs in < 1ms on small ROI arrays.
        """
        if opacity <= 0.0 or feathered_mask is None or np.max(feathered_mask) == 0:
            return dst_roi.copy()

        alpha = float(np.clip(opacity, 0.0, 1.0))
        m = feathered_mask * alpha

        if dst_roi.ndim == 3 and m.ndim == 2:
            m = m[:, :, np.newaxis]

        # Convert to float for accurate interpolation without clipping/overflow
        d_f = dst_roi.astype(np.float32)
        p_f = patched_roi.astype(np.float32)

        blended = d_f + (p_f - d_f) * m

        # Clip and return to original dtype
        if np.issubdtype(dst_roi.dtype, np.floating):
            return np.clip(blended, 0.0, 1.0).astype(dst_roi.dtype)
        elif dst_roi.dtype == np.uint16:
            return np.clip(blended, 0.0, 65535.0).astype(np.uint16)
        elif dst_roi.dtype == np.uint8:
            return np.clip(blended, 0.0, 255.0).astype(np.uint8)
        return blended.astype(dst_roi.dtype)

    @classmethod
    def blend_poisson(
        cls,
        dst_roi: np.ndarray,
        src_roi: np.ndarray,
        mask: np.ndarray,
        clone_mode: int = cv2.MIXED_CLONE,
    ) -> np.ndarray:
        """
        Poisson blending via cv2.seamlessClone.
        Handles mono/color, high-bit-depth data, and border safety.
        """
        h, w = dst_roi.shape[:2]
        if h < 5 or w < 5:
            return dst_roi.copy()

        # Prepare 8-bit versions for cv2.seamlessClone
        dst_8, norm_scale, min_val = cls._to_uint8_robust(dst_roi)
        src_8, _, _ = cls._to_uint8_robust(src_roi, norm_scale, min_val)

        # Make sure src_8 and dst_8 have 3 channels (required by seamlessClone)
        if dst_8.ndim == 2:
            dst_8_3c = cv2.cvtColor(dst_8, cv2.COLOR_GRAY2BGR)
            src_8_3c = cv2.cvtColor(src_8, cv2.COLOR_GRAY2BGR)
        else:
            dst_8_3c = dst_8
            src_8_3c = src_8

        # Prepare mask: must be uint8, 0 or 255
        m_8 = (mask > 0).astype(np.uint8) * 255

        # OpenCV seamlessClone requires mask NOT to touch the image boundary
        m_8[0, :] = 0
        m_8[-1, :] = 0
        m_8[:, 0] = 0
        m_8[:, -1] = 0

        # Also zero out 1 additional border pixel for safety
        if h > 4 and w > 4:
            m_8[1, :] = 0
            m_8[-2, :] = 0
            m_8[:, 1] = 0
            m_8[:, -2] = 0

        # Check if mask has any pixels left
        white_pts = np.where(m_8 > 0)
        if len(white_pts[0]) == 0:
            return dst_roi.copy()

        # Center of mask bounding box
        cy = int((np.min(white_pts[0]) + np.max(white_pts[0])) // 2)
        cx = int((np.min(white_pts[1]) + np.max(white_pts[1])) // 2)
        center = (cx, cy)

        try:
            cloned_8_3c = cv2.seamlessClone(src_8_3c, dst_8_3c, m_8, center, clone_mode)
            if dst_roi.ndim == 2:
                cloned_8 = cv2.cvtColor(cloned_8_3c, cv2.COLOR_BGR2GRAY)
            else:
                cloned_8 = cloned_8_3c

            # Map the 8-bit cloned correction back to native precision
            delta_8 = cloned_8.astype(np.float32) - dst_8.astype(np.float32)
            delta_native = delta_8 / norm_scale

            result = dst_roi.astype(np.float32) + delta_native

            if np.issubdtype(dst_roi.dtype, np.floating):
                return np.clip(result, 0.0, 1.0).astype(dst_roi.dtype)
            elif dst_roi.dtype == np.uint16:
                return np.clip(result, 0.0, 65535.0).astype(np.uint16)
            elif dst_roi.dtype == np.uint8:
                return np.clip(result, 0.0, 255.0).astype(np.uint8)
            return result.astype(dst_roi.dtype)

        except Exception:
            # Fallback to simple alpha replacement if seamlessClone fails on geometry
            m_feather = cls.feather_mask(mask, 3)
            return cls.apply_opacity(dst_roi, src_roi, m_feather, 1.0)

    @classmethod
    def blend_laplacian_pyramid(
        cls,
        dst_roi: np.ndarray,
        src_roi: np.ndarray,
        mask: np.ndarray,
        feather_radius: int = 3,
        num_levels: int = 4,
    ) -> np.ndarray:
        """
        Multi-band Laplacian pyramid blending.
        Runs natively on float32 arrays without 8-bit quantization.
        Ideal for smooth, low-gradient background nebulosity to avoid Poisson halos.
        """
        h, w = dst_roi.shape[:2]
        if h < 8 or w < 8:
            return dst_roi.copy()

        # Ensure float32 representation
        dst_f = dst_roi.astype(np.float32)
        src_f = src_roi.astype(np.float32)

        # Pad images to power of 2 multiples for clean pyramid downsampling
        max_div = 2 ** num_levels
        pad_h = (max_div - (h % max_div)) % max_div
        pad_w = (max_div - (w % max_div)) % max_div

        if pad_h > 0 or pad_w > 0:
            dst_f = cv2.copyMakeBorder(dst_f, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            src_f = cv2.copyMakeBorder(src_f, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask_padded = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        else:
            mask_padded = mask

        m_feather = cls.feather_mask(mask_padded, feather_radius)
        if dst_f.ndim == 3 and m_feather.ndim == 2:
            m_feather = m_feather[:, :, np.newaxis]

        # Build Gaussian pyramids
        G_src = [src_f]
        G_dst = [dst_f]
        G_mask = [m_feather]

        for i in range(num_levels):
            G_src.append(cv2.pyrDown(G_src[-1]))
            G_dst.append(cv2.pyrDown(G_dst[-1]))
            m_down = cv2.pyrDown(G_mask[-1])
            if dst_f.ndim == 3 and m_down.ndim == 2:
                m_down = m_down[:, :, np.newaxis]
            G_mask.append(m_down)

        # Build Laplacian pyramids
        L_src = [G_src[-1]]
        L_dst = [G_dst[-1]]

        for i in range(num_levels, 0, -1):
            src_up = cv2.pyrUp(G_src[i], dstsize=(G_src[i - 1].shape[1], G_src[i - 1].shape[0]))
            dst_up = cv2.pyrUp(G_dst[i], dstsize=(G_dst[i - 1].shape[1], G_dst[i - 1].shape[0]))
            L_src.append(G_src[i - 1] - src_up)
            L_dst.append(G_dst[i - 1] - dst_up)

        # Reverse so level 0 is base
        L_src.reverse()
        L_dst.reverse()

        # Blend pyramids at each level
        L_blend = []
        for i in range(num_levels):
            m_lvl = G_mask[i]
            l_b = L_src[i] * m_lvl + L_dst[i] * (1.0 - m_lvl)
            L_blend.append(l_b)

        # Blend coarsest Gaussian level
        m_top = G_mask[num_levels]
        top_blend = G_src[num_levels] * m_top + G_dst[num_levels] * (1.0 - m_top)

        # Reconstruct by recursive upsampling
        reconstructed = top_blend
        for i in range(num_levels - 1, -1, -1):
            up = cv2.pyrUp(reconstructed, dstsize=(L_blend[i].shape[1], L_blend[i].shape[0]))
            reconstructed = up + L_blend[i]

        # Crop back padding
        result = reconstructed[:h, :w]

        if np.issubdtype(dst_roi.dtype, np.floating):
            return np.clip(result, 0.0, 1.0).astype(dst_roi.dtype)
        elif dst_roi.dtype == np.uint16:
            return np.clip(result, 0.0, 65535.0).astype(np.uint16)
        elif dst_roi.dtype == np.uint8:
            return np.clip(result, 0.0, 255.0).astype(np.uint8)
        return result.astype(dst_roi.dtype)

    @classmethod
    def inpaint_shiftmap(cls, dst_roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Auto-heal inpainting using cv2.xphoto.inpaint with INPAINT_SHIFTMAP.
        Searches automatically for best matching textures from surrounding context.
        """
        h, w = dst_roi.shape[:2]
        if h < 5 or w < 5:
            return dst_roi.copy()

        m_8 = (mask > 0).astype(np.uint8)
        dst_8, norm_scale, min_val = cls._to_uint8_robust(dst_roi)

        try:
            if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "inpaint"):
                # cv2.xphoto.inpaint(src, mask, dst, algorithmType)
                inpainted_8 = cv2.xphoto.inpaint(
                    dst_8,
                    m_8,
                    algorithmType=cv2.xphoto.INPAINT_SHIFTMAP,
                )
            else:
                # Fallback to Telea or Navier-Stokes inpainting if contrib is missing
                inpaint_mask = (mask > 0).astype(np.uint8) * 255
                inpainted_8 = cv2.inpaint(dst_8, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            delta_8 = inpainted_8.astype(np.float32) - dst_8.astype(np.float32)
            delta_native = delta_8 / norm_scale

            result = dst_roi.astype(np.float32) + delta_native

            if np.issubdtype(dst_roi.dtype, np.floating):
                return np.clip(result, 0.0, 1.0).astype(dst_roi.dtype)
            elif dst_roi.dtype == np.uint16:
                return np.clip(result, 0.0, 65535.0).astype(np.uint16)
            elif dst_roi.dtype == np.uint8:
                return np.clip(result, 0.0, 255.0).astype(np.uint8)
            return result.astype(dst_roi.dtype)

        except Exception:
            # Fallback to standard inpaint
            try:
                inpaint_mask = (mask > 0).astype(np.uint8) * 255
                inpainted_8 = cv2.inpaint(dst_8, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                delta_8 = inpainted_8.astype(np.float32) - dst_8.astype(np.float32)
                result = dst_roi.astype(np.float32) + (delta_8 / norm_scale)
                if np.issubdtype(dst_roi.dtype, np.floating):
                    return np.clip(result, 0.0, 1.0).astype(dst_roi.dtype)
                elif dst_roi.dtype == np.uint16:
                    return np.clip(result, 0.0, 65535.0).astype(np.uint16)
                return np.clip(result, 0.0, 255.0).astype(np.uint8)
            except Exception:
                return dst_roi.copy()

    @staticmethod
    def _to_uint8_robust(arr: np.ndarray, fixed_scale: float = None, fixed_min: float = None):
        """Converts array to uint8 while recording scale factor for delta restoration."""
        if np.issubdtype(arr.dtype, np.floating):
            scale = fixed_scale if fixed_scale is not None else 255.0
            min_v = fixed_min if fixed_min is not None else 0.0
            clipped = np.clip((arr - min_v) * scale, 0.0, 255.0).astype(np.uint8)
            return clipped, scale, min_v
        elif arr.dtype == np.uint16:
            scale = fixed_scale if fixed_scale is not None else (255.0 / 65535.0)
            min_v = fixed_min if fixed_min is not None else 0.0
            u8 = (arr.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
            return u8, scale, min_v
        else:
            scale = 1.0
            min_v = 0.0
            return arr.copy(), scale, min_v
'''

# ============================================================================
# Module: dg_patch_tool.core.history
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.core.history"] = r'''
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
'''

# ============================================================================
# Module: dg_patch_tool.core.image_buffer
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.core.image_buffer"] = r'''
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


class ImageBuffer:
    """
    Manages the astrophotography image data in memory.
    Preserves exact original bit depth, channel configuration, and dynamic range
    for non-destructive editing and writing back to Siril.
    """

    def __init__(self, raw_data: np.ndarray):
        self.original_raw = raw_data
        self.original_dtype = raw_data.dtype
        self.original_shape = raw_data.shape
        self.is_chw = False

        # Standardize layout to (H, W) for mono or (H, W, C) for color
        standardized = raw_data.copy()
        if standardized.ndim == 3:
            if standardized.shape[0] in (1, 3, 4) and standardized.shape[0] < standardized.shape[1]:
                # Channel-first (C, H, W) -> transpose to (H, W, C)
                self.is_chw = True
                standardized = np.transpose(standardized, (1, 2, 0))

            if standardized.shape[2] == 1:
                # Squeeze single channel to 2D
                standardized = standardized[:, :, 0]
            elif standardized.shape[2] > 3:
                # Keep first 3 channels (RGB)
                standardized = standardized[:, :, :3]

        # Flip vertically to correct for FITS native bottom-up row order
        # (matches Siril screen display where row 0 is at the top)
        standardized = np.ascontiguousarray(np.flipud(standardized))

        self.data = standardized
        self.height, self.width = self.data.shape[:2]
        self.is_color = (self.data.ndim == 3 and self.data.shape[2] == 3)

        # Base display image (uint8 RGB or Grayscale) for Qt
        self._display_uint8 = self._convert_to_display_uint8(self.data)

    def _convert_to_display_uint8(self, arr: np.ndarray) -> np.ndarray:
        """Converts internal array to 8-bit uint8 for QImage display."""
        if np.issubdtype(arr.dtype, np.floating):
            # Already stretched non-linear data in [0.0, 1.0]
            clipped = np.clip(arr * 255.0, 0.0, 255.0)
            return clipped.astype(np.uint8)
        elif arr.dtype == np.uint16:
            # 16-bit to 8-bit
            return (arr >> 8).astype(np.uint8)
        elif arr.dtype == np.uint8:
            return arr.copy()
        else:
            # General fallback
            arr_f = arr.astype(np.float32)
            max_val = np.nanmax(arr_f)
            if max_val > 0:
                return np.clip((arr_f / max_val) * 255.0, 0.0, 255.0).astype(np.uint8)
            return np.zeros_like(arr, dtype=np.uint8)

    def get_qimage(self) -> QImage:
        """Constructs a QImage of the entire current working buffer."""
        return self.create_qimage_from_array(self._display_uint8)

    def get_qpixmap(self) -> QPixmap:
        """Constructs a QPixmap of the entire current working buffer."""
        return QPixmap.fromImage(self.get_qimage())

    def create_qimage_from_array(self, uint8_arr: np.ndarray) -> QImage:
        """Converts an 8-bit numpy array to QImage."""
        h, w = uint8_arr.shape[:2]
        if uint8_arr.ndim == 2:
            # Grayscale 8-bit
            stride = uint8_arr.strides[0]
            # Ensure memory is contiguous
            c_arr = np.ascontiguousarray(uint8_arr)
            qimg = QImage(c_arr.data, w, h, stride, QImage.Format.Format_Grayscale8)
            # Keep reference to underlying array to prevent garbage collection
            qimg._arr_ref = c_arr
            return qimg
        else:
            # RGB 8-bit (H, W, 3)
            c_arr = np.ascontiguousarray(uint8_arr)
            stride = c_arr.strides[0]
            qimg = QImage(c_arr.data, w, h, stride, QImage.Format.Format_RGB888)
            qimg._arr_ref = c_arr
            return qimg

    def get_roi(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Extracts a copy of the ROI in native dtype."""
        x0 = max(0, min(x, self.width - 1))
        y0 = max(0, min(y, self.height - 1))
        x1 = max(0, min(x + w, self.width))
        y1 = max(0, min(y + h, self.height))
        return self.data[y0:y1, x0:x1].copy()

    def set_roi(self, x: int, y: int, w: int, h: int, new_roi: np.ndarray):
        """Updates internal data and display buffer at the specified ROI."""
        x0 = max(0, min(x, self.width - 1))
        y0 = max(0, min(y, self.height - 1))
        x1 = max(0, min(x + w, self.width))
        y1 = max(0, min(y + h, self.height))

        roi_h = y1 - y0
        roi_w = x1 - x0

        if roi_h <= 0 or roi_w <= 0:
            return

        patch_crop = new_roi[:roi_h, :roi_w]
        self.data[y0:y1, x0:x1] = patch_crop

        # Update display buffer ROI
        display_patch = self._convert_to_display_uint8(patch_crop)
        self._display_uint8[y0:y1, x0:x1] = display_patch

    def get_display_roi_qimage(self, x: int, y: int, w: int, h: int) -> QImage:
        """Returns QImage of a subregion from the display buffer."""
        x0 = max(0, min(x, self.width - 1))
        y0 = max(0, min(y, self.height - 1))
        x1 = max(0, min(x + w, self.width))
        y1 = max(0, min(y + h, self.height))
        sub_arr = self._display_uint8[y0:y1, x0:x1]
        return self.create_qimage_from_array(sub_arr)

    def export_for_siril(self) -> np.ndarray:
        """
        Prepares the working buffer to be sent back to Siril,
        matching the original shape, orientation (CHW vs HWC), dtype,
        and inverting the vertical flip back to native FITS bottom-up row order.
        """
        # Restore native FITS bottom-up row order
        unflipped = np.ascontiguousarray(np.flipud(self.data))
        out = unflipped.astype(self.original_dtype, copy=True)
        if self.is_chw:
            # (H, W, C) -> (C, H, W)
            if out.ndim == 2:
                out = out[np.newaxis, :, :]
            else:
                out = np.transpose(out, (2, 0, 1))
        elif len(self.original_shape) == 3 and self.original_shape[2] == 1 and out.ndim == 2:
            out = out[:, :, np.newaxis]

        return out
'''

# ============================================================================
# Module: dg_patch_tool.core.safety_check
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.core.safety_check"] = r'''
import numpy as np


def check_linear_data(image_array: np.ndarray) -> tuple[bool, str, dict]:
    """
    Analyzes the histogram/dynamic range of the loaded image array.
    Starless astrophotography images in Siril should be stretched (non-linear)
    before running the patch/heal tool.
    """
    if image_array is None or image_array.size == 0:
        return False, "", {}

    # Sample data if image is large for instant analysis
    sample = image_array
    if sample.size > 2_000_000:
        step = int(np.ceil(np.sqrt(sample.size / 1_000_000)))
        sample = sample[::step, ::step]

    # Normalize to [0.0, 1.0] for calculation without artificially stretching
    if np.issubdtype(sample.dtype, np.floating):
        max_val = float(np.nanmax(sample))
        if max_val <= 1.0:
            norm_sample = np.clip(sample, 0.0, 1.0)
        elif max_val <= 255.0:
            norm_sample = np.clip(sample / 255.0, 0.0, 1.0)
        else:
            norm_sample = np.clip(sample / 65535.0, 0.0, 1.0)
    elif sample.dtype == np.uint16:
        norm_sample = sample.astype(np.float32) / 65535.0
    elif sample.dtype == np.uint8:
        norm_sample = sample.astype(np.float32) / 255.0
    else:
        norm_sample = np.clip(sample.astype(np.float32) / 65535.0, 0.0, 1.0)

    valid_mask = np.isfinite(norm_sample)
    if not np.any(valid_mask):
        return False, "", {}

    valid_pixels = norm_sample[valid_mask]
    median_val = float(np.median(valid_pixels))
    p95_val = float(np.percentile(valid_pixels, 95))
    p99_val = float(np.percentile(valid_pixels, 99))
    mean_val = float(np.mean(valid_pixels))

    stats = {
        "median": median_val,
        "mean": mean_val,
        "p95": p95_val,
        "p99": p99_val,
        "dtype": str(image_array.dtype),
        "shape": image_array.shape,
    }

    # In unstretched linear astro images, the sky background is typically < 0.01-0.02,
    # with 95% of pixels compressed below 0.04-0.05.
    # Stretched starless images have sky background raised to ~0.10 - 0.25 and median > 0.05.
    is_linear = (median_val < 0.02) and (p95_val < 0.06)

    if is_linear:
        message = (
            "This image doesn't look stretched (histogram is heavily compressed toward zero).\n\n"
            "Starless cleanup is typically performed on stretched (non-linear) data.\n"
            "Do you want to continue anyway?"
        )
    else:
        message = ""

    return is_linear, message, stats
'''

# ============================================================================
# Module: dg_patch_tool.core.settings
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.core.settings"] = r'''
"""
User configuration and settings persistence for DG_Patch_Tool.
Persists tool, algorithm, opacity, and feather settings across sessions using QSettings.
"""

from PyQt6.QtCore import QSettings
from dg_patch_tool.core.blend_engine import BlendEngine

ORG_NAME = "DaianGan"
APP_NAME = "DG_Patch_Tool"


def get_settings() -> QSettings:
    """Returns the QSettings instance for DG_Patch_Tool."""
    return QSettings(ORG_NAME, APP_NAME)


def load_user_preferences() -> dict:
    """Loads persisted user preferences with sensible fallback defaults."""
    s = get_settings()

    # Tool selection
    tool = str(s.value("tool", "lasso")).lower()
    if tool in ("rect", "rectangle"):
        tool = "rect"
    else:
        tool = "lasso"

    # Algorithm selection
    algo = str(s.value("algorithm", BlendEngine.MODE_POISSON_NORMAL))
    if algo not in BlendEngine.ALL_MODES:
        algo = BlendEngine.MODE_POISSON_NORMAL

    # Opacity (0.0 - 1.0)
    try:
        val = s.value("opacity", 1.0)
        opacity = float(val)
        opacity = max(0.0, min(1.0, opacity))
    except (ValueError, TypeError):
        opacity = 1.0

    # Feather (0 - 20)
    try:
        val = s.value("feather", 3)
        feather = int(val)
        feather = max(0, min(20, feather))
    except (ValueError, TypeError):
        feather = 3

    return {
        "tool": tool,
        "algorithm": algo,
        "opacity": opacity,
        "feather": feather,
    }


def save_user_preferences(tool: str, algorithm: str, opacity: float, feather: int):
    """Saves user preferences to persistent storage."""
    s = get_settings()
    tool_str = str(tool).lower()
    tool_val = "rect" if tool_str in ("rect", "rectangle") else "lasso"
    s.setValue("tool", tool_val)
    s.setValue("algorithm", algorithm)
    s.setValue("opacity", float(opacity))
    s.setValue("feather", int(feather))
    s.sync()
'''

# ============================================================================
# Module: dg_patch_tool.siril_bridge
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.siril_bridge"] = r'''
from contextlib import contextmanager
import numpy as np


class SirilBridge:
    """
    Interface bridge to Siril using sirilpy.
    Provides thread-safe access to Siril's loaded image buffer via image_lock(),
    with fallback support for standalone testing.
    """

    def __init__(self):
        self.siril = None
        self.connected = False
        self._try_connect()

    def _try_connect(self):
        try:
            import sirilpy
            # Try connecting to Siril instance
            if hasattr(sirilpy, "SirilInterface"):
                self.siril = sirilpy.SirilInterface()
                # Check connection
                if hasattr(self.siril, "connect"):
                    self.siril.connect()
                self.connected = True
        except Exception:
            self.siril = None
            self.connected = False

    def is_connected(self) -> bool:
        return self.connected and self.siril is not None

    @contextmanager
    def image_lock(self):
        """Context manager to lock the Siril image buffer during I/O."""
        if self.is_connected() and hasattr(self.siril, "image_lock"):
            with self.siril.image_lock():
                yield
        else:
            yield

    def get_image_pixeldata(self) -> np.ndarray | None:
        """Fetches the currently active image from Siril as a numpy array."""
        if not self.is_connected():
            return None

        with self.image_lock():
            if hasattr(self.siril, "get_image_pixeldata"):
                return self.siril.get_image_pixeldata()
            elif hasattr(self.siril, "get_image"):
                return self.siril.get_image()
        return None

    def set_image_pixeldata(self, data: np.ndarray, history_message: str = "DG_Patch_Tool") -> bool:
        """
        Sends the edited pixel array back to Siril.
        Saves an undo state first so Siril's Undo button is enabled.
        """
        if not self.is_connected():
            return False

        with self.image_lock():
            # Save undo state in Siril before modifying pixel data
            if hasattr(self.siril, "undo_save_state"):
                try:
                    self.siril.undo_save_state(history_message[:70])
                except Exception:
                    pass

            if hasattr(self.siril, "set_image_pixeldata"):
                self.siril.set_image_pixeldata(data)
                return True
            elif hasattr(self.siril, "set_image"):
                self.siril.set_image(data)
                return True
        return False

    def log(self, message: str) -> bool:
        """
        Writes a line to Siril's own log/console panel.
        Falls back to print() if Siril is not connected (e.g. standalone mode).
        """
        if self.is_connected() and hasattr(self.siril, "log"):
            try:
                self.siril.log(message)
                return True
            except Exception:
                pass
        print(message, flush=True)
        return False

    @staticmethod
    def generate_synthetic_starless() -> np.ndarray:
        """
        Generates a realistic synthetic starless astronomical image (1200x800, float32 RGB)
        featuring a stretched emission nebula gradient and several residual star halo artifacts,
        allowing immediate standalone testing.
        """
        h, w = 800, 1200
        y, x = np.ogrid[:h, :w]

        # Background sky gradient (~0.12 sky level, non-linear stretched)
        bg_r = 0.13 + 0.03 * np.sin(x / 400.0) * np.cos(y / 300.0)
        bg_g = 0.11 + 0.02 * np.cos(x / 500.0)
        bg_b = 0.14 + 0.04 * np.sin((x + y) / 600.0)

        img = np.zeros((h, w, 3), dtype=np.float32)
        img[:, :, 0] = bg_r
        img[:, :, 1] = bg_g
        img[:, :, 2] = bg_b

        # Stretched Hydrogen-Alpha emission nebulosity filaments
        nebula_core = np.exp(-(((x - 600) / 250.0) ** 2 + ((y - 400) / 180.0) ** 2))
        nebula_swirl = 0.15 * np.sin(x / 70.0) * np.cos(y / 90.0) * nebula_core
        img[:, :, 0] += 0.45 * nebula_core + nebula_swirl
        img[:, :, 1] += 0.12 * nebula_core
        img[:, :, 2] += 0.22 * nebula_core

        # Add faint noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.008, (h, w, 3)).astype(np.float32)
        img += noise

        # Add typical starless separation artifacts (residual star halos and blotches)
        artifacts = [
            (350, 260, 22, [0.45, 0.35, 0.25]),   # Bright star halo
            (750, 480, 18, [0.30, 0.40, 0.55]),   # Blue OIII star halo
            (520, 320, 14, [0.50, 0.30, 0.20]),   # Orange star residue
            (900, 200, 25, [0.35, 0.35, 0.35]),   # White diffraction remnant
            (220, 580, 16, [0.25, 0.25, 0.40]),   # Faint ring artifact
        ]

        for ax, ay, r, col in artifacts:
            dist_sq = (x - ax) ** 2 + (y - ay) ** 2
            # Characteristic starless remnant: a ring or soft gaussian core
            halo = 0.35 * np.exp(-dist_sq / (2.0 * (r ** 2)))
            for c in range(3):
                img[:, :, c] += halo * col[c]

        return np.clip(img, 0.0, 1.0)
'''

# ============================================================================
# Module: dg_patch_tool.ui
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.ui"] = r'''
from .main_window import MainWindow
from .canvas_view import CanvasView
from .canvas_scene import CanvasScene
from .toolbar import PatchToolBar
from .selection_tools import SelectionTool, SelectionData

__all__ = [
    "MainWindow",
    "CanvasView",
    "CanvasScene",
    "PatchToolBar",
    "SelectionTool",
    "SelectionData",
]
'''

# ============================================================================
# Module: dg_patch_tool.ui.canvas_scene
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.ui.canvas_scene"] = r'''
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (
    QPen,
    QColor,
    QBrush,
    QPixmap,
    QPainterPath,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
)


class CanvasScene(QGraphicsScene):
    """
    QGraphicsScene that hosts the astronomical image, drawing overlays,
    destination selection, source sampling indicator, and live patch preview.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Base image item
        self.base_item = QGraphicsPixmapItem()
        self.base_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.addItem(self.base_item)

        # Live patch preview item (rendered directly over the destination ROI)
        self.preview_item = QGraphicsPixmapItem()
        self.preview_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.preview_item.setZValue(10)
        self.preview_item.setVisible(False)
        self.addItem(self.preview_item)

        # Drawing outline (while user is drawing lasso or rectangle)
        self.drawing_item = QGraphicsPathItem()
        pen_draw = QPen(QColor(0, 210, 255, 230), 1.5, Qt.PenStyle.DashLine)
        pen_draw.setCosmetic(True)
        self.drawing_item.setPen(pen_draw)
        self.drawing_item.setBrush(QBrush(QColor(0, 210, 255, 35)))
        self.drawing_item.setZValue(20)
        self.drawing_item.setVisible(False)
        self.addItem(self.drawing_item)

        # Destination selection boundary
        self.dest_outline_item = QGraphicsPolygonItem()
        pen_dest = QPen(QColor(0, 230, 255, 240), 1.5, Qt.PenStyle.DashLine)
        pen_dest.setCosmetic(True)
        self.dest_outline_item.setPen(pen_dest)
        self.dest_outline_item.setBrush(QBrush(QColor(0, 230, 255, 25)))
        self.dest_outline_item.setZValue(15)
        self.dest_outline_item.setVisible(False)
        self.addItem(self.dest_outline_item)

        # Source indicator boundary (shows area being sampled)
        self.source_outline_item = QGraphicsPolygonItem()
        pen_source = QPen(QColor(255, 215, 0, 230), 1.5, Qt.PenStyle.DotLine)
        pen_source.setCosmetic(True)
        self.source_outline_item.setPen(pen_source)
        self.source_outline_item.setBrush(QBrush(QColor(255, 215, 0, 20)))
        self.source_outline_item.setZValue(14)
        self.source_outline_item.setVisible(False)
        self.addItem(self.source_outline_item)

    def set_base_pixmap(self, pixmap: QPixmap):
        self.base_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def update_base_pixmap(self, pixmap: QPixmap):
        self.base_item.setPixmap(pixmap)

    def show_drawing_path(self, path: QPainterPath):
        self.drawing_item.setPath(path)
        self.drawing_item.setVisible(True)

    def hide_drawing_path(self):
        self.drawing_item.setVisible(False)
        self.drawing_item.setPath(QPainterPath())

    def set_destination_selection(self, polygon: QPolygonF):
        self.dest_outline_item.setPolygon(polygon)
        self.dest_outline_item.setVisible(True)

    def hide_destination_selection(self):
        self.dest_outline_item.setVisible(False)

    def set_source_indicator(self, polygon: QPolygonF):
        self.source_outline_item.setPolygon(polygon)
        self.source_outline_item.setVisible(True)

    def hide_source_indicator(self):
        self.source_outline_item.setVisible(False)

    def set_preview_pixmap(self, pixmap: QPixmap, x: int, y: int):
        self.preview_item.setPixmap(pixmap)
        self.preview_item.setPos(x, y)
        self.preview_item.setVisible(True)

    def hide_preview(self):
        self.preview_item.setVisible(False)

    def clear_selection_overlays(self):
        self.hide_drawing_path()
        self.hide_destination_selection()
        self.hide_source_indicator()
        self.hide_preview()
'''

# ============================================================================
# Module: dg_patch_tool.ui.canvas_view
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.ui.canvas_view"] = r'''
from PyQt6.QtCore import Qt, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QPainter,
    QPainterPath,
    QPolygonF,
    QMouseEvent,
    QWheelEvent,
    QKeyEvent,
    QCursor,
    QBrush,
    QColor,
)
from PyQt6.QtWidgets import QGraphicsView

from dg_patch_tool.ui.canvas_scene import CanvasScene
from dg_patch_tool.ui.selection_tools import SelectionTool, SelectionData


class CanvasView(QGraphicsView):
    """
    High-performance QGraphicsView for large astrophotography images.
    Provides smooth pan & zoom, Lasso / Rectangle selection drawing,
    and live patch source preview on cursor movement.
    """

    cursor_position_changed = pyqtSignal(int, int)
    zoom_changed = pyqtSignal(float)
    selection_made = pyqtSignal(object)       # SelectionData
    patch_confirmed = pyqtSignal(object, tuple)  # (SelectionData, (dx, dy))
    selection_cancelled = pyqtSignal()

    STATE_READY = 0       # Waiting for selection
    STATE_DRAWING = 1     # Drawing lasso or rectangle
    STATE_PATCHING = 2    # Destination selected, moving cursor for source preview

    def __init__(self, scene: CanvasScene, parent=None):
        super().__init__(scene, parent)
        self.custom_scene = scene

        # View settings for smooth rendering
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet("QGraphicsView { border: none; background-color: #141518; }")
        self.setBackgroundBrush(QBrush(QColor("#141518")))

        # State
        self.state = self.STATE_READY
        self.active_tool = SelectionTool.TOOL_LASSO
        self.current_zoom = 1.0
        self._has_been_fitted = False

        # Panning state
        self._is_panning = False
        self._pan_start = QPointF()
        self._space_pressed = False

        # Drawing state
        self._draw_start_scene = QPointF()
        self._draw_points = []

        # Current destination selection
        self.current_selection: SelectionData | None = None

        # Source cursor tracking
        self.current_source_offset = (0, 0)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(16)  # ~60fps throttle for live blend preview
        self._preview_timer.timeout.connect(self._trigger_preview_update)
        self._pending_source_center = None

        # Callbacks
        self.preview_calculator = None  # Function: (selection, dx, dy) -> QPixmap

        self.setMouseTracking(True)
        self._update_cursor()

    def set_active_tool(self, tool_name: str):
        if str(tool_name).lower() in (SelectionTool.TOOL_RECTANGLE, "rectangle", "rect"):
            self.active_tool = SelectionTool.TOOL_RECTANGLE
        else:
            self.active_tool = SelectionTool.TOOL_LASSO
        self._update_cursor()

    def _update_cursor(self):
        if self._is_panning or self._space_pressed:
            self.setCursor(Qt.CursorShape.ClosedHandCursor if self._is_panning else Qt.CursorShape.OpenHandCursor)
        elif self.state == self.STATE_PATCHING:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    # --- Zoom and Pan ---

    def wheelEvent(self, event: QWheelEvent):
        """Smooth zoom anchored under cursor."""
        zoom_in_factor = 1.2
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
        else:
            factor = zoom_out_factor

        new_zoom = self.current_zoom * factor
        if 0.05 <= new_zoom <= 32.0:
            self.scale(factor, factor)
            self.current_zoom = new_zoom
            self.zoom_changed.emit(self.current_zoom)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self._update_cursor()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_selection()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self._update_cursor()
        else:
            super().keyReleaseEvent(event)

    # --- Mouse Events ---

    def mousePressEvent(self, event: QMouseEvent):
        # Pan with Middle Mouse OR Space + Left Mouse
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_pressed
        ):
            self._is_panning = True
            self._pan_start = event.position()
            self._update_cursor()
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            # Right click cancels selection/patching
            self.cancel_selection()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            img_rect = self.custom_scene.sceneRect()

            if self.state == self.STATE_READY:
                # Start drawing selection
                if img_rect.contains(scene_pos):
                    self.state = self.STATE_DRAWING
                    self._draw_start_scene = scene_pos
                    self._draw_points = [scene_pos]
                    event.accept()
                    return

            elif self.state == self.STATE_PATCHING:
                # Left click in patching state confirms the patch!
                if self.current_selection and self.current_source_offset != (0, 0):
                    self.patch_confirmed.emit(self.current_selection, self.current_source_offset)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        scene_pos = self.mapToScene(event.pos())
        self.cursor_position_changed.emit(int(scene_pos.x()), int(scene_pos.y()))

        # Handle Pan
        if self._is_panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return

        # Handle Selection Drawing
        if self.state == self.STATE_DRAWING:
            if self.active_tool == SelectionTool.TOOL_LASSO:
                self._draw_points.append(scene_pos)
                path = QPainterPath()
                if self._draw_points:
                    path.moveTo(self._draw_points[0])
                    for pt in self._draw_points[1:]:
                        path.lineTo(pt)
                self.custom_scene.show_drawing_path(path)
            elif self.active_tool in (SelectionTool.TOOL_RECTANGLE, "rectangle", "rect"):
                poly = SelectionTool.rect_to_polygon(self._draw_start_scene, scene_pos)
                path = QPainterPath()
                path.addPolygon(poly)
                self.custom_scene.show_drawing_path(path)
            event.accept()
            return

        # Handle Live Patch Source Preview
        if self.state == self.STATE_PATCHING and self.current_selection:
            # The cursor defines the source sample center
            src_cx = int(scene_pos.x())
            src_cy = int(scene_pos.y())
            dest_cx, dest_cy = self.current_selection.center
            dx = src_cx - dest_cx
            dy = src_cy - dest_cy

            self.current_source_offset = (dx, dy)

            # Update yellow source indicator outline
            source_poly = self.current_selection.polygon.translated(dx, dy)
            self.custom_scene.set_source_indicator(source_poly)

            # Request debounced/throttled ROI preview update
            self._pending_source_center = (dx, dy)
            if not self._preview_timer.isActive():
                self._preview_timer.start()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton) and self._is_panning:
            self._is_panning = False
            self._update_cursor()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.state == self.STATE_DRAWING:
            self.state = self.STATE_READY
            self.custom_scene.hide_drawing_path()

            # Finalize selection polygon
            scene_rect = self.custom_scene.sceneRect()
            w_img = int(scene_rect.width())
            h_img = int(scene_rect.height())

            polygon = QPolygonF()
            if self.active_tool == SelectionTool.TOOL_LASSO:
                if len(self._draw_points) > 2:
                    polygon = QPolygonF(self._draw_points)
                    # Close the polygon
                    polygon.append(self._draw_points[0])
            elif self.active_tool in (SelectionTool.TOOL_RECTANGLE, "rectangle", "rect"):
                scene_pos = self.mapToScene(event.pos())
                polygon = SelectionTool.rect_to_polygon(self._draw_start_scene, scene_pos)

            self._draw_points = []

            selection_data = SelectionTool.polygon_to_mask(polygon, w_img, h_img)
            if selection_data:
                self.set_selection(selection_data)
                self.selection_made.emit(selection_data)

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def set_selection(self, selection_data: SelectionData):
        """Sets active destination selection and transitions view to patching mode."""
        self.current_selection = selection_data
        self.custom_scene.set_destination_selection(selection_data.polygon)
        self.state = self.STATE_PATCHING
        self._update_cursor()

    def cancel_selection(self):
        """Cancels active selection and returns to ready state."""
        self.current_selection = None
        self.current_source_offset = (0, 0)
        self.custom_scene.clear_selection_overlays()
        self.state = self.STATE_READY
        self._update_cursor()
        self.selection_cancelled.emit()

    def _trigger_preview_update(self):
        """Computes and updates live preview pixmap on the destination ROI."""
        if not self.current_selection or not self.preview_calculator:
            return

        dx, dy = self.current_source_offset
        preview_pixmap = self.preview_calculator(self.current_selection, dx, dy)
        if preview_pixmap and not preview_pixmap.isNull():
            self.custom_scene.set_preview_pixmap(
                preview_pixmap,
                self.current_selection.x,
                self.current_selection.y,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._has_been_fitted:
            rect = self.custom_scene.sceneRect()
            if not rect.isEmpty() and self.viewport().width() > 50 and self.viewport().height() > 50:
                self._has_been_fitted = True
                self.fit_image_in_view()

    def fit_image_in_view(self):
        """Fits the entire astronomical image inside the viewport."""
        rect = self.custom_scene.sceneRect()
        if not rect.isEmpty() and self.viewport().width() > 10 and self.viewport().height() > 10:
            self.resetTransform()
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            # Calculate current zoom scale from transform
            self.current_zoom = self.transform().m11()
            self.zoom_changed.emit(self.current_zoom)

    def reset_zoom_to_100(self):
        """Resets canvas zoom to 100% (1:1 pixel scale)."""
        self.resetTransform()
        self.current_zoom = 1.0
        self.zoom_changed.emit(self.current_zoom)
'''

# ============================================================================
# Module: dg_patch_tool.ui.main_window
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.ui.main_window"] = r'''
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QLabel,
    QMessageBox,
    QFileDialog,
)

from dg_patch_tool.core.image_buffer import ImageBuffer
from dg_patch_tool.core.history import HistoryManager
from dg_patch_tool.core.blend_engine import BlendEngine
from dg_patch_tool.core.safety_check import check_linear_data
from dg_patch_tool.siril_bridge import SirilBridge
from dg_patch_tool.ui.canvas_scene import CanvasScene
from dg_patch_tool.ui.canvas_view import CanvasView
from dg_patch_tool.ui.toolbar import PatchToolBar
from dg_patch_tool.ui.selection_tools import SelectionData
from dg_patch_tool.core.settings import load_user_preferences, save_user_preferences


class MainWindow(QMainWindow):
    """
    Main application window for DG Patch Tool.
    Hosts canvas, image buffer, blend engine, and Siril bridge.
    """

    def __init__(self, raw_image: np.ndarray, siril_bridge: SirilBridge, parent=None):
        super().__init__(parent)
        self.raw_image = raw_image
        self.siril_bridge = siril_bridge

        self.setWindowTitle("DG_Patch_Tool — Starless Cleanup for Siril")
        self.resize(1300, 850)
        self.setStyleSheet("""
            QMainWindow {
                background: #141518;
            }
            QStatusBar {
                background: #1c1d22;
                color: #9ca3af;
                font-size: 11px;
                border-top: 1px solid #2e313b;
            }
            QMessageBox {
                background: #1c1d22;
                color: #e5e7eb;
            }
            QMessageBox QPushButton {
                background: #2b2e38;
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 4px;
                border: 1px solid #3f4452;
            }
        """)

        # Core logic components
        self.image_buffer = ImageBuffer(raw_image)
        self.history = HistoryManager(max_depth=50)

        # Cache for live preview (to avoid re-running blend when only opacity changes)
        self._cached_preview_data = None  # (dest_roi, patched_roi, feathered_mask, roi_rect)
        self._has_unapplied_changes = False

        # Build UI
        self._init_ui()

        # Connect callbacks
        self.canvas_view.preview_calculator = self._calculate_preview_roi

        # Initial display
        self._refresh_canvas()
        QTimer.singleShot(100, self.canvas_view.fit_image_in_view)

        # Check linear data safety guard
        self._check_linear_safety()

    def _init_ui(self):
        # Scene & View
        self.canvas_scene = CanvasScene(self)
        self.canvas_view = CanvasView(self.canvas_scene, self)
        self.setCentralWidget(self.canvas_view)

        # Toolbar
        self.toolbar = PatchToolBar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Status Bar
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.lbl_dims = QLabel()
        self.lbl_cursor = QLabel("X: -  Y: -")
        self.lbl_zoom = QLabel("Zoom: 100%")
        self.lbl_msg = QLabel("Draw selection around artifact (Lasso / Rect), then hover source to live-preview.")
        self.lbl_msg.setStyleSheet("color: #60a5fa; font-weight: 500;")

        self.statusbar.addWidget(self.lbl_msg, 1)
        self.statusbar.addPermanentWidget(self.lbl_dims)
        self.statusbar.addPermanentWidget(self.lbl_cursor)
        self.statusbar.addPermanentWidget(self.lbl_zoom)

        h, w = self.image_buffer.height, self.image_buffer.width
        channels_str = "RGB" if self.image_buffer.is_color else "Mono"
        dtype_str = str(self.image_buffer.original_dtype)
        self.lbl_dims.setText(f"{w} × {h} px | {channels_str} ({dtype_str})")

        # Wire Toolbar Signals
        self.toolbar.tool_changed.connect(self._on_tool_changed)
        self.toolbar.algorithm_changed.connect(self._on_algorithm_changed)
        self.toolbar.opacity_changed.connect(self._on_opacity_changed)
        self.toolbar.feather_changed.connect(self._on_feather_changed)
        self.toolbar.undo_requested.connect(self._on_undo)
        self.toolbar.redo_requested.connect(self._on_redo)
        self.toolbar.fit_requested.connect(self.canvas_view.fit_image_in_view)
        self.toolbar.zoom_100_requested.connect(self.canvas_view.reset_zoom_to_100)
        self.toolbar.apply_requested.connect(self._on_apply_to_siril)
        self.toolbar.close_requested.connect(self.close)

        # Restore user preferences from last session
        self._load_preferences()

        # Wire Canvas View Signals
        self.canvas_view.cursor_position_changed.connect(self._on_cursor_moved)
        self.canvas_view.zoom_changed.connect(self._on_zoom_changed)
        self.canvas_view.selection_made.connect(self._on_selection_made)
        self.canvas_view.patch_confirmed.connect(self._on_patch_confirmed)
        self.canvas_view.selection_cancelled.connect(self._on_selection_cancelled)

        # Keyboard shortcuts
        shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        shortcut_undo.activated.connect(self._on_undo)

        shortcut_redo = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        shortcut_redo.activated.connect(self._on_redo)
        shortcut_redo_alt = QShortcut(QKeySequence("Ctrl+Y"), self)
        shortcut_redo_alt.activated.connect(self._on_redo)

    def _check_linear_safety(self):
        """Pre-flight check for unstretched linear data."""
        is_linear, msg, stats = check_linear_data(self.raw_image)
        if is_linear:
            reply = QMessageBox.warning(
                self,
                "Linear Data Warning",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                self.close()

    def _refresh_canvas(self):
        pixmap = self.image_buffer.get_qpixmap()
        self.canvas_scene.set_base_pixmap(pixmap)

    MAX_PREVIEW_DIM = 200

    # --- Live Preview Calculation (ROI bounded with Adaptive Downsampling) ---

    def _calculate_preview_roi(self, selection: SelectionData, dx: int, dy: int) -> QPixmap | None:
        """
        Computes live patch preview restricted strictly to the ROI bounding box + margin.
        Uses adaptive downsampling for large ROIs (>256px) so preview renders in <20ms
        at smooth 60 FPS on multi-megapixel images.
        """
        if dx == 0 and dy == 0:
            return None

        # ROI Bounding Box with margin for boundary integration
        feather = self.toolbar.get_current_feather()
        algo = self.toolbar.get_current_algorithm()
        margin = max(16, feather * 2)

        x0 = max(0, selection.x - margin)
        y0 = max(0, selection.y - margin)
        x1 = min(self.image_buffer.width, selection.x + selection.w + margin)
        y1 = min(self.image_buffer.height, selection.y + selection.h + margin)

        roi_w = x1 - x0
        roi_h = y1 - y0
        if roi_w <= 4 or roi_h <= 4:
            return None

        # Source coordinates shifted by offset
        src_x0 = x0 + dx
        src_y0 = y0 + dy
        src_x1 = src_x0 + roi_w
        src_y1 = src_y0 + roi_h

        # Check bounds for source
        if src_x0 < 0 or src_y0 < 0 or src_x1 > self.image_buffer.width or src_y1 > self.image_buffer.height:
            return None

        # Extract destination and source ROIs
        dst_roi = self.image_buffer.get_roi(x0, y0, roi_w, roi_h)
        src_roi = self.image_buffer.get_roi(src_x0, src_y0, roi_w, roi_h)

        if dst_roi.shape[:2] != (roi_h, roi_w) or src_roi.shape[:2] != (roi_h, roi_w):
            return None

        # Create padded mask corresponding to the padded ROI
        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        mask_x_offset = selection.x - x0
        mask_y_offset = selection.y - y0
        roi_mask[mask_y_offset : mask_y_offset + selection.h, mask_x_offset : mask_x_offset + selection.w] = selection.mask

        max_dim = max(roi_w, roi_h)
        is_downsampled = max_dim > self.MAX_PREVIEW_DIM

        if is_downsampled:
            scale = self.MAX_PREVIEW_DIM / float(max_dim)
            new_w = max(4, int(round(roi_w * scale)))
            new_h = max(4, int(round(roi_h * scale)))

            dst_small = cv2.resize(dst_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            src_small = cv2.resize(src_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(roi_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            feather_small = max(1, int(round(feather * scale))) if feather > 0 else 0

            patched_small = BlendEngine.blend(
                dst_roi=dst_small,
                src_roi=src_small,
                mask=mask_small,
                mode=algo,
                feather_radius=feather_small,
            )
            feathered_mask_small = BlendEngine.feather_mask(mask_small, feather_small)

            sub_x = int(round(mask_x_offset * scale))
            sub_y = int(round(mask_y_offset * scale))
            sub_w = max(1, min(int(round(selection.w * scale)), new_w - sub_x))
            sub_h = max(1, min(int(round(selection.h * scale)), new_h - sub_y))

            # Cache for fast opacity slider updates
            self._cached_preview_data = {
                "is_downsampled": True,
                "x0": x0,
                "y0": y0,
                "w": roi_w,
                "h": roi_h,
                "dst_roi": dst_small,
                "patched_roi": patched_small,
                "feathered_mask": feathered_mask_small,
                "sub_x": sub_x,
                "sub_y": sub_y,
                "sub_w": sub_w,
                "sub_h": sub_h,
                "orig_sel_w": selection.w,
                "orig_sel_h": selection.h,
                "dx": dx,
                "dy": dy,
            }

            opacity = self.toolbar.get_current_opacity()
            blended_small = BlendEngine.apply_opacity(dst_small, patched_small, feathered_mask_small, opacity)
            sub_blended = blended_small[sub_y : sub_y + sub_h, sub_x : sub_x + sub_w]

            sub_uint8 = self.image_buffer._convert_to_display_uint8(sub_blended)
            qimg = self.image_buffer.create_qimage_from_array(sub_uint8)
            pixmap = QPixmap.fromImage(qimg)
            # Scale back to selection dimensions with smooth bilinear filtering
            if pixmap.width() != selection.w or pixmap.height() != selection.h:
                pixmap = pixmap.scaled(
                    selection.w,
                    selection.h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            return pixmap

        else:
            # Full native resolution (already small ROI, fast enough)
            patched_roi = BlendEngine.blend(
                dst_roi=dst_roi,
                src_roi=src_roi,
                mask=roi_mask,
                mode=algo,
                feather_radius=feather,
            )
            feathered_mask = BlendEngine.feather_mask(roi_mask, feather)

            self._cached_preview_data = {
                "is_downsampled": False,
                "x0": x0,
                "y0": y0,
                "w": roi_w,
                "h": roi_h,
                "dst_roi": dst_roi,
                "patched_roi": patched_roi,
                "feathered_mask": feathered_mask,
                "sub_x": mask_x_offset,
                "sub_y": mask_y_offset,
                "sub_w": selection.w,
                "sub_h": selection.h,
                "orig_sel_w": selection.w,
                "orig_sel_h": selection.h,
                "dx": dx,
                "dy": dy,
            }

            opacity = self.toolbar.get_current_opacity()
            blended_roi = BlendEngine.apply_opacity(dst_roi, patched_roi, feathered_mask, opacity)
            sub_blended = blended_roi[
                mask_y_offset : mask_y_offset + selection.h,
                mask_x_offset : mask_x_offset + selection.w,
            ]

            sub_uint8 = self.image_buffer._convert_to_display_uint8(sub_blended)
            qimg = self.image_buffer.create_qimage_from_array(sub_uint8)
            return QPixmap.fromImage(qimg)

    def _update_cached_preview_display(self):
        """Re-evaluates opacity on cached preview without re-computing blending."""
        if not self._cached_preview_data:
            return

        cache = self._cached_preview_data
        opacity = self.toolbar.get_current_opacity()

        blended_roi = BlendEngine.apply_opacity(
            cache["dst_roi"],
            cache["patched_roi"],
            cache["feathered_mask"],
            opacity,
        )

        sub_x = cache.get("sub_x", cache.get("selection_rel_x", 0))
        sub_y = cache.get("sub_y", cache.get("selection_rel_y", 0))
        sub_w = cache.get("sub_w", cache.get("sel_w", 0))
        sub_h = cache.get("sub_h", cache.get("sel_h", 0))

        sub_blended = blended_roi[sub_y : sub_y + sub_h, sub_x : sub_x + sub_w]

        sub_uint8 = self.image_buffer._convert_to_display_uint8(sub_blended)
        qimg = self.image_buffer.create_qimage_from_array(sub_uint8)
        pixmap = QPixmap.fromImage(qimg)

        target_w = cache.get("orig_sel_w", self.canvas_view.current_selection.w if self.canvas_view.current_selection else pixmap.width())
        target_h = cache.get("orig_sel_h", self.canvas_view.current_selection.h if self.canvas_view.current_selection else pixmap.height())

        if pixmap.width() != target_w or pixmap.height() != target_h:
            pixmap = pixmap.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.canvas_scene.set_preview_pixmap(
            pixmap,
            self.canvas_view.current_selection.x,
            self.canvas_view.current_selection.y,
        )

    # --- Interaction Callbacks ---

    def _on_selection_made(self, selection: SelectionData):
        algo = self.toolbar.get_current_algorithm()
        if algo == BlendEngine.MODE_SHIFTMAP:
            # Auto-Heal: immediately compute and display inpainting preview
            self._execute_auto_heal(selection)
        else:
            self.lbl_msg.setText("Destination selected. Move cursor over clean source area to preview, then Click to apply.")

    def _on_patch_confirmed(self, selection: SelectionData, source_offset: tuple[int, int]):
        """Commits the patch into the image buffer and history stack at full native resolution."""
        if not self._cached_preview_data:
            return

        cache = self._cached_preview_data
        opacity = self.toolbar.get_current_opacity()
        algo = self.toolbar.get_current_algorithm()
        feather = self.toolbar.get_current_feather()

        if cache.get("is_downsampled", False):
            # Compute full native resolution patch
            dx, dy = source_offset
            margin = max(16, feather * 2)
            x0 = max(0, selection.x - margin)
            y0 = max(0, selection.y - margin)
            x1 = min(self.image_buffer.width, selection.x + selection.w + margin)
            y1 = min(self.image_buffer.height, selection.y + selection.h + margin)
            roi_w = x1 - x0
            roi_h = y1 - y0

            src_x0 = x0 + dx
            src_y0 = y0 + dy
            src_x1 = src_x0 + roi_w
            src_y1 = src_y0 + roi_h

            if (src_x0 < 0 or src_y0 < 0 or
                src_x1 > self.image_buffer.width or src_y1 > self.image_buffer.height or
                roi_w <= 4 or roi_h <= 4):
                return

            dst_roi = self.image_buffer.get_roi(x0, y0, roi_w, roi_h)
            src_roi = self.image_buffer.get_roi(src_x0, src_y0, roi_w, roi_h)

            roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            mask_x_offset = selection.x - x0
            mask_y_offset = selection.y - y0
            roi_mask[mask_y_offset : mask_y_offset + selection.h, mask_x_offset : mask_x_offset + selection.w] = selection.mask

            patched_roi = BlendEngine.blend(
                dst_roi=dst_roi,
                src_roi=src_roi,
                mask=roi_mask,
                mode=algo,
                feather_radius=feather,
            )
            feathered_mask = BlendEngine.feather_mask(roi_mask, feather)

            before_roi = dst_roi
            final_roi = BlendEngine.apply_opacity(
                dst_roi,
                patched_roi,
                feathered_mask,
                opacity,
            )
        else:
            # Re-use already computed full-resolution ROI
            x0, y0, roi_w, roi_h = cache["x0"], cache["y0"], cache["w"], cache["h"]
            before_roi = cache["dst_roi"]
            feathered_mask = cache["feathered_mask"]
            final_roi = BlendEngine.apply_opacity(
                before_roi,
                cache["patched_roi"],
                feathered_mask,
                opacity,
            )

        # Record undo
        self.history.record_patch(
            x=x0,
            y=y0,
            w=roi_w,
            h=roi_h,
            before_data=before_roi,
            after_data=final_roi,
            mask=feathered_mask,
            algorithm=algo,
            opacity=opacity,
        )

        # Update image buffer
        self.image_buffer.set_roi(x0, y0, roi_w, roi_h, final_roi)

        # Update base pixmap in scene
        self.canvas_scene.update_base_pixmap(self.image_buffer.get_qpixmap())

        # Reset selection state
        self._cached_preview_data = None
        self._has_unapplied_changes = True
        self.canvas_view.cancel_selection()
        self.toolbar.update_history_state(self.history.can_undo(), self.history.can_redo())
        self.lbl_msg.setText("Patch applied. Ready for next selection.")

    def _execute_auto_heal(self, selection: SelectionData):
        """Runs ShiftMap inpainting on the active selection."""
        margin = max(24, self.toolbar.get_current_feather() * 3)
        x0 = max(0, selection.x - margin)
        y0 = max(0, selection.y - margin)
        x1 = min(self.image_buffer.width, selection.x + selection.w + margin)
        y1 = min(self.image_buffer.height, selection.y + selection.h + margin)

        roi_w = x1 - x0
        roi_h = y1 - y0

        dst_roi = self.image_buffer.get_roi(x0, y0, roi_w, roi_h)

        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        mask_x_offset = selection.x - x0
        mask_y_offset = selection.y - y0
        roi_mask[mask_y_offset : mask_y_offset + selection.h, mask_x_offset : mask_x_offset + selection.w] = selection.mask

        self.lbl_msg.setText("Calculating Auto-Heal (ShiftMap)...")
        self.statusbar.repaint()

        patched_roi = BlendEngine.inpaint_shiftmap(dst_roi, roi_mask)
        feather = self.toolbar.get_current_feather()
        feathered_mask = BlendEngine.feather_mask(roi_mask, feather)

        self._cached_preview_data = {
            "is_downsampled": False,
            "x0": x0,
            "y0": y0,
            "w": roi_w,
            "h": roi_h,
            "dst_roi": dst_roi,
            "patched_roi": patched_roi,
            "feathered_mask": feathered_mask,
            "sub_x": mask_x_offset,
            "sub_y": mask_y_offset,
            "sub_w": selection.w,
            "sub_h": selection.h,
            "orig_sel_w": selection.w,
            "orig_sel_h": selection.h,
            "dx": 1,
            "dy": 1,
        }

        # Set source offset dummy so click confirms
        self.canvas_view.current_source_offset = (1, 1)
        self._update_cached_preview_display()
        self.lbl_msg.setText("Auto-Heal computed. Adjust opacity slider or click canvas to confirm patch.")

    def _on_selection_cancelled(self):
        self._cached_preview_data = None
        self.lbl_msg.setText("Selection cleared. Draw around artifact to start.")

    def _on_tool_changed(self, tool_name: str):
        self.canvas_view.set_active_tool(tool_name)
        self._save_preferences()

    def _on_algorithm_changed(self, algo_name: str):
        self._save_preferences()
        if self.canvas_view.state == CanvasView.STATE_PATCHING:
            if algo_name == BlendEngine.MODE_SHIFTMAP and self.canvas_view.current_selection:
                self._execute_auto_heal(self.canvas_view.current_selection)
            else:
                self.canvas_view._trigger_preview_update()

    def _on_opacity_changed(self, opacity: float):
        self._save_preferences()
        if self._cached_preview_data:
            self._update_cached_preview_display()

    def _on_feather_changed(self, feather: int):
        self._save_preferences()
        if self.canvas_view.state == CanvasView.STATE_PATCHING:
            self.canvas_view._trigger_preview_update()

    def _load_preferences(self):
        """Loads and applies persisted settings from previous session."""
        try:
            prefs = load_user_preferences()
            self.toolbar.set_current_tool(prefs["tool"])
            self.toolbar.set_current_algorithm(prefs["algorithm"])
            self.toolbar.set_current_opacity(prefs["opacity"])
            self.toolbar.set_current_feather(prefs["feather"])
            self.canvas_view.set_active_tool(prefs["tool"])
        except Exception:
            pass

    def _save_preferences(self):
        """Persists current toolbar settings to QSettings."""
        try:
            save_user_preferences(
                tool=self.toolbar.get_current_tool(),
                algorithm=self.toolbar.get_current_algorithm(),
                opacity=self.toolbar.get_current_opacity(),
                feather=self.toolbar.get_current_feather(),
            )
        except Exception:
            pass

    def _on_undo(self):
        action = self.history.undo(self.image_buffer)
        if action:
            self.canvas_scene.update_base_pixmap(self.image_buffer.get_qpixmap())
            self.toolbar.update_history_state(self.history.can_undo(), self.history.can_redo())
            if not self.history.can_undo():
                self._has_unapplied_changes = False
            self.lbl_msg.setText("Undid last patch.")

    def _on_redo(self):
        action = self.history.redo(self.image_buffer)
        if action:
            self._has_unapplied_changes = True
            self.canvas_scene.update_base_pixmap(self.image_buffer.get_qpixmap())
            self.toolbar.update_history_state(self.history.can_undo(), self.history.can_redo())
            self.lbl_msg.setText("Redid patch.")

    def _on_cursor_moved(self, x: int, y: int):
        self.lbl_cursor.setText(f"X: {x}  Y: {y}")

    def _on_zoom_changed(self, zoom: float):
        self.lbl_zoom.setText(f"Zoom: {int(zoom * 100)}%")

    def _on_apply_to_siril(self, show_confirmation: bool = True) -> bool:
        """Sends composite result to Siril via set_image_pixeldata()."""
        export_data = self.image_buffer.export_for_siril()
        if self.siril_bridge and self.siril_bridge.is_connected():
            success = self.siril_bridge.set_image_pixeldata(export_data)
            if success:
                self._has_unapplied_changes = False
                self.lbl_msg.setText("Successfully applied composite to Siril!")
                if show_confirmation:
                    QMessageBox.information(
                        self,
                        "Applied to Siril",
                        "The patched image has been successfully written back to Siril in memory.\n\n"
                        "The tool will remain open so you can continue cleaning if needed.",
                    )
                return True
            else:
                QMessageBox.critical(self, "Error", "Failed to write image data back to Siril.")
                return False
        else:
            # Standalone mode: offer to save file
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Patched Image",
                "patched_starless.tiff",
                "TIFF Image (*.tif *.tiff);;PNG Image (*.png);;FITS (*.fit *.fits)",
            )
            if file_path:
                try:
                    import cv2
                    if export_data.dtype == np.float32 or export_data.dtype == np.float64:
                        save_arr = (np.clip(export_data, 0.0, 1.0) * 65535.0).astype(np.uint16)
                    else:
                        save_arr = export_data

                    if save_arr.ndim == 3 and save_arr.shape[2] == 3:
                        save_arr = cv2.cvtColor(save_arr, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(file_path, save_arr)
                    self._has_unapplied_changes = False
                    self.lbl_msg.setText(f"Saved to {file_path}")
                    if show_confirmation:
                        QMessageBox.information(self, "Saved", f"Image successfully saved to:\n{file_path}")
                    return True
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Error saving image: {e}")
                    return False
            return False

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure image is automatically fitted to screen when window appears
        QTimer.singleShot(50, self.canvas_view.fit_image_in_view)

    def closeEvent(self, event):
        """Safeguard: warn user if closing with unapplied patch changes."""
        self._save_preferences()
        if self._has_unapplied_changes:
            reply = QMessageBox.question(
                self,
                "Unapplied Changes",
                "You have patches that have not been applied to Siril yet.\n\n"
                "Do you want to apply them before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Save:
                success = self._on_apply_to_siril(show_confirmation=False)
                if success:
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
'''

# ============================================================================
# Module: dg_patch_tool.ui.selection_tools
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.ui.selection_tools"] = r'''
from dataclasses import dataclass
import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QPainter, QPainterPath, QPolygonF, QImage


@dataclass
class SelectionData:
    """Stores selection geometry and rasterized binary mask."""
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray          # 2D uint8 (h, w) binary mask [0 or 255]
    polygon: QPolygonF        # Selection boundary in scene coordinates

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class SelectionTool:
    TOOL_LASSO = "lasso"
    TOOL_RECTANGLE = "rect"
    TOOL_RECT = "rect"

    @classmethod
    def polygon_to_mask(cls, polygon: QPolygonF, img_width: int, img_height: int) -> SelectionData | None:
        """Converts a QPolygonF in image coordinates to a cropped ROI binary mask and bounding rect."""
        if polygon.isEmpty() or polygon.count() < 3:
            return None

        boundingRect = polygon.boundingRect()
        x0 = max(0, int(np.floor(boundingRect.left())))
        y0 = max(0, int(np.floor(boundingRect.top())))
        x1 = min(img_width, int(np.ceil(boundingRect.right())))
        y1 = min(img_height, int(np.ceil(boundingRect.bottom())))

        w = x1 - x0
        h = y1 - y0

        if w <= 1 or h <= 1:
            return None

        # Render polygon into a 1-bit / 8-bit QImage
        mask_img = QImage(w, h, QImage.Format.Format_Grayscale8)
        mask_img.fill(0)

        painter = QPainter(mask_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        # Shift polygon to ROI local coordinates
        local_poly = polygon.translated(-x0, -y0)
        painter.setBrush(painter.pen().color().fromRgb(255, 255, 255))
        painter.setPen(painter.pen().color().fromRgb(255, 255, 255))
        painter.drawPolygon(local_poly)
        painter.end()

        # Convert QImage to numpy array
        ptr = mask_img.bits()
        ptr.setsize(h * mask_img.bytesPerLine())
        mask_np = np.frombuffer(ptr, dtype=np.uint8).reshape((h, mask_img.bytesPerLine()))[:, :w].copy()

        # Ensure binary 0 or 255
        mask_binary = (mask_np > 127).astype(np.uint8) * 255

        if np.max(mask_binary) == 0:
            return None

        return SelectionData(x=x0, y=y0, w=w, h=h, mask=mask_binary, polygon=polygon)

    @classmethod
    def rect_to_polygon(cls, p1: QPointF, p2: QPointF) -> QPolygonF:
        """Creates a rectangular QPolygonF between two points."""
        x0 = min(p1.x(), p2.x())
        y0 = min(p1.y(), p2.y())
        x1 = max(p1.x(), p2.x())
        y1 = max(p1.y(), p2.y())
        poly = QPolygonF()
        poly.append(QPointF(x0, y0))
        poly.append(QPointF(x1, y0))
        poly.append(QPointF(x1, y1))
        poly.append(QPointF(x0, y1))
        poly.append(QPointF(x0, y0))
        return poly
'''

# ============================================================================
# Module: dg_patch_tool.ui.toolbar
# ============================================================================
_MODULE_SOURCES["dg_patch_tool.ui.toolbar"] = r'''
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut, QDesktopServices
from PyQt6.QtWidgets import (
    QToolBar,
    QComboBox,
    QSlider,
    QLabel,
    QPushButton,
    QButtonGroup,
    QWidget,
    QHBoxLayout,
    QSizePolicy,
    QSpinBox,
)

from dg_patch_tool.core.blend_engine import BlendEngine
from dg_patch_tool.ui.selection_tools import SelectionTool


class PatchToolBar(QToolBar):
    """
    Main toolbar providing selection tools, blending algorithm selector,
    real-time opacity and feather sliders, undo/redo, and the 'Apply to Siril' button.
    """

    DONATE_URL = "https://www.paypal.com/donate/?hosted_button_id=48L9ULQ5PTS9A"

    tool_changed = pyqtSignal(str)
    algorithm_changed = pyqtSignal(str)
    opacity_changed = pyqtSignal(float)
    feather_changed = pyqtSignal(int)
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    apply_requested = pyqtSignal()
    close_requested = pyqtSignal()
    fit_requested = pyqtSignal()
    zoom_100_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Patch Tools", parent)
        self.setMovable(False)
        self.setStyleSheet("""
            QToolBar {
                background: #1c1d22;
                border-bottom: 1px solid #2e313b;
                padding: 6px 12px;
                spacing: 10px;
            }
            QLabel {
                color: #d1d5db;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton {
                background: #2b2e38;
                color: #e5e7eb;
                border: 1px solid #3f4452;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3a3e4c;
                border-color: #555b6e;
            }
            QPushButton:checked {
                background: #1e3a8a;
                border-color: #3b82f6;
                color: #ffffff;
            }
            QPushButton:disabled {
                background: #1e2026;
                color: #6b7280;
                border-color: #2b2e38;
            }
            QComboBox {
                background: #2b2e38;
                color: #e5e7eb;
                border: 1px solid #3f4452;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 180px;
            }
            QComboBox QAbstractItemView {
                background: #2b2e38;
                color: #e5e7eb;
                selection-background-color: #3b82f6;
                selection-color: #ffffff;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #374151;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #60a5fa;
                border: 1px solid #93c5fd;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #93c5fd;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        # 1. Tool Selection (Lasso vs Rect)
        self.btn_lasso = QPushButton("Lasso (L)")
        self.btn_lasso.setCheckable(True)
        self.btn_lasso.setChecked(True)
        self.btn_lasso.setToolTip("Freehand Lasso selection (Press L)")

        self.btn_rect = QPushButton("Rect (R)")
        self.btn_rect.setCheckable(True)
        self.btn_rect.setToolTip("Rectangle selection (Press R)")

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_lasso)
        self.tool_group.addButton(self.btn_rect)
        self.tool_group.buttonClicked.connect(self._on_tool_clicked)

        self.addWidget(self.btn_lasso)
        self.addWidget(self.btn_rect)

        self.addSeparator()

        # 2. Blending Algorithm
        self.combo_algo = QComboBox()
        self.combo_algo.setToolTip("Select blending / healing algorithm")
        for mode in BlendEngine.ALL_MODES:
            self.combo_algo.addItem(mode)
        self.combo_algo.setCurrentText(BlendEngine.MODE_POISSON_NORMAL)
        self.combo_algo.currentTextChanged.connect(self._on_algo_changed)
        self.addWidget(self.combo_algo)

        self.addSeparator()

        # 3. Opacity Slider
        self.addWidget(QLabel("Opacity:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.setFixedWidth(100)
        self.slider_opacity.setToolTip("Adjust blend opacity (0-100%) in real time")
        self.slider_opacity.valueChanged.connect(self._on_opacity_slider_changed)

        self.lbl_opacity = QLabel("100%")
        self.lbl_opacity.setFixedWidth(38)

        self.addWidget(self.slider_opacity)
        self.addWidget(self.lbl_opacity)

        self.addSeparator()

        # 4. Feather Slider
        self.addWidget(QLabel("Feather:"))
        self.slider_feather = QSlider(Qt.Orientation.Horizontal)
        self.slider_feather.setRange(0, 20)
        self.slider_feather.setValue(3)
        self.slider_feather.setFixedWidth(80)
        self.slider_feather.setToolTip("Feather selection edges (0-20px)")
        self.slider_feather.valueChanged.connect(self._on_feather_slider_changed)

        self.lbl_feather = QLabel("3 px")
        self.lbl_feather.setFixedWidth(32)

        self.addWidget(self.slider_feather)
        self.addWidget(self.lbl_feather)

        self.addSeparator()

        # 5. Undo / Redo
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo last patch (Ctrl+Z)")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_requested.emit)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setToolTip("Redo patch (Ctrl+Shift+Z / Ctrl+Y)")
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self.redo_requested.emit)

        self.addWidget(self.btn_undo)
        self.addWidget(self.btn_redo)

        self.addSeparator()

        # 6. Zoom controls
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit image in viewport")
        self.btn_fit.clicked.connect(self.fit_requested.emit)

        self.btn_100 = QPushButton("1:1")
        self.btn_100.setToolTip("View at 100% pixel scale")
        self.btn_100.clicked.connect(self.zoom_100_requested.emit)

        self.addWidget(self.btn_fit)
        self.addWidget(self.btn_100)

        # Spacer to push Apply button to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)

        # 7. Apply to Siril Button
        self.btn_apply = QPushButton("Apply to Siril")
        self.btn_apply.setToolTip("Writes composite result directly to Siril memory")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background: #2563eb;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #3b82f6;
                padding: 6px 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #3b82f6;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background: #1d4ed8;
            }
        """)
        self.btn_apply.clicked.connect(self.apply_requested.emit)
        self.addWidget(self.btn_apply)

        # Spacing separation
        btn_sep1 = QWidget()
        btn_sep1.setFixedWidth(8)
        self.addWidget(btn_sep1)

        # 8. Buy Me a Coffee Button
        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setToolTip("Buy me a coffee — Support this project")
        self.btn_coffee.setStyleSheet("""
            QPushButton {
                background: #92400e;
                color: #ffffff;
                font-size: 14px;
                border: 1px solid #b45309;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #b45309;
                border-color: #d97706;
            }
            QPushButton:pressed {
                background: #78350f;
            }
        """)
        self.btn_coffee.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.DONATE_URL)))
        self.addWidget(self.btn_coffee)

        # Spacing separation
        btn_sep2 = QWidget()
        btn_sep2.setFixedWidth(8)
        self.addWidget(btn_sep2)

        # 9. Close Window Button
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close the DG_Patch_Tool window")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: #374151;
                color: #f3f4f6;
                font-weight: 500;
                border: 1px solid #4b5563;
                padding: 6px 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #4b5563;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #1f2937;
            }
        """)
        self.btn_close.clicked.connect(self.close_requested.emit)
        self.addWidget(self.btn_close)

        # Shortcuts
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        shortcut_lasso = QShortcut(QKeySequence("L"), self)
        shortcut_lasso.activated.connect(lambda: self.btn_lasso.click())

        shortcut_rect = QShortcut(QKeySequence("R"), self)
        shortcut_rect.activated.connect(lambda: self.btn_rect.click())

    def _on_tool_clicked(self, button):
        if button == self.btn_lasso:
            self.tool_changed.emit(SelectionTool.TOOL_LASSO)
        else:
            self.tool_changed.emit(SelectionTool.TOOL_RECTANGLE)

    def _on_algo_changed(self, algo_name: str):
        self.algorithm_changed.emit(algo_name)

    def _on_opacity_slider_changed(self, value: int):
        self.lbl_opacity.setText(f"{value}%")
        self.opacity_changed.emit(value / 100.0)

    def _on_feather_slider_changed(self, value: int):
        self.lbl_feather.setText(f"{value} px")
        self.feather_changed.emit(value)

    def update_history_state(self, can_undo: bool, can_redo: bool):
        self.btn_undo.setEnabled(can_undo)
        self.btn_redo.setEnabled(can_redo)

    def get_current_tool(self) -> str:
        if self.btn_rect.isChecked():
            return SelectionTool.TOOL_RECTANGLE
        return SelectionTool.TOOL_LASSO

    def set_current_tool(self, tool: str):
        if str(tool).lower() in (SelectionTool.TOOL_RECTANGLE, "rectangle", "rect"):
            self.btn_rect.setChecked(True)
            self.btn_lasso.setChecked(False)
            self.tool_changed.emit(SelectionTool.TOOL_RECTANGLE)
        else:
            self.btn_lasso.setChecked(True)
            self.btn_rect.setChecked(False)
            self.tool_changed.emit(SelectionTool.TOOL_LASSO)

    def get_current_algorithm(self) -> str:
        return self.combo_algo.currentText()

    def set_current_algorithm(self, algo_name: str):
        if algo_name in BlendEngine.ALL_MODES:
            self.combo_algo.setCurrentText(algo_name)

    def get_current_opacity(self) -> float:
        return self.slider_opacity.value() / 100.0

    def set_current_opacity(self, opacity: float):
        val = int(round(max(0.0, min(1.0, opacity)) * 100))
        self.slider_opacity.setValue(val)
        self.lbl_opacity.setText(f"{val}%")

    def get_current_feather(self) -> int:
        return self.slider_feather.value()

    def set_current_feather(self, feather: int):
        val = max(0, min(20, feather))
        self.slider_feather.setValue(val)
        self.lbl_feather.setText(f"{val} px")
'''

from dg_patch_tool.app import run

if __name__ == "__main__":
    sys.exit(run())
