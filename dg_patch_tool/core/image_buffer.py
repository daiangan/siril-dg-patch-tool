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
