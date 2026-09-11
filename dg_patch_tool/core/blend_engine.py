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
