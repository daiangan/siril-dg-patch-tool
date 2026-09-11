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
