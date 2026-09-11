# DG_Patch_Tool

A high-performance patch and heal cleanup tool designed for astrophotography processing in **Siril**.

Residual star halos, diffraction spikes, color blotches, and gradient artifacts often remain after separating nebulae from stars in starless workflows. **DG_Patch_Tool** allows you to seamlessly repair and heal these artifacts directly within your Siril workflow without needing intermediate TIFF exports or third-party graphic editors like Photoshop or Affinity Photo.

---

## Features

- **Direct In-Memory Workflow**: Fetches non-linear image data directly from Siril via `sirilpy.SirilInterface`, processes edits in memory, and writes results straight back with `set_image_pixeldata()`.
- **Pre-flight Dynamic Range Guard**: Automatically analyzes the histogram to warn if the image is still linear (unstretched), ensuring cleanup is performed at the proper non-linear stage.
- **ROI-Bounded Live Preview**: Astrophotography frames often exceed 30–60 megapixels. The live patch preview operates exclusively on the region of interest (bounding box + margin), delivering smooth 60 FPS feedback as you move the cursor.
- **Three Interchangeable Blending Modes**:
  1. **Poisson Normal Clone (`cv2.seamlessClone`)** *(Default)*: Ideal for uniform background areas and gradient matching. Also includes **Mixed Clone** for preserving complex underlying nebulosity.
  2. **Auto-Heal (`cv2.xphoto.inpaint` with `INPAINT_SHIFTMAP`)**: Exemplar-based texture synthesis that automatically detects optimal surrounding context without manual source selection.
  3. **Laplacian Pyramid Multi-Band Blending**: Multi-scale frequency blending with feathered masks, preventing color shifts or Poisson halos on ultra-smooth deep-sky backgrounds.
- **Real-Time Opacity Slider**: Non-destructively mixes the patched result with the original image data in real time without re-running the solver.
- **Feathering Control**: Smooth Gaussian edge transition (0–20 px) to eliminate visible seam borders.
- **Memory-Efficient Undo / Redo**: Stores only localized ROI sub-arrays, avoiding massive memory spikes on multi-gigabyte astronomical image sets.
- **Smooth Navigation**: Pan effortlessly with middle mouse or `Space` + left-click drag; zoom smoothly with the scroll wheel centered under the cursor.
- **Non-blocking Workflow**: "Apply to Siril" keeps the tool window open so you can continue inspecting and patching without restarting the script.

---

## Installation in Siril

1. Ensure Siril (version 1.2 or later) is installed.
2. In Siril, go to **Preferences > Scripts** to verify your user script directories.
3. Build the single-file script:
   ```bash
   python build/bundle.py
   ```
   This automatically generates and copies `DG_Patch_Tool.py` into your Siril scripts directory.
4. Re-scan scripts in Siril: **Scripts > Reload Scripts**.
5. **"DG_Patch_Tool"** will now appear in Siril's **Scripts** menu.

---

## Usage

1. Open your stretched (non-linear) starless image in Siril.
2. Launch **DG_Patch_Tool** from the Siril **Scripts** menu.
3. **Select Destination**:
   - Choose **Lasso (L)** or **Rectangle (R)** from the top toolbar.
   - Draw an outline around the star remnant or blotch you want to clean.
4. **Choose Source**:
   - Move your cursor to a nearby clean area of nebulosity or background sky.
   - The live preview shows in real time how the destination will look when patched.
5. **Adjust & Commit**:
   - Select your preferred algorithm from the dropdown (Poisson Mixed, Laplacian, or ShiftMap).
   - Adjust the **Opacity** and **Feather** sliders if needed.
   - **Left-Click** anywhere on the canvas to confirm and apply the patch.
   - Press **Escape** or **Right-Click** to cancel an active selection.
6. **Apply to Siril**:
   - When satisfied, click **Apply to Siril** to composite and write the edits back into Siril.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `L` | Activate Lasso Tool |
| `R` | Activate Rectangle Tool |
| `Space` + Left Drag | Pan Canvas |
| Middle Mouse Drag | Pan Canvas |
| Mouse Wheel | Zoom centered at cursor |
| Left Click (in patching state) | Commit patch |
| Right Click / `Esc` | Cancel active selection |
| `Ctrl+Z` / `Cmd+Z` | Undo last patch |
| `Ctrl+Shift+Z` / `Ctrl+Y` | Redo patch |

---

## Requirements

- Python 3.10+
- PyQt6 >= 6.5.0
- opencv-contrib-python >= 4.8.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- sirilpy (included with Siril 1.2+)

---

## Author

Daian Gan
