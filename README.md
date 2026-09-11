# DG PATCH TOOL

A high-performance patch, clone, and heal cleanup tool designed specifically for deep-sky astrophotography processing in **Siril**.

Residual star halos, diffraction spikes, color blotches, and gradient artifacts often remain after separating nebulae from stars in starless workflows (e.g., Starnet++, StarXTerminator). **DG Patch Tool** allows you to seamlessly repair and heal these artifacts directly inside your Siril workflow without intermediate TIFF exports or third-party graphic editors like Photoshop or Affinity Photo.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & How It Works](#architecture--how-it-works)
  - [In-Memory Siril Pipeline](#in-memory-siril-pipeline)
  - [Pre-Flight Dynamic Range Validation](#pre-flight-dynamic-range-validation)
  - [Adaptive Downsampled Live Preview](#adaptive-downsampled-live-preview)
  - [Memory-Bounded Undo/Redo Engine](#memory-bounded-undoredo-engine)
- [Blending Algorithms & Technical Attribution](#blending-algorithms--technical-attribution)
  - [1. Poisson Image Editing (Normal Clone & Mixed Clone)](#1-poisson-image-editing-normal-clone--mixed-clone)
  - [2. Multi-Band Laplacian Pyramid Blending](#2-multi-band-laplacian-pyramid-blending)
  - [3. Exemplar-Based Inpainting (ShiftMap Auto-Heal)](#3-exemplar-based-inpainting-shiftmap-auto-heal)
  - [4. Real-Time Alpha Compositing & Gaussian Feathering](#4-real-time-alpha-compositing--gaussian-feathering)
- [Installation in Siril](#installation-in-siril)
- [Usage Guide](#usage-guide)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Requirements](#requirements)
- [Support the Project](#support-the-project)
- [Author & License](#author--license)

---

## Overview

In modern astrophotography processing, star removal is a standard step for enhancing faint emission, reflection, and dark nebulosity. However, star removal algorithms frequently leave behind optical remnants—such as saturated cores, clipped ring halos, or chromatic aberration fringing—that need manual cosmetic correction.

**DG Patch Tool** runs as a native Siril script with a dedicated, hardware-accelerated PyQt6 interface. It operates directly on Siril's memory buffer, providing real-time interactive visual feedback, multiple specialized blending solvers, and one-click synchronization back to Siril with full undo history.

---

## Key Features

- **Direct In-Memory Workflow**: Fetches non-linear image data directly from Siril via `sirilpy.SirilInterface`, edits in memory, and commits changes back using `set_image_pixeldata()` with Siril undo state preservation.
- **Pre-flight Dynamic Range Guard**: Analyzes image histogram statistics upon launch. Warns if the image is still in a linear (unstretched) state, ensuring cleanup is performed at the proper non-linear stage.
- **Adaptive 60 FPS Live Preview**: Employs adaptive ROI downsampling during interactive cursor dragging on large 30–100+ megapixel images (>500 MB), while always computing the final committed patch at 100% full native floating-point / 16-bit resolution.
- **Four Specialized Blending Algorithms**: Switch between Poisson Normal Clone, Poisson Mixed Clone, Laplacian Pyramid Multi-Band Blending, and ShiftMap Exemplar Auto-Heal depending on the background context.
- **Real-Time Opacity & Feathering**: Fine-tune edge softness (0–20 px Gaussian feather) and blend strength (0–100% opacity) with instant visual feedback.
- **Lightweight ROI History**: Stores only localized sub-arrays for undo/redo rather than duplicating full multi-gigabyte astronomical image frames.
- **Persistent Preferences**: Remembers your preferred tool, algorithm, feathering, and opacity across sessions via `QSettings`.
- **Non-blocking Workflow**: The "Apply to Siril" action writes composite data back to Siril while keeping the window open for continuous inspection and cleanup.

---

## Architecture & How It Works

### In-Memory Siril Pipeline
```
┌─────────────────┐       sirilpy (shared memory)       ┌────────────────────────┐
│  Siril Session  │ ─────────────────────────────────>  │  ImageBuffer (Numpy)   │
│  (Active Image) │ <─────────────────────────────────  │ (Native Dtype & Shape) │
└─────────────────┘      set_image_pixeldata()          └───────────┬────────────┘
                                                                    │
                                                     ROI Slices     │  Display
                                                     (Float/Uint)   │  Buffer (8-bit)
                                                                    v
                                                        ┌───────────────────────┐
                                                        │     CanvasScene       │
                                                        │   (PyQt6 Graphics)    │
                                                        └───────────────────────┘
```

1. **Orientation & Precision Handling**: FITS images natively store pixel rows bottom-up. `ImageBuffer` normalizes orientation, dimensions, and channel layouts (grayscale 2D, mono 3D, and RGB), flipping vertically to align with screen display coordinates while preserving the original bit-depth (`float32`, `uint16`, or `uint8`). Upon export, coordinates and row ordering are inverted back to native FITS convention.
2. **Thread-Safe Image Locking**: Interacts with Siril using context-managed image locks (`bridge.image_lock()`) to prevent concurrent read/write race conditions.

### Pre-Flight Dynamic Range Validation
Astrophotographical starless cleanup is designed to operate on **stretched (non-linear)** data. If executed on unstretched linear frames, math solvers can produce clipping or severe contrast artifacts:
- The pre-flight check samples image values and computes median, mean, 95th percentile, and 99th percentile distributions.
- Linear astro frames typically exhibit sky backgrounds with median < 0.02 and 95th percentile < 0.06.
- If linear data is detected, a warning modal alerts the user with options to proceed or safely exit.

### Adaptive Downsampled Live Preview
High-resolution astrophotography sensors (e.g., Sony IMX455, IMX571) produce 26–61 MP images (up to 9576 × 6388 px). Solving differential equations (Poisson) across a large selection (e.g., 1000 × 1000 px) can require >1 second per frame:
- **Interactive Dragging**: If the selection bounding box exceeds `MAX_PREVIEW_DIM = 200` px, the ROI slice is adaptively downsampled using area interpolation (`cv2.INTER_AREA`). The solver executes in <20 ms (delivering 60 FPS), and the result is upscaled to the viewport using bilinear smoothing (`SmoothTransformation`).
- **Confirmation**: When the user left-clicks to commit the patch, `_on_patch_confirmed()` executes the blend at **100% native resolution** on the full data buffer without downsampling.

### Memory-Bounded Undo/Redo Engine
Rather than cloning full 200–800 MB image matrices into memory on each step:
- `HistoryManager` records only the localized bounding box coordinates `(x, y, w, h)`, the cropped `before_data` array, the `after_data` array, the binary mask, and the algorithm parameters.
- Undo and redo operations execute in sub-millisecond time by replacing only the relevant sub-region in `ImageBuffer`.

---

## Blending Algorithms & Technical Attribution

Different astronomical targets present distinct mathematical challenges: uniform sky backgrounds, faint nebular filaments, or complex high-contrast textures. **DG Patch Tool** incorporates four specialized techniques:

| Target Scenario | Recommended Algorithm | Characteristics |
|---|---|---|
| Uniform sky background | Poisson (Normal Clone) | Seamless illumination & boundary integration |
| Faint nebulosity / dust | Poisson (Mixed Clone) | Preserves underlying texture & filaments |
| Ultra-smooth gradients | Laplacian Pyramid Multi-Band | Native float32, zero Poisson halos |
| Fast isolated healing | Auto-Heal (ShiftMap Exemplar) | Context texture synthesis, no source needed |

### 1. Poisson Image Editing (Normal Clone & Mixed Clone)

Poisson blending solves a partial differential equation (PDE) with Dirichlet boundary conditions over the selected region, matching the Laplacian of the source while clamping boundary pixels to the destination image:

- **Poisson Normal Clone (`cv2.NORMAL_CLONE`)** *(Default)*:
  Uses the guidance field from the source donor patch directly. Perfectly matches surrounding background illumination and gradient slope across the seam boundary.
- **Poisson Mixed Clone (`cv2.MIXED_CLONE`)**:
  Computes the guidance field by taking the gradient of maximum magnitude between source and destination. This preserves underlying background structures and faint nebular filaments while seamlessly absorbing the donor patch's color.

**References & Attribution:**
- Pérez, P., Gangnet, M., & Blake, A. (2003). *Poisson Image Editing*. ACM Transactions on Graphics (SIGGRAPH '03), 22(3), 313–318. [DOI: 10.1145/882262.882269](https://doi.org/10.1145/882262.882269) | [PDF (JHU)](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
- OpenCV Seamless Cloning Module: [`cv2.seamlessClone`](https://docs.opencv.org/4.x/df/da0/group__photo__clone.html)

---

### 2. Multi-Band Laplacian Pyramid Blending

In smooth, low-contrast astronomical nebulosity, gradient-domain Poisson solvers can occasionally introduce subtle local luminance shifts or dark halos if the mean donor level differs from the target. Laplacian pyramid blending solves this by decomposing the donor and destination images into separate spatial frequency bands:

1. **Gaussian Pyramid**: Constructs progressively downsampled, low-pass filtered representations.
2. **Laplacian Pyramid**: Computes bandpass detail layers by subtracting upsampled adjacent levels.
3. **Multi-Scale Blending**: High spatial frequencies (noise, fine texture) are blended across a narrow spatial seam, while low spatial frequencies (large-scale nebular gradients) are blended across a broad transition zone.
4. **Reconstruction**: Re-synthesizes the final composite by recursive upsampling and addition.

Executed natively in 32-bit floating point precision without 8-bit quantization.

**References & Attribution:**
- Burt, P. J., & Adelson, E. H. (1983). *A Multiresolution Spline with Application to Image Mosaics*. ACM Transactions on Graphics, 2(4), 217–236. [DOI: 10.1145/245.247](https://doi.org/10.1145/245.247) | [PDF (MIT PerSci)](https://persci.mit.edu/pub_pdfs/spline83.pdf)

---

### 3. Exemplar-Based Inpainting (ShiftMap Auto-Heal)

For isolated artifacts (e.g., small star halos or hot pixel clusters) where manual donor selection is unnecessary, **Auto-Heal** synthesizes replacement texture automatically from surrounding context:

- Utilizes `cv2.xphoto.inpaint` configured with `INPAINT_SHIFTMAP`.
- Evaluates patch similarities and global gradient consistency in surrounding unmasked pixels to synthesize believable, artifact-free textures.
- Automatically falls back to Fast Marching Method (Telea) inpainting if the OpenCV contrib `xphoto` module is not installed in the Python environment.

**References & Attribution:**
- Criminisi, A., Pérez, P., & Toyama, K. (2004). *Region Filling and Object Removal by Exemplar-Based Image Inpainting*. IEEE Transactions on Image Processing, 13(9), 1200–1212. [DOI: 10.1109/TIP.2004.833105](https://doi.org/10.1109/TIP.2004.833105)
- Telea, A. (2004). *An Image Inpainting Technique Based on the Fast Marching Method*. Journal of Graphics Tools, 9(1), 23–34. [DOI: 10.1080/10867651.2004.10487596](https://doi.org/10.1080/10867651.2004.10487596)
- OpenCV Extended Photo Module: [`cv2.xphoto.inpaint`](https://docs.opencv.org/4.x/de/daa/group__xphoto.html#gaac1e089b0d5c80377464b97d80f83695)

---

### 4. Real-Time Alpha Compositing & Gaussian Feathering

Once a patch is computed by the selected solver, real-time edge blending and opacity modulation run in <1 ms without re-solving differential equations:

$$\text{Mask}_{\text{feathered}} = \operatorname{clip}\left( \text{Mask} * K_{\text{Gaussian}}(\sigma, k), 0.0, 1.0 \right)$$

$$I_{\text{final}} = I_{\text{dst}} + \left( I_{\text{patched}} - I_{\text{dst}} \right) \cdot \left( \text{Mask}_{\text{feathered}} \cdot \alpha \right)$$

where $\alpha \in [0.0, 1.0]$ is the real-time opacity value.

---

## Installation in Siril

1. Ensure **Siril** (version 1.2 or later) is installed.
2. In Siril, go to **Preferences > Scripts** to confirm your user script directory (e.g., `~/Siril/scripts` or `~/siril/scripts`).
3. Build and bundle the single-file script:
   ```bash
   python build/bundle.py
   ```
   This compiles the modular `dg_patch_tool/` package into a standalone file `DG_Patch_Tool.py` and copies it directly into detected Siril script folders.
4. In Siril, select **Scripts > Reload Scripts** (or restart Siril).
5. **"DG_Patch_Tool"** will now appear in Siril's **Scripts** menu.

---

## Usage Guide

1. Open your stretched (non-linear) starless image in Siril.
2. Launch **DG_Patch_Tool** from Siril's **Scripts** menu.
3. **Select Artifact (Destination)**:
   - Choose **Lasso (L)** or **Rect (R)** from the toolbar.
   - Left-click and drag around the star remnant, halo, or blotch.
4. **Select Clean Donor (Source)**:
   - Move your cursor to an adjacent clean region of background sky or nebulosity.
   - The live preview instantly shows the blended composite inside the selection boundary.
5. **Tune & Commit**:
   - Change blending algorithm if needed (**Poisson Normal**, **Poisson Mixed**, **Laplacian**, or **Auto-Heal**).
   - Adjust the **Opacity** and **Feather** sliders in real time.
   - **Left-Click** to commit the patch.
   - Press **Escape** or **Right-Click** at any time to cancel the selection.
6. **Save Back to Siril**:
   - When finished cleaning artifacts, click **Apply to Siril**.
   - Edits are committed directly to Siril's working buffer with an undo state saved in Siril's history.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `L` | Activate Lasso Tool |
| `R` | Activate Rectangle Tool |
| `Space` + Left Drag | Pan Canvas |
| Middle Mouse Drag | Pan Canvas |
| Mouse Wheel | Zoom centered at cursor position |
| Left Click (in patching state) | Confirm and apply patch |
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

## Support the Project

If you find this tool helpful for your astrophotography workflows, consider supporting its development:

[![Buy Me A Coffee](https://img.shields.io/badge/Donate-PayPal-blue.svg?logo=paypal)](https://www.paypal.com/donate/?hosted_button_id=48L9ULQ5PTS9A)

---

## Author & License

Developed by **Daian Gan**  
- Email: [daian@ganmedia.com](mailto:daian@ganmedia.com)  
- Web: [https://daiangan.com](https://daiangan.com)  
- GitHub: [https://github.com/daiangan/siril-dg-patch-tool](https://github.com/daiangan/siril-dg-patch-tool)
