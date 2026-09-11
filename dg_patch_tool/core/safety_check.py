import numpy as np


def check_linear_data(image_array: np.ndarray) -> tuple[bool, str, dict]:
    """
    Analyzes the histogram/dynamic range of the loaded image array.
    Starless astrophotography images in Siril should be stretched (non-linear)
    before running the patch/heal tool.

    If the data appears linear (histogram severely compressed toward zero),
    returns (True, warning_message, stats_dict).
    Otherwise returns (False, "", stats_dict).
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
