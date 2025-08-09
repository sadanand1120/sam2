import base64
import io
import numpy as np
import requests
from typing import Dict, Optional, List, Any, Union
from PIL import Image
import yaml
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sam2.features.utils import SAM2utils


def encode_image(image: Union[str, Image.Image]) -> str:
    """Encode local image file path or PIL Image to base64"""
    if isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Image must be a file path (str) or a PIL.Image.Image object.")


def encode_mask(mask: np.ndarray) -> str:
    """Encode numpy mask to base64"""
    mask_bytes = mask.astype(np.float32).tobytes()
    return base64.b64encode(mask_bytes).decode('utf-8')


def encode_mask_with_shape(mask: np.ndarray) -> tuple:
    """Encode numpy mask to base64 with shape information"""
    mask_bytes = mask.astype(np.float32).tobytes()
    encoded = base64.b64encode(mask_bytes).decode('utf-8')
    return encoded, list(mask.shape)


def decode_masks(masks_base64: str, shape: list) -> np.ndarray:
    """Decode base64 masks to numpy array"""
    masks_bytes = base64.b64decode(masks_base64)
    masks = np.frombuffer(masks_bytes, dtype=np.float32).reshape(shape)
    return masks


def decode_iou_predictions(iou_base64: str, shape: list) -> np.ndarray:
    """Decode base64 IoU predictions to numpy array"""
    iou_bytes = base64.b64decode(iou_base64)
    iou = np.frombuffer(iou_bytes, dtype=np.float32).reshape(shape)
    return iou


def decode_auto_masks(masks_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Decode auto masks from base64 encoded segmentation fields"""
    decoded_masks = []
    for mask_dict in masks_list:
        # Create a copy of the mask dict
        decoded_mask = mask_dict.copy()

        # Decode the segmentation field if it's base64 encoded
        if isinstance(mask_dict['segmentation'], str):
            # Decode the segmentation mask using the provided shape
            mask_bytes = base64.b64decode(mask_dict['segmentation'])
            shape = mask_dict.get('segmentation_shape', [])
            if shape:
                mask_array = np.frombuffer(mask_bytes, dtype=np.float32).reshape(shape)
                # Convert float32 mask back to boolean for proper visualization
                decoded_mask['segmentation'] = mask_array.astype(bool)
            else:
                # Fallback: assume it's a 2D mask and infer shape
                mask_array = np.frombuffer(mask_bytes, dtype=np.float32)
                h = w = int(np.sqrt(mask_array.size))
                # Convert float32 mask back to boolean for proper visualization
                decoded_mask['segmentation'] = mask_array.reshape(h, w).astype(bool)

        # Convert any list fields back to numpy arrays if they were originally arrays
        for key, value in decoded_mask.items():
            if key in ['bbox', 'point_coords'] and isinstance(value, list):
                decoded_mask[key] = np.array(value)
            elif key in ['area', 'predicted_iou', 'stability_score'] and isinstance(value, (int, float)):
                # These should remain as Python types for compatibility
                pass

        decoded_masks.append(decoded_mask)

    return decoded_masks


def auto_masks_to_instance_mask(
    masks_list: List[Dict[str, Any]],
    min_iou: float = 0.0,
    min_area: float = 0.0,
    assign_by: str = "iou",
    start_from: str = "low",
) -> np.ndarray:
    """Convert list of masks to an instance mask with configurable ordering.

    - Background (no mask) has id 0.
    - IDs are assigned by descending quality so that id 1 is best (e.g., highest IoU), id 2 next, etc.
    - The write order is controlled by `start_from` to determine overwrite behavior in overlaps.
      Using `start_from='low'` ensures higher-quality masks are written last so they win overlaps.
    """
    if not masks_list:
        return np.array([])

    # Filter masks using existing utils
    filtered_masks = SAM2utils.filter_masks_by_score(masks_list, min_iou=min_iou)
    filtered_masks = SAM2utils.filter_masks_by_area(filtered_masks, min_area=min_area)

    if not filtered_masks:
        return np.array([])

    # Choose sort key according to assign_by
    def _mask_key(m: Dict[str, Any]) -> float:
        if assign_by == "area":
            area_val = m.get("area")
            if isinstance(area_val, (int, float)):
                return float(area_val)
            seg = m.get("segmentation")
            return float(np.count_nonzero(seg)) if isinstance(seg, np.ndarray) else 0.0
        # default: IoU
        return float(m.get('predicted_iou', 0.0))

    # Rank masks by descending key for IDs (1=best)
    indices_desc = sorted(range(len(filtered_masks)), key=lambda i: _mask_key(filtered_masks[i]), reverse=True)
    rank_id = {idx: rank + 1 for rank, idx in enumerate(indices_desc)}

    # Determine traversal order for writing (controls overwrite semantics)
    write_reverse = (start_from == "high")
    traversal = sorted(range(len(filtered_masks)), key=lambda i: _mask_key(filtered_masks[i]), reverse=write_reverse)

    # Get image dimensions from first in traversal
    first_mask = filtered_masks[traversal[0]]['segmentation']
    height, width = first_mask.shape
    instance_mask = np.zeros((height, width), dtype=np.uint8)

    # Overwrite semantics: later writes win. IDs come from rank_id so 1=best consistently.
    for idx in traversal:
        mask_dict = filtered_masks[idx]
        mask = mask_dict['segmentation']
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        instance_mask[mask] = rank_id[idx]

    return instance_mask


def compute_overlap_stats(decoded_masks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overlap statistics for a list of decoded masks."""
    if not decoded_masks:
        return {"num_masks": 0, "overlap_pixels": 0, "max_overlap": 0, "total_pixels": 0}
    stack = np.stack([m["segmentation"].astype(np.uint8) for m in decoded_masks], axis=0)
    per_pixel_counts = stack.sum(axis=0)
    overlap_pixels = int((per_pixel_counts > 1).sum())
    max_overlap = int(per_pixel_counts.max())
    total_pixels = per_pixel_counts.size
    return {
        "num_masks": stack.shape[0],
        "overlap_pixels": overlap_pixels,
        "max_overlap": max_overlap,
        "total_pixels": total_pixels,
    }


def build_instance_colormap(num_instances: int) -> ListedColormap:
    """Colormap with black background (index 0) and spaced colors for instances.

    Colors are assigned by alternating from both ends of a base gradient to maximize perceptual separation,
    e.g., order: 0, N-1, 1, N-2, ... over a base palette.
    """
    if num_instances <= 0:
        return ListedColormap([(0, 0, 0, 1)])

    # Base palette from a continuous colormap (e.g., 'tab20' is discrete but small; use hsv for many)
    base = [plt.cm.hsv(t) for t in np.linspace(0, 1, num_instances, endpoint=False)]

    # Alternating index sequence: 0, N-1, 1, N-2, 2, N-3, ...
    order = []
    lo, hi = 0, num_instances - 1
    while lo <= hi:
        order.append(lo)
        if lo != hi:
            order.append(hi)
        lo += 1
        hi -= 1

    # Assemble RGBA colors: black for background, then ordered instance colors
    colors = [(0, 0, 0, 1)] + [tuple(base[i][:3]) + (1,) for i in order]
    return ListedColormap(colors)


def make_viz_mask_and_cmap(instance_mask: np.ndarray) -> tuple:
    """Return (viz_mask, cmap, norm) with black background and well-separated instance colors.

    - Remaps instance IDs (only for visualization) by alternating extremes over area-sorted IDs
      so large regions get well-separated colors: order on IDs by area desc, then indices 0, N-1, 1, N-2, ...
    - Uses BoundaryNorm to ensure discrete color bins per integer ID.
    """
    if instance_mask.size == 0:
        cmap = ListedColormap([(0, 0, 0, 1)])
        norm = BoundaryNorm([-0.5, 0.5], ncolors=1)
        return instance_mask, cmap, norm

    ids = np.unique(instance_mask)
    ids = ids[ids != 0]
    N = int(len(ids))
    if N == 0:
        cmap = ListedColormap([(0, 0, 0, 1)])
        norm = BoundaryNorm([-0.5, 0.5], ncolors=1)
        return instance_mask, cmap, norm

    # Compute pixel counts per id
    max_id = int(instance_mask.max())
    counts = np.bincount(instance_mask.ravel(), minlength=max_id + 1)
    id_counts = [(int(i), int(counts[i])) for i in ids]
    # Sort by area descending
    id_counts.sort(key=lambda x: x[1], reverse=True)

    # Alternating extremes order over indices 0..N-1
    order = []
    lo, hi = 0, N - 1
    while lo <= hi:
        order.append(lo)
        if lo != hi:
            order.append(hi)
        lo += 1
        hi -= 1

    # Build mapping: id -> new viz index 1..N according to alternating-extreme positions
    viz_rank_for_sorted = {idx: rank + 1 for rank, idx in enumerate(order)}
    mapping = {id_counts[i][0]: viz_rank_for_sorted[i] for i in range(N)}

    # Remap instance ids for visualization only (background stays 0)
    viz_mask = np.zeros_like(instance_mask, dtype=np.uint16)
    for inst_id, viz_id in mapping.items():
        viz_mask[instance_mask == inst_id] = viz_id

    # Build a spaced color list using golden-ratio hue stepping
    base_colors = []
    phi = (1 + 5 ** 0.5) / 2.0
    step = 1.0 / phi
    for k in range(N):
        h = (k * step) % 1.0
        r, g, b, _ = plt.cm.hsv(h)
        base_colors.append((r, g, b, 1))
    cmap = ListedColormap([(0, 0, 0, 1)] + base_colors)
    # Discrete bins for 0..N
    boundaries = np.arange(N + 2) - 0.5
    norm = BoundaryNorm(boundaries, ncolors=N + 1)
    return viz_mask, cmap, norm


def _prevent_oom_downscale_img(image: Union[str, Image.Image], target: int) -> Image.Image:
    """Return a PIL image where max(H, W) <= target; keep aspect ratio. No-op if already small."""
    img = Image.open(image) if isinstance(image, str) else image
    w, h = img.width, img.height
    max_side = max(w, h)
    if max_side <= target:
        return img
    scale = float(target) / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.LANCZOS)


def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Get headers with optional API key"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def generate_sam2_auto_masks(image: Optional[Union[str, Image.Image]] = None, image_url: Optional[str] = None,
                             base_url: str = "",
                             preset: Optional[str] = None,
                             points_per_side: Optional[int] = None,
                             points_per_batch: Optional[int] = None,
                             pred_iou_thresh: Optional[float] = None,
                             stability_score_thresh: Optional[float] = None,
                             stability_score_offset: Optional[float] = None,
                             mask_threshold: Optional[float] = None,
                             box_nms_thresh: Optional[float] = None,
                             crop_n_layers: Optional[int] = None,
                             crop_nms_thresh: Optional[float] = None,
                             crop_overlap_ratio: Optional[float] = None,
                             crop_n_points_downscale_factor: Optional[int] = None,
                             point_grids: Optional[List[List[List[float]]]] = None,
                             min_mask_region_area: Optional[float] = None,
                             use_m2m: Optional[bool] = None,
                             multimask_output: Optional[bool] = None,
                             output_mode: Optional[str] = None,
                             api_key: str = "") -> Dict:
    """Generate automatic masks using SAM2"""
    payload = {
        'preset': preset,
        'points_per_side': points_per_side,
        'points_per_batch': points_per_batch,
        'pred_iou_thresh': pred_iou_thresh,
        'stability_score_thresh': stability_score_thresh,
        'stability_score_offset': stability_score_offset,
        'mask_threshold': mask_threshold,
        'box_nms_thresh': box_nms_thresh,
        'crop_n_layers': crop_n_layers,
        'crop_nms_thresh': crop_nms_thresh,
        'crop_overlap_ratio': crop_overlap_ratio,
        'crop_n_points_downscale_factor': crop_n_points_downscale_factor,
        'point_grids': point_grids,
        'min_mask_region_area': min_mask_region_area,
        'use_m2m': use_m2m,
        'multimask_output': multimask_output,
        'output_mode': output_mode
    }

    if image:
        payload['image'] = encode_image(image)
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")

    response = requests.post(
        f"{base_url}/auto_mask",
        json=payload,
        headers=_get_headers(api_key)
    )
    response.raise_for_status()
    return response.json()


def generate_sam2_prompt_masks(image: Optional[Union[str, Image.Image]] = None, image_url: Optional[str] = None,
                               base_url: str = "",
                               point_coords: Optional[List[List[float]]] = None,
                               point_labels: Optional[List[int]] = None,
                               box: Optional[List[float]] = None,
                               mask_input: Optional[str] = None,
                               mask_input_shape: Optional[List[int]] = None,
                               multimask_output: bool = True,
                               return_logits: bool = False,
                               normalize_coords: bool = True,
                               mask_threshold: float = 0.0,
                               max_hole_area: float = 0.0,
                               max_sprinkle_area: float = 0.0,
                               api_key: str = "") -> Dict:
    """Generate prompt-based masks using SAM2"""
    payload = {
        'point_coords': point_coords,
        'point_labels': point_labels,
        'box': box,
        'mask_input': mask_input,
        'mask_input_shape': mask_input_shape,
        'multimask_output': multimask_output,
        'return_logits': return_logits,
        'normalize_coords': normalize_coords,
        'mask_threshold': mask_threshold,
        'max_hole_area': max_hole_area,
        'max_sprinkle_area': max_sprinkle_area
    }

    if image:
        payload['image'] = encode_image(image)
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")

    response = requests.post(
        f"{base_url}/prompt_mask",
        json=payload,
        headers=_get_headers(api_key)
    )
    response.raise_for_status()
    return response.json()


def get_sam2_health(base_url: str = "") -> Dict:
    """Get SAM2 server health status (no authentication required)"""
    headers = {"Content-Type": "application/json"}
    response = requests.get(f"{base_url}/health", headers=headers)
    response.raise_for_status()
    return response.json()


# Convenience functions that combine request + decode
def generate_sam2_auto_masks_decoded(image: Union[str, Image.Image] = None, **kwargs) -> List[Dict[str, Any]]:
    """Generate automatic masks and return the decoded mask dictionaries"""
    if image:
        kwargs['image'] = image

    result = generate_sam2_auto_masks(**kwargs)
    return decode_auto_masks(result['masks'])


def generate_sam2_prompt_masks_decoded(image: Union[str, Image.Image] = None, **kwargs) -> tuple:
    """Generate prompt masks and return decoded numpy arrays"""
    if image:
        kwargs['image'] = image

    result = generate_sam2_prompt_masks(**kwargs)

    masks = decode_masks(result['masks'], result['masks_shape'])
    iou_predictions = decode_iou_predictions(result['iou_predictions'], result['iou_shape'])
    low_res_masks = decode_masks(result['low_res_masks'], result['low_res_shape'])

    return masks, iou_predictions, low_res_masks


# Advanced convenience functions for common use cases
def sam2_point_mask(image: Union[str, Image.Image], point_coords: List[List[float]], point_labels: List[int], **kwargs):
    """Generate mask from point prompts"""
    return generate_sam2_prompt_masks_decoded(
        image=image,
        point_coords=point_coords,
        point_labels=point_labels,
        **kwargs
    )


def sam2_box_mask(image: Union[str, Image.Image], box: List[float], **kwargs):
    """Generate mask from box prompt"""
    return generate_sam2_prompt_masks_decoded(
        image=image,
        box=box,
        **kwargs
    )


def sam2_combined_mask(image: Union[str, Image.Image], point_coords: List[List[float]] = None,
                       point_labels: List[int] = None, box: List[float] = None, **kwargs):
    """Generate mask from combined point and box prompts"""
    return generate_sam2_prompt_masks_decoded(
        image=image,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        **kwargs
    )


if __name__ == "__main__":
    # Load server config
    config_path = "sam2/server/client/servers.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_url = config["sam2"]["base_url"]
    api_key = config["sam2"]["api_key"]

    test_image_path = "notebooks/images/cars.jpg"
    img = Image.open(test_image_path)
    print("=== SAM2 Client Demo ===")

    # Health check (no authentication required)
    health = get_sam2_health(base_url=base_url)
    print(f"Server health: {health}")

    print("=== Automatic Mask Generation Demo ===")

    # Generate coarse masks (exactly matching main file)
    print("Generating coarse masks...")
    coarse_masks = generate_sam2_auto_masks_decoded(
        test_image_path,
        base_url=base_url,
        api_key=api_key,
        preset="coarse"
    )
    print(f"Generated {len(coarse_masks)} coarse masks")
    SAM2utils.visualize_masks(img, coarse_masks)
    # Overlap analysis and instance mask demo for coarse
    coarse_stats = compute_overlap_stats(coarse_masks)
    if coarse_stats["total_pixels"] > 0:
        coarse_pct = 100.0 * coarse_stats["overlap_pixels"] / coarse_stats["total_pixels"]
        print(f"Coarse Overlap: masks={coarse_stats['num_masks']}, overlap_pixels={coarse_stats['overlap_pixels']} ({coarse_pct:.3f}%), max_overlap={coarse_stats['max_overlap']}")
    # 2x2 grid: assign_by ∈ {iou, area} × start_from ∈ {low, high}
    configs = [("iou", "low"), ("iou", "high"), ("area", "low"), ("area", "high")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (assign_by, start_from) in zip(axes.flat, configs):
        inst = auto_masks_to_instance_mask(coarse_masks, assign_by=assign_by, start_from=start_from)
        viz_inst, cmap, norm = make_viz_mask_and_cmap(inst)
        ax.imshow(viz_inst, cmap=cmap, norm=norm)
        ax.set_title(f"coarse: {assign_by}, {start_from}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Generate fine-grained masks (downscale only if needed to avoid OOM)
    small_img = _prevent_oom_downscale_img(img, target=512)
    print("Generating fine-grained masks...")
    fine_masks = generate_sam2_auto_masks_decoded(
        small_img,
        base_url=base_url,
        api_key=api_key,
        preset="fine_grained"
    )
    print(f"Generated {len(fine_masks)} fine-grained masks")
    SAM2utils.visualize_masks(small_img, fine_masks)
    # Overlap analysis and instance mask demo for fine
    fine_stats = compute_overlap_stats(fine_masks)
    if fine_stats["total_pixels"] > 0:
        fine_pct = 100.0 * fine_stats["overlap_pixels"] / fine_stats["total_pixels"]
        print(f"Fine Overlap: masks={fine_stats['num_masks']}, overlap_pixels={fine_stats['overlap_pixels']} ({fine_pct:.3f}%), max_overlap={fine_stats['max_overlap']}")
    # 2x2 grid for fine masks as well
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (assign_by, start_from) in zip(axes.flat, configs):
        inst = auto_masks_to_instance_mask(fine_masks, assign_by=assign_by, start_from=start_from)
        viz_inst, cmap, norm = make_viz_mask_and_cmap(inst)
        ax.imshow(viz_inst, cmap=cmap, norm=norm)
        ax.set_title(f"fine: {assign_by}, {start_from}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    print("\n=== Prompt-based Mask Generation Demo ===")

    # Example 1: Single point (ambiguous) - exactly matching main file
    print("\n--- Single Point Example ---")
    point_coords = np.array([[520, 820]])
    point_labels = np.array([1])  # 1 for foreground, 0 for background

    masks, scores, logits = generate_sam2_prompt_masks_decoded(
        test_image_path,
        base_url=base_url,
        api_key=api_key,
        point_coords=point_coords.tolist(),
        point_labels=point_labels.tolist(),
        multimask_output=True  # set to False for single mask output
    )

    print(f"Generated {masks.shape[0]} masks from single point")
    SAM2utils.visualize_prompt_masks(np.array(img), masks, scores,
                                     point_coords, point_labels)

    # Example 2: Multiple points with mask input - exactly matching main file
    print("\n--- Multiple Points with Mask Input ---")
    best_mask_idx = np.argmax(scores)
    mask_input = logits[best_mask_idx:best_mask_idx + 1, :, :]

    point_coords_multi = np.array([[520, 820], [1090, 750]])
    point_labels_multi = np.array([1, 1])

    # Encode mask input with shape information
    mask_input_encoded, mask_input_shape = encode_mask_with_shape(mask_input)

    masks_multi, scores_multi, _ = generate_sam2_prompt_masks_decoded(
        test_image_path,
        base_url=base_url,
        api_key=api_key,
        point_coords=point_coords_multi.tolist(),
        point_labels=point_labels_multi.tolist(),
        mask_input=mask_input_encoded,
        mask_input_shape=mask_input_shape,
        multimask_output=False
    )

    print(f"Refined with multiple points and mask input")
    SAM2utils.visualize_prompt_masks(np.array(img), masks_multi, scores_multi,
                                     point_coords_multi, point_labels_multi)

    # Example 3: Box prompt - exactly matching main file
    print("\n--- Box Prompt Example ---")
    box = np.array([280, 440, 1474, 1225])  # [x1, y1, x2, y2]

    box_masks, box_scores, _ = generate_sam2_prompt_masks_decoded(
        test_image_path,
        base_url=base_url,
        api_key=api_key,
        box=box.tolist(),
        multimask_output=False
    )

    print(f"Generated mask from box prompt")
    SAM2utils.visualize_prompt_masks(np.array(img), box_masks, box_scores, box=box)

    # Example 4: Points + Box - exactly matching main file
    print("\n--- Points + Box Example ---")
    point_coords_4 = np.array([[1398, 752], [520, 820]])
    point_labels_4 = np.array([0, 1])

    combined_masks, combined_scores, _ = generate_sam2_prompt_masks_decoded(
        test_image_path,
        base_url=base_url,
        api_key=api_key,
        point_coords=point_coords_4.tolist(),
        point_labels=point_labels_4.tolist(),  # Negative point
        box=box.tolist(),
        multimask_output=False
    )

    print(f"Generated mask from box + negative point")
    SAM2utils.visualize_prompt_masks(np.array(img), combined_masks, combined_scores,
                                     point_coords=point_coords_4,
                                     point_labels=point_labels_4,
                                     box=box)

    # Example 5: Using SAM2utils directly for analysis - exactly matching main file
    print("\n--- SAM2utils Analysis Demo ---")

    # Get mask statistics
    stats = SAM2utils.get_mask_statistics(coarse_masks)
    print(f"Coarse mask statistics: {stats}")

    # Filter masks by area
    large_masks = SAM2utils.filter_masks_by_area(coarse_masks, min_area=5000)
    print(f"Large masks (>5000 pixels): {len(large_masks)}")

    # Filter by quality scores
    high_quality_masks = SAM2utils.filter_masks_by_score(coarse_masks, min_iou=0.8)
    print(f"High quality masks (IoU>0.8): {len(high_quality_masks)}")

    # Visualize filtered results
    if large_masks:
        print("Visualizing large masks:")
        SAM2utils.visualize_masks(img, large_masks)

    print("✅ SAM2 client demo completed successfully!")
