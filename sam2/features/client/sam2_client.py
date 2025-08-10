import asyncio
import numpy as np
import torch
from typing import Dict, Optional, Union, Tuple, List, Any
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import gc

from sam2.features.sam2_main import SAM2features
from sam2.features.utils import SAM2utils, AsyncModelRunnerParallel, AsyncMultiWrapper


class SAM2FeaturesUnified(AsyncModelRunnerParallel):
    def __init__(self, device: Union[str, torch.device]):
        super().__init__(device=device)

    def _get_model_key(self, model_cfg: str, checkpoint_path: str) -> str:
        return f"{id(self)}_{model_cfg}_{checkpoint_path}"

    def _get_model(self, model_cfg: str, checkpoint_path: str) -> SAM2features:
        key = self._get_model_key(model_cfg, checkpoint_path)
        return self.get_or_create_model(
            key,
            lambda: SAM2features(model_cfg=model_cfg, checkpoint_path=checkpoint_path, device=self.device),
        )

    def _get_lock(self, model_cfg: str, checkpoint_path: str):
        key = self._get_model_key(model_cfg, checkpoint_path)
        return self.get_lock(key)

    def auto_mask(self,
                  image: Optional[Union[str, Path, Image.Image]] = None,
                  image_url: Optional[str] = None,
                  model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                  checkpoint_path: str = "checkpoints/sam2.1_hiera_large.pt",
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
                  output_mode: Optional[str] = "binary_mask") -> List[Dict[str, Any]]:
        """
        Generate automatic masks for the entire image.

        Returns:
            List[Dict]: List of mask dictionaries directly (no base64 encoding)
        """
        pil_img = self.load_image(image, image_url)
        model = self._get_model(model_cfg, checkpoint_path)
        lock = self._get_lock(model_cfg, checkpoint_path)

        # Direct call; SAM2features.auto_mask ignores None values
        with lock:
            result = model.auto_mask(
                pil_img,
                preset=preset,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                mask_threshold=mask_threshold,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=crop_n_layers,
                crop_nms_thresh=crop_nms_thresh,
                crop_overlap_ratio=crop_overlap_ratio,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                point_grids=point_grids,
                min_mask_region_area=min_mask_region_area,
                use_m2m=use_m2m,
                multimask_output=multimask_output,
                output_mode=output_mode,
            )
        return result

    def prompt_mask(self,
                    image: Optional[Union[str, Path, Image.Image]] = None,
                    image_url: Optional[str] = None,
                    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                    checkpoint_path: str = "checkpoints/sam2.1_hiera_large.pt",
                    point_coords: Optional[List[List[float]]] = None,
                    point_labels: Optional[List[int]] = None,
                    box: Optional[List[float]] = None,
                    mask_input: Optional[np.ndarray] = None,
                    multimask_output: bool = True,
                    return_logits: bool = False,
                    normalize_coords: bool = True,
                    mask_threshold: float = 0.0,
                    max_hole_area: float = 0.0,
                    max_sprinkle_area: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks using point/box prompts.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (masks, iou_predictions, low_res_masks) directly
        """
        pil_img = self.load_image(image, image_url)
        model = self._get_model(model_cfg, checkpoint_path)
        lock = self._get_lock(model_cfg, checkpoint_path)

        # Convert inputs to numpy arrays if provided
        point_coords_np = np.array(point_coords) if point_coords else None
        point_labels_np = np.array(point_labels) if point_labels else None
        box_np = np.array(box) if box else None

        with lock:
            result = model.prompt_mask(
                pil_img=pil_img,
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                box=box_np,
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=return_logits,
                normalize_coords=normalize_coords,
                mask_threshold=mask_threshold,
                max_hole_area=max_hole_area,
                max_sprinkle_area=max_sprinkle_area
            )
        return result

    async def auto_mask_async(self, **kwargs) -> List[Dict[str, Any]]:
        return await self.run_in_executor(self.auto_mask, **kwargs)

    async def prompt_mask_async(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return await self.run_in_executor(self.prompt_mask, **kwargs)


if __name__ == "__main__":
    sam2 = AsyncMultiWrapper(SAM2FeaturesUnified, num_objects=2)
    test_image_path = "notebooks/images/cars.jpg"
    img = Image.open(test_image_path)
    print("=== Direct SAM2 API Demo ===")
    print(f"Input image size: h={img.height}, w={img.width}")

    print("=== Automatic Mask Generation Demo ===")

    # Generate coarse masks
    print("Generating coarse masks...")
    coarse_masks = sam2.auto_mask(
        image=test_image_path,
        preset="coarse"
    )
    print(f"Generated {len(coarse_masks)} coarse masks")
    SAM2utils.visualize_masks(img, coarse_masks)

    # Overlap analysis and instance mask demo for coarse
    coarse_stats = SAM2utils.compute_overlap_stats(coarse_masks)
    if coarse_stats["total_masks"] > 0:
        coarse_pct = 100.0 * coarse_stats["high_overlap_pairs"] / (coarse_stats["total_masks"] * (coarse_stats["total_masks"] - 1) / 2) if coarse_stats["total_masks"] > 1 else 0
        print(f"Coarse Overlap: masks={coarse_stats['total_masks']}, mean_overlap={coarse_stats['mean_overlap']:.3f}, high_overlap_pairs={coarse_stats['high_overlap_pairs']} ({coarse_pct:.1f}%)")

    # 2x2 grid of instance masks for (assign_by x start_from)
    configs = [("iou", "high"), ("iou", "low"), ("area", "high"), ("area", "low")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (assign_by, start_from) in zip(axes.flat, configs):
        inst_mask, stats = SAM2utils.auto_masks_to_instance_mask(
            coarse_masks, min_iou=0.7, assign_by=assign_by, start_from=start_from
        )
        viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(inst_mask)
        ax.imshow(viz_mask, cmap=cmap, norm=norm)
        ax.set_title(f"{assign_by}, {start_from}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Generate fine-grained masks (resize for memory efficiency)
    small_img = SAM2utils.prevent_oom_resizing(img, target=512)
    print("Generating fine-grained masks...")
    fine_masks = sam2.auto_mask(
        image=small_img,
        preset="fine_grained"
    )
    print(f"Generated {len(fine_masks)} fine-grained masks")
    SAM2utils.visualize_masks(small_img, fine_masks)
    del fine_masks
    torch.cuda.empty_cache()

    print("\n=== Prompt-based Mask Generation Demo ===")

    # Example 1: Single point (ambiguous)
    print("\n--- Single Point Example ---")
    point_coords = [[520, 820]]
    point_labels = [1]  # 1 for foreground, 0 for background

    masks, scores, logits = sam2.prompt_mask(
        image=test_image_path,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True  # set to False for single mask output
    )

    print(f"Generated {masks.shape[0]} masks from single point")
    SAM2utils.visualize_prompt_masks(np.array(img), masks, scores,
                                     np.array(point_coords), np.array(point_labels))

    # Example 2: Multiple points with mask input
    print("\n--- Multiple Points with Mask Input ---")
    best_mask_idx = np.argmax(scores)
    mask_input = logits[best_mask_idx:best_mask_idx + 1, :, :]

    point_coords_multi = [[520, 820], [1090, 750]]
    point_labels_multi = [1, 1]

    masks_multi, scores_multi, _ = sam2.prompt_mask(
        image=test_image_path,
        point_coords=point_coords_multi,
        point_labels=point_labels_multi,
        mask_input=mask_input,
        multimask_output=False
    )

    print(f"Refined with multiple points and mask input")
    SAM2utils.visualize_prompt_masks(np.array(img), masks_multi, scores_multi,
                                     np.array(point_coords_multi), np.array(point_labels_multi))

    # Example 3: Box prompt
    print("\n--- Box Prompt Example ---")
    box = [280, 440, 1474, 1225]  # [x1, y1, x2, y2]

    box_masks, box_scores, _ = sam2.prompt_mask(
        image=test_image_path,
        box=box,
        multimask_output=False
    )

    print(f"Generated mask from box prompt")
    SAM2utils.visualize_prompt_masks(np.array(img), box_masks, box_scores, box=np.array(box))

    # Example 4: Points + Box
    print("\n--- Points + Box Example ---")
    point_coords_4 = [[1398, 752], [520, 820]]
    point_labels_4 = [0, 1]  # Negative point, positive point

    combined_masks, combined_scores, _ = sam2.prompt_mask(
        image=test_image_path,
        point_coords=point_coords_4,
        point_labels=point_labels_4,
        box=box,
        multimask_output=False
    )

    print(f"Generated mask from box + negative point")
    SAM2utils.visualize_prompt_masks(np.array(img), combined_masks, combined_scores,
                                     point_coords=np.array(point_coords_4),
                                     point_labels=np.array(point_labels_4),
                                     box=np.array(box))

    # Example 5: Using SAM2utils directly for analysis
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

    print("=== Async Demo ===")
    # cleanup figures created by SAM2utils visualizations before async work
    plt.close('all')
    gc.collect()
    # warmup num_objects times so it doesnt load twice in async mode
    _ = sam2.auto_mask(image=test_image_path, preset="coarse")
    _ = sam2.auto_mask(image=test_image_path, preset="coarse")

    tasks = [sam2.auto_mask_async(image=test_image_path, preset="coarse") for _ in range(4)]
    tasks += [sam2.prompt_mask_async(image=test_image_path, point_coords=[[520, 820]], point_labels=[1]) for _ in range(4)]
    results = asyncio.run(AsyncMultiWrapper.async_run_tasks(tasks, desc="SAM2 tasks"))

    auto_result = results[0]
    prompt_result = results[-1]
    SAM2utils.visualize_masks(img, auto_result)
    SAM2utils.visualize_prompt_masks(np.array(img), prompt_result[0], prompt_result[1],
                                     point_coords=np.array([[520, 820]]),
                                     point_labels=np.array([1]))
    print("✅ Direct SAM2 API demo completed successfully!")
