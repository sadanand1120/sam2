import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Union, Tuple

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.features.utils import SAM2utils


class SAM2features:
    def __init__(self, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint_path="checkpoints/sam2.1_hiera_large.pt", device=None):
        """
        SAM2 feature extractor for automatic and prompt-based mask generation.

        Args:
            model_cfg: Path to model configuration file
            checkpoint_path: Path to model checkpoint (required)
            device: Device to run model on (auto-detected if None)
        """
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cpu")

        # Build SAM2 model (hydra fix: add /  or // to paths)
        try:
            self.model = build_sam2("/" + model_cfg, "/" + checkpoint_path, device=self.device, apply_postprocessing=False)
        except:
            self.model = build_sam2("//" + model_cfg, "//" + checkpoint_path, device=self.device, apply_postprocessing=False)

        # Predefined intelligent presets for automatic mask generation
        self.AUTO_FINE_GRAINED_MASKS = {
            "points_per_side": 64,
            "points_per_batch": 128,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.92,
            "stability_score_offset": 0.7,
            "mask_threshold": 0.0,
            "box_nms_thresh": 0.7,
            "crop_n_layers": 1,
            "crop_nms_thresh": 0.7,
            "crop_overlap_ratio": 512 / 1500,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 25.0,
            "use_m2m": True,
            "multimask_output": True,
            "output_mode": "binary_mask"
        }

        self.AUTO_COARSE_MASKS = {
            "points_per_side": 16,
            "points_per_batch": 64,
            "pred_iou_thresh": 0.8,
            "stability_score_thresh": 0.95,
            "stability_score_offset": 1.0,
            "mask_threshold": 0.0,
            "box_nms_thresh": 0.7,
            "crop_n_layers": 0,
            "crop_nms_thresh": 0.7,
            "crop_overlap_ratio": 512 / 1500,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 100.0,
            "use_m2m": False,
            "multimask_output": True,
            "output_mode": "binary_mask"
        }

    def auto_mask(self,
                  pil_img: Image.Image,
                  preset: Optional[str] = None,
                  **kwargs) -> List[Dict[str, Any]]:
        """
        Generate automatic masks for the entire image.

        Args:
            pil_img: PIL Image to process
            preset: Use predefined settings ("fine_grained" or "coarse")
            **kwargs: Additional parameters for SAM2AutomaticMaskGenerator:
                points_per_side: Number of points sampled along one side of image
                points_per_batch: Number of points run simultaneously by model
                pred_iou_thresh: Filtering threshold using model's predicted mask quality
                stability_score_thresh: Filtering threshold using mask stability
                stability_score_offset: Amount to shift cutoff for stability score
                mask_threshold: Threshold for binarizing mask logits
                box_nms_thresh: Box IoU cutoff for non-maximal suppression
                crop_n_layers: Number of crop layers (0 = no cropping)
                crop_nms_thresh: Box IoU cutoff for NMS between different crops
                crop_overlap_ratio: Degree to which crops overlap
                crop_n_points_downscale_factor: Points-per-side scaling in crop layers
                point_grids: Explicit grids of points for sampling
                min_mask_region_area: Remove disconnected regions smaller than this
                output_mode: Format of returned masks ("binary_mask", "uncompressed_rle", "coco_rle")
                use_m2m: Whether to add one step refinement using previous mask predictions
                multimask_output: Whether to output multimask at each point

        Returns:
            List of mask dictionaries with keys: segmentation, area, bbox, predicted_iou, 
            point_coords, stability_score, crop_box
        """
        # Apply preset if specified
        if preset == "fine_grained":
            params = self.AUTO_FINE_GRAINED_MASKS.copy()
        elif preset == "coarse":
            params = self.AUTO_COARSE_MASKS.copy()
        else:
            params = {}

        # Override preset with explicitly provided parameters via kwargs
        params.update(kwargs)

        # Create mask generator with specified parameters
        mask_generator = SAM2AutomaticMaskGenerator(model=self.model, **params)

        # Convert PIL to numpy array
        image_array = np.array(pil_img.convert("RGB"))

        # Generate masks
        masks = mask_generator.generate(image_array)

        return masks

    def prompt_mask(self,
                    pil_img: Image.Image,
                    point_coords: Optional[np.ndarray] = None,
                    point_labels: Optional[np.ndarray] = None,
                    box: Optional[np.ndarray] = None,
                    mask_input: Optional[np.ndarray] = None,
                    multimask_output: bool = True,
                    return_logits: bool = False,
                    normalize_coords: bool = True,
                    mask_threshold: float = 0.0,
                    max_hole_area: float = 0.0,
                    max_sprinkle_area: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks using point/box prompts.

        Args:
            pil_img: PIL Image to process
            point_coords: Nx2 array of point prompts in (X,Y) pixels
            point_labels: Length N array of labels (1=foreground, 0=background)
            box: Length 4 array of box prompt in XYXY format
            mask_input: Low resolution mask input from previous iteration
            multimask_output: If True, return three masks for ambiguous prompts
            return_logits: If True, return un-thresholded mask logits
            normalize_coords: If True, normalize coordinates to [0,1] range
            mask_threshold: Threshold for binarizing mask logits
            max_hole_area: Fill small holes up to this area in masks
            max_sprinkle_area: Remove small sprinkles up to this area in masks

        Returns:
            Tuple of (masks, iou_predictions, low_res_masks)
            - masks: Output masks in CxHxW format
            - iou_predictions: Model's prediction of mask quality
            - low_res_masks: Low resolution logits for subsequent iterations
        """
        # Create predictor with specified parameters
        predictor = SAM2ImagePredictor(
            self.model,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area
        )

        # Set image
        predictor.set_image(pil_img)

        # Predict masks
        masks, iou_predictions, low_res_masks = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords
        )

        return masks, iou_predictions, low_res_masks

    def visualize_masks(self, image: Union[Image.Image, np.ndarray], masks: List[Dict[str, Any]],
                        borders: bool = True, alpha: float = 0.5):
        """Visualize automatic masks (delegates to SAM2utils)."""
        return SAM2utils.visualize_masks(image, masks, borders=borders, alpha=alpha)

    def visualize_prompt_masks(self, image: Union[Image.Image, np.ndarray], masks: np.ndarray, scores: np.ndarray,
                               point_coords: Optional[np.ndarray] = None,
                               point_labels: Optional[np.ndarray] = None,
                               box: Optional[np.ndarray] = None,
                               borders: bool = True,
                               sort_by_score: bool = True):
        """Visualize prompt-based masks (delegates to SAM2utils)."""
        return SAM2utils.visualize_prompt_masks(image, masks, scores,
                                                point_coords=point_coords,
                                                point_labels=point_labels,
                                                box=box,
                                                borders=borders,
                                                sort_by_score=sort_by_score)


if __name__ == "__main__":
    # Demo usage
    img = Image.open("notebooks/images/cars.jpg")
    sam2 = SAM2features(model_cfg="/robodata/smodak/repos/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                        checkpoint_path="/robodata/smodak/repos/sam2/checkpoints/sam2.1_hiera_large.pt",
                        device=torch.device("cuda"))

    print("=== Automatic Mask Generation Demo ===")

    # Generate coarse masks
    print("Generating coarse masks...")
    coarse_masks = sam2.auto_mask(img, preset="coarse")
    print(f"Generated {len(coarse_masks)} coarse masks")
    sam2.visualize_masks(img, coarse_masks)

    # Generate fine-grained masks
    print("Generating fine-grained masks...")
    fine_masks = sam2.auto_mask(img, preset="fine_grained")
    print(f"Generated {len(fine_masks)} fine-grained masks")
    sam2.visualize_masks(img, fine_masks)

    print("\n=== Prompt-based Mask Generation Demo ===")

    # Example 1: Single point (ambiguous)
    print("\n--- Single Point Example ---")
    point_coords = np.array([[520, 820]])
    point_labels = np.array([1])  # 1 for foreground, 0 for background

    masks, scores, logits = sam2.prompt_mask(
        img,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True  # set to False for single mask output
    )

    print(f"Generated {masks.shape[0]} masks from single point")
    sam2.visualize_prompt_masks(np.array(img), masks, scores,
                                point_coords, point_labels)

    # Example 2: Multiple points with mask input
    print("\n--- Multiple Points with Mask Input ---")
    best_mask_idx = np.argmax(scores)
    mask_input = logits[best_mask_idx:best_mask_idx + 1, :, :]

    point_coords_multi = np.array([[520, 820], [1090, 750]])
    point_labels_multi = np.array([1, 1])

    masks_multi, scores_multi, _ = sam2.prompt_mask(
        img,
        point_coords=point_coords_multi,
        point_labels=point_labels_multi,
        mask_input=mask_input,
        multimask_output=False
    )

    print(f"Refined with multiple points and mask input")
    sam2.visualize_prompt_masks(np.array(img), masks_multi, scores_multi,
                                point_coords_multi, point_labels_multi)

    # Example 3: Box prompt
    print("\n--- Box Prompt Example ---")
    box = np.array([280, 440, 1474, 1225])  # [x1, y1, x2, y2]

    box_masks, box_scores, _ = sam2.prompt_mask(
        img,
        box=box,
        multimask_output=False
    )

    print(f"Generated mask from box prompt")
    sam2.visualize_prompt_masks(np.array(img), box_masks, box_scores, box=box)

    # Example 4: Points + Box
    print("\n--- Points + Box Example ---")
    point_coords_4 = np.array([[1398, 752], [520, 820]])
    point_labels_4 = np.array([0, 1])

    combined_masks, combined_scores, _ = sam2.prompt_mask(
        img,
        point_coords=point_coords_4,
        point_labels=point_labels_4,  # Negative point
        box=box,
        multimask_output=False
    )

    print(f"Generated mask from box + negative point")
    sam2.visualize_prompt_masks(np.array(img), combined_masks, combined_scores,
                                point_coords=point_coords_4,
                                point_labels=point_labels_4,
                                box=box)

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
