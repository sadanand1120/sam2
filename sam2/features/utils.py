from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop, Compose, Resize, InterpolationMode, ToTensor, Normalize
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import matplotlib.pyplot as plt


def interpolate_positional_embedding(positional_embedding: torch.Tensor, x: torch.Tensor, patch_size: int, w: int, h: int):
    """
    Interpolate the positional encoding for CLIP to the number of patches in the image given width and height.
    Modified from DINO ViT `interpolate_pos_encoding` method.
    https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174
    """
    assert positional_embedding.ndim == 2, "pos_encoding must be 2D"

    # Number of patches in input
    num_patches = x.shape[1] - 1
    # Original number of patches for square images
    num_og_patches = positional_embedding.shape[0] - 1

    if num_patches == num_og_patches and w == h:
        # No interpolation needed
        return positional_embedding.to(x.dtype)

    dim = x.shape[-1]
    class_pos_embed = positional_embedding[:1]  # (1, dim)
    patch_pos_embed = positional_embedding[1:]  # (num_og_patches, dim)

    # Compute number of tokens
    w0 = w // patch_size
    h0 = h // patch_size
    assert w0 * h0 == num_patches, "Number of patches does not match"

    # Add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1

    # Interpolate
    patch_per_ax = int(np.sqrt(num_og_patches))
    patch_pos_embed_interp = torch.nn.functional.interpolate(
        patch_pos_embed.reshape(1, patch_per_ax, patch_per_ax, dim).permute(0, 3, 1, 2),
        # (1, dim, patch_per_ax, patch_per_ax)
        scale_factor=(w0 / patch_per_ax, h0 / patch_per_ax),
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=False,
    )  # (1, dim, w0, h0)
    assert (
        int(w0) == patch_pos_embed_interp.shape[-2] and int(h0) == patch_pos_embed_interp.shape[-1]
    ), "Interpolation error."

    patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).reshape(-1, dim)  # (w0 * h0, dim)
    # Concat class token embedding and interpolated patch embeddings
    pos_embed_interp = torch.cat([class_pos_embed, patch_pos_embed_interp], dim=0)  # (w0 * h0 + 1, dim)
    return pos_embed_interp.to(x.dtype)


@torch.inference_mode()
def preprocess_image(
    image: Union[str, Path, Image.Image],
    load_size: Optional[int] = None,
    center_crop: bool = False,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[torch.Tensor, int, int]:
    """
    Unified preprocessing for both CLIP and DINO models.

    Args:
        image: PIL Image, file path, or Path object
        load_size: Target size for resizing. If None, no resizing. If negative, scale smallest side.
        center_crop: Whether to center crop after resizing
        mean: Normalization mean values
        std: Normalization std values

    Returns:
        Tuple of (processed_tensor, original_width, original_height)
        processed_tensor: (1, C, H, W) tensor ready for model
    """
    # Load and convert to RGB
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image.convert('RGB')
    else:
        raise ValueError("image must be str, Path, or PIL Image")

    # Store original size
    orig_w, orig_h = pil_image.size

    # Build transform pipeline
    transforms_list = []

    # Resize if specified
    if load_size is not None:
        if load_size < 0:
            # Scale smallest side by the absolute value
            load_size = abs(load_size) * min(orig_w, orig_h)
        transforms_list.append(Resize(load_size, interpolation=InterpolationMode.BICUBIC))

        # Center crop if requested
        if center_crop:
            transforms_list.append(CenterCrop(load_size))

    # Convert to tensor and normalize
    transforms_list.extend([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    # Apply transforms
    transform = Compose(transforms_list)
    img_tensor = transform(pil_image)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, orig_w, orig_h


def apply_pca_colormap(
    image: torch.Tensor,
    proj_V: Optional[torch.Tensor] = None,
    low_rank_min: Optional[torch.Tensor] = None,
    low_rank_max: Optional[torch.Tensor] = None,
    niter: int = 5,
    return_proj: bool = False,
    q_min: float = 0.01,
    q_max: float = 0.99,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert a multichannel image to color using PCA. Fully torch, CUDA compatible.
    If return_proj is True, also return (proj_V, low_rank_min, low_rank_max) for reuse.
    q_min, q_max: quantile range for normalization (default 0.01, 0.99)
    """
    orig_shape = image.shape
    image_flat = image.reshape(-1, image.shape[-1])
    if proj_V is None:
        mean = image_flat.mean(0)
        U, S, V = torch.pca_lowrank(image_flat - mean, niter=niter)
        proj_V = V[:, :3]
    low_rank = image_flat @ proj_V
    if low_rank_min is None:
        low_rank_min = torch.quantile(low_rank, q_min, dim=0)
    if low_rank_max is None:
        low_rank_max = torch.quantile(low_rank, q_max, dim=0)
    low_rank = (low_rank - low_rank_min) / (low_rank_max - low_rank_min)
    low_rank = torch.clamp(low_rank, 0, 1)
    colored_image = low_rank.reshape(*orig_shape[:-1], 3)
    if return_proj:
        return colored_image, proj_V, low_rank_min, low_rank_max
    else:
        return colored_image


def pad_to_multiple(img_tensor: torch.Tensor, patch_size: int, format: str = "CHW", mode: str = "constant") -> Tuple[torch.Tensor, int, int]:
    """
    Pads only the bottom and right edges so H and W become exact multiples of patch_size.
    This ensures perfect inverse relationship with unpadding operations.

    Args:
        img_tensor: Input tensor to pad
        patch_size: Size that H and W dimensions should be multiples of
        format: Tensor format - "BCHW", "CHW", "HWC", or "HW"

    Returns: padded_tensor, pad_h, pad_w
    """
    if format == "BCHW":  # (B, C, H, W)
        H, W = img_tensor.shape[2], img_tensor.shape[3]
    elif format == "CHW":  # (C, H, W)
        H, W = img_tensor.shape[1], img_tensor.shape[2]
    elif format == "HWC":  # (H, W, C)
        H, W = img_tensor.shape[0], img_tensor.shape[1]
    elif format == "HW":  # (H, W)
        H, W = img_tensor.shape[0], img_tensor.shape[1]
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'BCHW', 'CHW', 'HWC', or 'HW'")

    pad_h = (-H) % patch_size     # same as: (patch_size - H % patch_size) % patch_size
    pad_w = (-W) % patch_size

    if pad_h or pad_w:
        if format in ["BCHW", "CHW"]:
            # For formats where H, W are last two dims: pad = (left, right, top, bottom)
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode=mode, value=0)
        elif format == "HWC":
            # For (H, W, C): pad = (left_C, right_C, left_W, right_W, left_H, right_H)
            img_tensor = torch.nn.functional.pad(img_tensor, (0, 0, 0, pad_w, 0, pad_h), mode=mode, value=0)
        elif format == "HW":
            # For (H, W): pad = (left_W, right_W, left_H, right_H)
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode=mode, value=0)

    return img_tensor, pad_h, pad_w


def remove_padding(tensor: torch.Tensor, pad_h: int, pad_w: int, format: str = "HWC") -> torch.Tensor:
    """
    Removes the bottom-row and right-column padding that was added by pad_to_multiple.
    This is the exact mathematical inverse of pad_to_multiple.

    Args:
        tensor: Input tensor to unpad
        pad_h: Height padding to remove
        pad_w: Width padding to remove  
        format: Tensor format - "BCHW", "CHW", "HWC", or "HW"
    """
    if format == "BCHW":  # (B, C, H, W)
        if pad_h:
            tensor = tensor[:, :, :-pad_h, :]
        if pad_w:
            tensor = tensor[:, :, :, :-pad_w]
    elif format == "CHW":  # (C, H, W)
        if pad_h:
            tensor = tensor[:, :-pad_h, :]
        if pad_w:
            tensor = tensor[:, :, :-pad_w]
    elif format == "HWC":  # (H, W, C)
        if pad_h:
            tensor = tensor[:-pad_h, :, :]
        if pad_w:
            tensor = tensor[:, :-pad_w, :]
    elif format == "HW":  # (H, W)
        if pad_h:
            tensor = tensor[:-pad_h, :]
        if pad_w:
            tensor = tensor[:, :-pad_w]
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'BCHW', 'CHW', 'HWC', or 'HW'")

    return tensor


def upsample_and_unpad(patch_desc: torch.Tensor, target_size: Tuple[int, int], pad_h: int, pad_w: int, tensor_format: str = "HWC", mode: str = "bilinear") -> torch.Tensor:
    """
    Unified function to upsample patch descriptors and remove padding.

    Args:
        patch_desc: (h, w, d) or (h, w, 3) torch tensor
        target_size: (H, W) target size including padding
        pad_h, pad_w: padding amounts to remove
        tensor_format: Format for the output tensor ("HWC", "CHW", etc.)

    Returns:
        torch tensor of shape (H-pad_h, W-pad_w, d) in specified format
    """
    # First upsample to target size (including padding)
    patch_desc_t = patch_desc.permute(2, 0, 1).unsqueeze(0).float()  # (1, c, h, w)
    upsampled = torch.nn.functional.interpolate(patch_desc_t, size=target_size, mode=mode, align_corners=False)
    upsampled = upsampled.squeeze(0).permute(1, 2, 0)  # (H, W, c)

    # Then remove padding using the mathematically inverse operation
    return remove_padding(upsampled, pad_h, pad_w, format=tensor_format)


class SAM2utils:
    """
    Utility class for SAM2 output visualization and processing.
    This class provides visualization methods that can work with SAM2 outputs
    independently of the main SAM2features class.
    """

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
        """Show a single mask with proper coloring and borders (from notebook)."""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            try:
                import cv2
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=8)
            except ImportError:
                pass  # Skip borders if cv2 not available
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        """Show points with proper colors (from notebook)."""
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        """Show bounding box (from notebook)."""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                                   facecolor=(0, 0, 0, 0), lw=2))

    @staticmethod
    def visualize_masks(image: Union[Image.Image, np.ndarray], masks: List[Dict[str, Any]],
                        borders: bool = True, alpha: float = 0.5):
        """
        Visualize automatic masks on an image.

        Args:
            image: PIL Image or numpy array
            masks: List of mask dictionaries from auto_mask()
            borders: Whether to draw mask borders
            alpha: Transparency of mask overlay
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        if len(masks) == 0:
            plt.imshow(image)
            plt.title("No masks generated")
            plt.axis('off')
            plt.show()
            return

        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

        plt.figure(figsize=(12, 8))
        plt.imshow(image)

        # Create overlay
        overlay = np.ones((image.shape[0], image.shape[1], 4))
        overlay[:, :, 3] = 0

        for mask_dict in sorted_masks:
            mask = mask_dict['segmentation']
            color_mask = np.concatenate([np.random.random(3), [alpha]])
            overlay[mask] = color_mask

            if borders:
                try:
                    import cv2
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                    cv2.drawContours(overlay, contours, -1, (0, 0, 1, 0.4), thickness=8)
                except ImportError:
                    pass  # Skip borders if cv2 not available

        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f"Automatic Masks ({len(masks)} masks)")
        plt.show()

    @staticmethod
    def visualize_prompt_masks(image: Union[Image.Image, np.ndarray], masks: np.ndarray, scores: np.ndarray,
                               point_coords: Optional[np.ndarray] = None,
                               point_labels: Optional[np.ndarray] = None,
                               box: Optional[np.ndarray] = None,
                               borders: bool = True,
                               sort_by_score: bool = True):
        """
        Visualize prompt-based masks on an image (following notebook style).

        Args:
            image: PIL Image or numpy array  
            masks: Masks from prompt_mask()
            scores: IoU scores from prompt_mask()
            point_coords: Point coordinates used for prompting
            point_labels: Point labels used for prompting
            box: Box coordinates used for prompting (XYXY format)
            borders: Whether to show mask borders
            sort_by_score: Whether to sort masks by score (highest first)
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        # Sort masks by score if requested
        if sort_by_score and len(scores) > 1:
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]

        # Show each mask in a separate figure (following notebook style)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            SAM2utils.show_mask(mask, plt.gca(), borders=borders)

            # Show prompts
            if point_coords is not None and point_labels is not None:
                SAM2utils.show_points(point_coords, point_labels, plt.gca())
            if box is not None:
                SAM2utils.show_box(box, plt.gca())

            # Show title with score if multiple masks
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            else:
                plt.title(f"Mask, Score: {score:.3f}", fontsize=18)

            plt.axis('off')
            plt.show()

    @staticmethod
    def save_masks_as_images(masks: List[Dict[str, Any]], output_dir: str, prefix: str = "mask"):
        """
        Save automatic masks as individual image files.

        Args:
            masks: List of mask dictionaries from auto_mask()
            output_dir: Directory to save mask images
            prefix: Filename prefix for saved masks
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation'].astype(np.uint8) * 255
            mask_img = Image.fromarray(mask, mode='L')
            mask_img.save(os.path.join(output_dir, f"{prefix}_{i:03d}.png"))

        print(f"Saved {len(masks)} masks to {output_dir}")

    @staticmethod
    def save_prompt_masks_as_images(masks: np.ndarray, scores: np.ndarray,
                                    output_dir: str, prefix: str = "prompt_mask"):
        """
        Save prompt-based masks as individual image files.

        Args:
            masks: Masks from prompt_mask()
            scores: IoU scores from prompt_mask()
            output_dir: Directory to save mask images
            prefix: Filename prefix for saved masks
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_img = (mask.astype(np.uint8) * 255)
            mask_pil = Image.fromarray(mask_img, mode='L')
            filename = f"{prefix}_{i:03d}_score_{score:.3f}.png"
            mask_pil.save(os.path.join(output_dir, filename))

        print(f"Saved {len(masks)} prompt masks to {output_dir}")

    @staticmethod
    def get_mask_statistics(masks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics for automatic masks.

        Args:
            masks: List of mask dictionaries from auto_mask()

        Returns:
            Dictionary with mask statistics
        """
        if not masks:
            return {"num_masks": 0}

        areas = [mask_dict['area'] for mask_dict in masks]
        ious = [mask_dict.get('predicted_iou', 0) for mask_dict in masks]
        stability_scores = [mask_dict.get('stability_score', 0) for mask_dict in masks]

        stats = {
            "num_masks": int(len(masks)),
            "mean_iou": round(float(np.mean(ious)), 3) if ious else 0.0,
            "mean_stability": round(float(np.mean(stability_scores)), 3) if stability_scores else 0.0,
            "mean_area": round(float(np.mean(areas)), 3),
            "median_area": round(float(np.median(areas)), 3),
            "min_area": int(min(areas)),
            "max_area": int(max(areas)),
            "total_area": int(sum(areas)),
        }
        return stats

    @staticmethod
    def filter_masks_by_area(masks: List[Dict[str, Any]],
                             min_area: Optional[float] = None,
                             max_area: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Filter masks by area constraints.

        Args:
            masks: List of mask dictionaries from auto_mask()
            min_area: Minimum area threshold (inclusive)
            max_area: Maximum area threshold (inclusive)

        Returns:
            Filtered list of masks
        """
        filtered = masks

        if min_area is not None:
            filtered = [m for m in filtered if m['area'] >= min_area]

        if max_area is not None:
            filtered = [m for m in filtered if m['area'] <= max_area]

        return filtered

    @staticmethod
    def filter_masks_by_score(masks: List[Dict[str, Any]],
                              min_iou: Optional[float] = None,
                              min_stability: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Filter masks by quality scores.

        Args:
            masks: List of mask dictionaries from auto_mask()
            min_iou: Minimum predicted IoU threshold
            min_stability: Minimum stability score threshold

        Returns:
            Filtered list of masks
        """
        filtered = masks

        if min_iou is not None:
            filtered = [m for m in filtered if m.get('predicted_iou', 0) >= min_iou]

        if min_stability is not None:
            filtered = [m for m in filtered if m.get('stability_score', 0) >= min_stability]

        return filtered


if __name__ == "__main__":
    # Test with different tensor shapes and formats
    test_cases = [
        (torch.randn(3, 21, 34), "CHW", "(C, H, W)"),        # (C, H, W) - channels first
        (torch.randn(1, 3, 21, 34), "BCHW", "(B, C, H, W)"),  # (B, C, H, W) - batch
        (torch.randn(21, 34), "HW", "(H, W)"),              # (H, W) - 2D
        (torch.randn(21, 34, 3), "HWC", "(H, W, C)"),        # (H, W, C) - channels last
    ]

    for i, (img, fmt, desc) in enumerate(test_cases):
        print(f'Test case {i+1}: {desc} - Original shape {img.shape}')
        padded, ph, pw = pad_to_multiple(img, patch_size=16, format=fmt, mode="constant")
        print(f'  Padded shape: {padded.shape}, pad_h={ph}, pad_w={pw}')
        recovered = remove_padding(padded, ph, pw, format=fmt)
        print(f'  Recovered shape: {recovered.shape}')
        assert torch.equal(img, recovered), f'Test case {i+1} failed!'
        print(f'  ✓ Perfect inverse confirmed')
        print()

    print('All tests passed! Padding and unpadding are perfect inverses.')
