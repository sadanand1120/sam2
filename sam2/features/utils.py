from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop, Compose, Resize, InterpolationMode, ToTensor, Normalize
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any, Callable, Type
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import cycle
import threading
import asyncio
from tqdm.auto import tqdm
import io
import requests


class DeviceRoundRobin:
    def __init__(self):
        if torch.cuda.is_available():
            self._iter = cycle(range(torch.cuda.device_count()))
        else:
            self._iter = None

    def next(self):
        if self._iter is None:
            return 'cpu'
        return f'cuda:{next(self._iter)}'


class AsyncModelRunnerParallel:
    def __init__(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        self._models: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}

    def get_or_create_model(self, key: str, factory: Callable[[], Any]) -> Any:
        if key not in self._models:
            self._models[key] = factory()
            self._locks[key] = threading.Lock()
        return self._models[key]

    def get_lock(self, key: str) -> threading.Lock:
        if key not in self._locks:
            self._locks[key] = threading.Lock()
        return self._locks[key]

    def load_image(self,
                   image: Optional[Union[str, Path, Image.Image]] = None,
                   image_url: Optional[str] = None) -> Image.Image:
        if image is not None:
            if isinstance(image, (str, Path)):
                return Image.open(image).convert('RGB')
            if isinstance(image, Image.Image):
                return image.convert('RGB')
            raise ValueError("image must be str, Path, or PIL Image")
        if image_url is not None:
            resp = requests.get(image_url)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert('RGB')
        raise ValueError("Either image or image_url must be provided")

    async def run_in_executor(self, func: Callable, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    @classmethod
    async def async_run_tasks(
        cls,
        awaitables: List[asyncio.Future],
        desc: Optional[str] = None,
        position: Optional[int] = None,
        disable: Optional[bool] = None,
        leave: Optional[bool] = None,
    ) -> List[Any]:
        if desc is None:
            desc = cls.__name__

        async def _wrap_with_index(i, aw):
            res = await aw
            return i, res

        tasks = [asyncio.create_task(_wrap_with_index(i, aw)) for i, aw in enumerate(awaitables)]
        results: List[Any] = [None] * len(tasks)
        bar_kwargs: Dict[str, Any] = {"desc": desc, "total": len(tasks)}
        if position is not None:
            bar_kwargs["position"] = position
        if disable is not None:
            bar_kwargs["disable"] = disable
        if leave is not None:
            bar_kwargs["leave"] = leave
        with tqdm(**bar_kwargs) as pbar:
            for t in asyncio.as_completed(tasks):
                i, res = await t
                results[i] = res
                pbar.update(1)
        return results


class AsyncMultiWrapper:
    """
    Minimal wrapper to dispatch method calls across multiple worker instances
    in a round-robin fashion. If num_objects > available GPUs, devices repeat
    from the beginning (circular assignment).

    Example:
        wrapper = AsyncMultiWrapper(DINOFeaturesUnified, num_objects=2)
        task = wrapper.extract_features_async(...)
    """

    def __init__(
        self,
        worker_cls: Type[Any],
        num_objects: int = 1,
        devices: Optional[Union[str, torch.device, List[Union[str, torch.device]]]] = None,
        **worker_kwargs: Any,
    ) -> None:
        if num_objects < 1:
            raise ValueError("num_objects must be >= 1")

        if devices is None:
            dev_iter = DeviceRoundRobin()
            dev_list = [dev_iter.next() for _ in range(num_objects)]
        elif isinstance(devices, (str, torch.device)):
            dev_list = [devices] * num_objects
        else:
            dev_list = list(devices)
            if len(dev_list) < num_objects:
                cyc = cycle(dev_list)
                dev_list = [next(cyc) for _ in range(num_objects)]
            else:
                dev_list = dev_list[:num_objects]

        self._workers: List[Any] = [worker_cls(device=d, **worker_kwargs) for d in dev_list]
        self._rr_lock = threading.Lock()
        self._rr_index = 0

    def _next_worker(self) -> Any:
        with self._rr_lock:
            idx = self._rr_index
            self._rr_index = (self._rr_index + 1) % len(self._workers)
        return self._workers[idx]

    def __getattr__(self, name: str):  # Dispatch any unknown attribute to the next worker
        def _call_through(*args, **kwargs):
            worker = self._next_worker()
            attr = getattr(worker, name)
            return attr(*args, **kwargs)

        return _call_through

    @staticmethod
    async def async_run_tasks(
        awaitables: List[asyncio.Future],
        desc: Optional[str] = None,
        position: Optional[int] = None,
        disable: Optional[bool] = None,
        leave: Optional[bool] = None,
    ) -> List[Any]:
        return await AsyncModelRunnerParallel.async_run_tasks(
            awaitables,
            desc=desc,
            position=position,
            disable=disable,
            leave=leave,
        )


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


@torch.inference_mode()
def repackage_pixels_to_patch_grid(pixel_desc: torch.Tensor, pad_h: int, pad_w: int, patch_size: int) -> torch.Tensor:
    """
    Convert per-pixel descriptors (H, W, C) back to a patch grid (h, w, C).

    Steps:
    - Re-pad the pixel map at bottom/right by (pad_h, pad_w) to restore the padded shape
    - Average-pool with kernel=stride=patch_size to obtain the patch grid

    Args:
        pixel_desc: Tensor of shape (H, W, C) without padding
        pad_h: Bottom padding to restore
        pad_w: Right padding to restore
        patch_size: Patch size used by the model (also pooling kernel/stride)

    Returns:
        Tensor of shape (H_pad // patch_size, W_pad // patch_size, C)
    """
    feat_t = pixel_desc.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    if pad_h or pad_w:
        feat_t = torch.nn.functional.pad(feat_t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    pooled = torch.nn.functional.avg_pool2d(feat_t, kernel_size=patch_size, stride=patch_size)
    return pooled.squeeze(0).permute(1, 2, 0)


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
    def build_instance_colormap(num_instances: int) -> ListedColormap:
        if num_instances <= 0:
            return ListedColormap([(0, 0, 0, 1)])
        # Evenly spaced hues; alpha=1
        base = [plt.cm.hsv(t) for t in np.linspace(0, 1, num_instances, endpoint=False)]
        colors = [(0, 0, 0, 1)] + [tuple(c[:3]) + (1,) for c in base]
        return ListedColormap(colors)

    @staticmethod
    def make_viz_mask_and_cmap(instance_mask: np.ndarray) -> Tuple[np.ndarray, ListedColormap, BoundaryNorm]:
        if instance_mask.size == 0:
            cmap = ListedColormap([(0, 0, 0, 1)])
            norm = BoundaryNorm([-0.5, 0.5], ncolors=1)
            return instance_mask, cmap, norm

        ids = np.unique(instance_mask)
        ids = ids[ids != 0]
        num_instances = int(len(ids))
        if num_instances == 0:
            cmap = ListedColormap([(0, 0, 0, 1)])
            norm = BoundaryNorm([-0.5, 0.5], ncolors=1)
            return instance_mask, cmap, norm

        # 1) Centroids per instance id (row, col)
        centroids = {}
        for iid in ids:
            rr, cc = np.where(instance_mask == iid)
            # Safe even for 1-pixel instances
            centroids[iid] = np.array([rr.mean(), cc.mean()], dtype=np.float64)

        # 2) Farthest-point sampling over centroids to order instance IDs
        ids_list = ids.tolist()
        start_id = int(np.random.choice(ids_list))
        ordered = [start_id]
        remaining = set(ids_list)
        remaining.remove(start_id)

        while remaining:
            # for each remaining, distance to nearest selected centroid
            d_best, pick = -1.0, None
            for r in remaining:
                cr = centroids[r]
                dmin = min(np.linalg.norm(cr - centroids[s]) for s in ordered)
                if dmin > d_best:
                    d_best, pick = dmin, r
            ordered.append(pick)
            remaining.remove(pick)

        # 3) Remap mask IDs so color index 1..K follows the farthest-point order
        viz_mask = np.zeros_like(instance_mask, dtype=instance_mask.dtype)
        for new_idx, iid in enumerate(ordered, start=1):
            viz_mask[instance_mask == iid] = new_idx

        # 4) Colors: black for 0, then gradual hues for 1..K
        cmap = SAM2utils.build_instance_colormap(num_instances)
        boundaries = np.arange(num_instances + 2) - 0.5  # [-0.5, 0.5, 1.5, ..., K+0.5]
        norm = BoundaryNorm(boundaries, ncolors=num_instances + 1)
        return viz_mask, cmap, norm

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

    # ==== Misc helpers (moved from sam2_client) ====
    @staticmethod
    def prevent_oom_resizing(image: Union[str, Image.Image], target: int) -> Image.Image:
        img = Image.open(image) if isinstance(image, str) else image
        w, h = img.width, img.height
        max_side = max(w, h)
        if max_side == target:
            return img
        scale = float(target) / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), resample=Image.LANCZOS)

    @staticmethod
    def auto_masks_to_instance_mask(
        masks_list: List[Dict[str, Any]],
        min_iou: float = 0.0,
        min_area: float = 0.0,
        assign_by: str = "iou",
        start_from: str = "high",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not masks_list:
            return np.zeros((1, 1), dtype=np.uint16), {"num_instances": 0}

        # Filter masks by criteria (binary mask expectation)
        filtered_masks = [m for m in masks_list
                          if m.get('predicted_iou', 0) >= min_iou and m.get('area', 0) >= min_area]
        if not filtered_masks:
            return np.zeros((1, 1), dtype=np.uint16), {"num_instances": 0}

        # Choose ranking key
        def _area_of(m: Dict[str, Any]) -> float:
            a = m.get('area')
            if isinstance(a, (int, float)):
                return float(a)
            seg = m.get('segmentation')
            return float(np.count_nonzero(seg)) if isinstance(seg, np.ndarray) else 0.0

        def _key(m: Dict[str, Any]) -> float:
            return float(m.get('predicted_iou', 0.0)) if assign_by == "iou" else _area_of(m)

        # Order to assign instance ids and claim pixels
        reverse = (start_from == "low")
        ordered = sorted(filtered_masks, key=_key, reverse=reverse)

        # Allocate instance ids; earlier assignments keep pixels (no overwrite)
        first_mask = ordered[0]['segmentation']
        h, w = first_mask.shape
        instance_mask = np.zeros((h, w), dtype=np.uint16)

        for idx, mask_dict in enumerate(ordered):
            instance_id = idx + 1
            mask = mask_dict['segmentation']
            if mask.dtype != np.bool_:
                mask = mask.astype(bool)
            claim = mask & (instance_mask == 0)
            instance_mask[claim] = instance_id

        stats = {
            "num_instances": len(filtered_masks),
            "total_masks_before_filter": len(masks_list),
            "min_iou_used": min_iou,
            "min_area_used": min_area,
            "assign_by": assign_by,
            "start_from": start_from,
        }
        return instance_mask, stats

    @staticmethod
    def compute_overlap_stats(masks_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not masks_list:
            return {"total_masks": 0, "overlap_matrix": []}
        n_masks = len(masks_list)
        overlap_matrix = np.zeros((n_masks, n_masks))
        mask_arrays = [(m['segmentation'] > 0) for m in masks_list]
        for i in range(n_masks):
            for j in range(i, n_masks):
                mi = mask_arrays[i]
                mj = mask_arrays[j]
                inter = np.logical_and(mi, mj).sum()
                uni = np.logical_or(mi, mj).sum()
                iou = inter / uni if uni > 0 else 0.0
                overlap_matrix[i, j] = iou
                overlap_matrix[j, i] = iou
        upper = overlap_matrix[np.triu_indices(n_masks, k=1)]
        return {
            "total_masks": n_masks,
            "overlap_matrix": overlap_matrix.tolist(),
            "mean_overlap": float(upper.mean()) if len(upper) > 0 else 0.0,
            "max_overlap": float(upper.max()) if len(upper) > 0 else 0.0,
            "high_overlap_pairs": int((upper > 0.5).sum()) if len(upper) > 0 else 0
        }

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

    @staticmethod
    def resize_auto_masks(auto_masks: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
        """
        Resize SAM2 automatic masks to have max(image_side) == target.
        - Resizes binary `segmentation` with nearest-neighbor
        - Updates `area`, `bbox` (xywh), `point_coords`, and `crop_box` (xywh)
        - Keeps `predicted_iou` and `stability_score`
        """
        if not auto_masks:
            return auto_masks

        # All masks share the same original size
        seg0 = auto_masks[0]["segmentation"]
        h, w = seg0.shape
        max_side = max(w, h)
        if max_side == target:
            return auto_masks

        scale = float(target) / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized_masks: List[Dict[str, Any]] = []
        for m in auto_masks:
            seg = m["segmentation"].astype(np.uint8) * 255
            seg_img = Image.fromarray(seg, mode="L")
            seg_resized = seg_img.resize((new_w, new_h), resample=Image.NEAREST)
            seg_resized_np = (np.array(seg_resized) > 127)

            # Scale geometry (xywh for bbox/crop_box; xy for points)
            bbox = m.get("bbox", None)
            if bbox is not None:
                x, y, bw, bh = bbox
                bbox_scaled = [x * scale, y * scale, bw * scale, bh * scale]
            else:
                bbox_scaled = None

            pts = m.get("point_coords", None)
            if pts is not None:
                pts_scaled = [[px * scale, py * scale] for px, py in pts]
            else:
                pts_scaled = None

            crop = m.get("crop_box", None)
            if crop is not None:
                cx, cy, cw, ch = crop
                crop_scaled = [cx * scale, cy * scale, cw * scale, ch * scale]
            else:
                crop_scaled = None

            resized_masks.append({
                "segmentation": seg_resized_np,
                "area": int(seg_resized_np.sum()),
                "bbox": bbox_scaled if bbox_scaled is not None else m.get("bbox"),
                "predicted_iou": m.get("predicted_iou", 0.0),
                "point_coords": pts_scaled if pts_scaled is not None else m.get("point_coords"),
                "stability_score": m.get("stability_score", 0.0),
                "crop_box": crop_scaled if crop_scaled is not None else m.get("crop_box"),
            })

        return resized_masks


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
