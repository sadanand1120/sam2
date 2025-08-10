import asyncio
import torch
from typing import Optional, Union
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from sam2.features.dino_main import DINOfeatures
from sam2.features.utils import AsyncModelRunnerParallel, AsyncMultiWrapper


class DINOFeaturesUnified(AsyncModelRunnerParallel):
    def __init__(self, device: Union[str, torch.device]):
        super().__init__(device=device)

    def _get_model_key(self, model_type: str, stride: Optional[int]) -> str:
        """Unique key for model instance per device per class object"""
        return f"{id(self)}_{model_type}_{stride}"

    def extract_features(self,
                         image: Optional[Union[str, Path, Image.Image]] = None,
                         image_url: Optional[str] = None,
                         model_type: str = "dinov2_vitl14",
                         stride: Optional[int] = None,
                         ret_pca: bool = False,
                         ret_patches: bool = True,
                         load_size: int = -1,
                         facet: str = "token",
                         layer: int = -1,
                         bin: bool = False,
                         pca_niter: int = 5,
                         pca_q_min: float = 0.01,
                         pca_q_max: float = 0.99,
                         interpolation_mode: str = "bilinear",
                         tensor_format: str = "HWC",
                         padding_mode: str = "constant") -> torch.Tensor:
        """Extract DINO features from image."""
        pil_img = self.load_image(image, image_url)
        key = self._get_model_key(model_type, stride)
        model = self.get_or_create_model(key, lambda: DINOfeatures(model_type=model_type, stride=stride, device=self.device))
        lock = self.get_lock(key)
        with lock:
            features = model.extract(
                pil_img=pil_img,
                ret_pca=ret_pca,
                ret_patches=ret_patches,
                load_size=load_size,
                facet=facet,
                layer=layer,
                bin=bin,
                pca_niter=pca_niter,
                pca_q_min=pca_q_min,
                pca_q_max=pca_q_max,
                interpolation_mode=interpolation_mode,
                tensor_format=tensor_format,
                padding_mode=padding_mode
            )
        return features

    async def extract_features_async(self, **kwargs) -> torch.Tensor:
        return await self.run_in_executor(self.extract_features, **kwargs)


if __name__ == "__main__":

    # Single wrapper that internally manages multiple workers/devices
    dino = AsyncMultiWrapper(DINOFeaturesUnified, num_objects=2)

    test_image_path = "notebooks/images/cars.jpg"

    print("=== DINO API Demo ===")
    img = Image.open(test_image_path)
    print(f"Input image size: h={img.height}, w={img.width}")
    desc_pca = dino.extract_features(
        image=test_image_path,
        ret_pca=True,
        ret_patches=False,
        load_size=2048
    )
    print(f"Output shape: {desc_pca.shape}")
    plt.imshow(desc_pca.cpu().numpy())
    plt.title("DINO Patch PCA Visualization")
    plt.axis("off")
    plt.show()

    print("=== Async Demo ===")
    # Warmup num_objects times to load models on underlying workers
    _ = dino.extract_features(image=test_image_path, ret_pca=True, ret_patches=False, load_size=512)
    _ = dino.extract_features(image=test_image_path, ret_pca=True, ret_patches=False, load_size=512)

    tasks = [dino.extract_features_async(image=test_image_path, ret_pca=True, ret_patches=False, load_size=2048) for _ in range(8)]
    results = asyncio.run(AsyncMultiWrapper.async_run_tasks(tasks, desc="DINO tasks"))

    plt.imshow(results[0].cpu().numpy())
    plt.title("DINO Patch PCA Visualization")
    plt.axis("off")
    plt.show()

    print("✅ Direct DINO API demo completed successfully!")
