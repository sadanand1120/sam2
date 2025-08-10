import asyncio
import torch
from typing import Dict, Optional, Union, Tuple
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from sam2.features.clip_main import CLIPfeatures
from sam2.features.utils import upsample_and_unpad, AsyncModelRunnerParallel, AsyncMultiWrapper


class CLIPFeaturesUnified(AsyncModelRunnerParallel):
    def __init__(self, device: Union[str, torch.device]):
        super().__init__(device=device)

    def _get_model_key(self, model_name: str, model_pretrained: str) -> str:
        # Unique per instance to avoid cross-instance sharing
        return f"{id(self)}_{model_name}_{model_pretrained}"

    def _get_model(self, model_name: str, model_pretrained: str) -> CLIPfeatures:
        key = self._get_model_key(model_name, model_pretrained)
        return self.get_or_create_model(key, lambda: CLIPfeatures(model_name=model_name, model_pretrained=model_pretrained, device=self.device))

    def _get_lock(self, model_name: str, model_pretrained: str):
        key = self._get_model_key(model_name, model_pretrained)
        return self.get_lock(key)

    def extract_features(self,
                         image: Optional[Union[str, Path, Image.Image]] = None,
                         image_url: Optional[str] = None,
                         model_name: str = "ViT-L-14-336-quickgelu",
                         model_pretrained: str = "openai",
                         ret_pca: bool = False,
                         ret_patches: bool = True,
                         load_size: Optional[int] = 1024,
                         center_crop: bool = False,
                         pca_niter: int = 5,
                         pca_q_min: float = 0.01,
                         pca_q_max: float = 0.99,
                         interpolation_mode: str = "bilinear",
                         tensor_format: str = "HWC",
                         padding_mode: str = "constant",
                         return_meta: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, int]]]:
        """Extract CLIP features from image."""
        pil_img = self.load_image(image, image_url)
        model = self._get_model(model_name, model_pretrained)
        lock = self._get_lock(model_name, model_pretrained)

        with lock:
            result = model.extract(
                pil_img=pil_img,
                ret_pca=ret_pca,
                ret_patches=ret_patches,
                load_size=load_size,
                center_crop=center_crop,
                pca_niter=pca_niter,
                pca_q_min=pca_q_min,
                pca_q_max=pca_q_max,
                interpolation_mode=interpolation_mode,
                tensor_format=tensor_format,
                padding_mode=padding_mode,
                return_meta=return_meta
            )
        return result

    def encode_text(self,
                    text: str,
                    model_name: str = "ViT-L-14-336-quickgelu",
                    model_pretrained: str = "openai") -> torch.Tensor:
        """Encode text to embeddings."""
        model = self._get_model(model_name, model_pretrained)
        return model.encode_text(text)

    def compute_similarity(self,
                           image: Optional[Union[str, Path, Image.Image]] = None,
                           image_url: Optional[str] = None,
                           text: str = "",
                           model_name: str = "ViT-L-14-336-quickgelu",
                           model_pretrained: str = "openai",
                           softmax: float = 1.0,
                           load_size: Optional[int] = 1024,
                           center_crop: bool = False,
                           padding_mode: str = "constant") -> torch.Tensor:
        """Compute CLIP image-text similarity map."""
        pil_img = self.load_image(image, image_url)
        model = self._get_model(model_name, model_pretrained)
        lock = self._get_lock(model_name, model_pretrained)
        # Extract image features at patch resolution by default for memory efficiency
        with lock:
            features, meta = model.extract(
                pil_img=pil_img,
                ret_pca=False,
                ret_patches=True,
                load_size=load_size,
                center_crop=center_crop,
                padding_mode=padding_mode,
                return_meta=True
            )
            # Encode text
            text_emb = model.encode_text(text)
        # Compute similarity at patch resolution
        sim_patch = model.compute_similarity(features, text_emb, softmax=softmax)
        sim_full = upsample_and_unpad(
            sim_patch.unsqueeze(-1),
            (meta["padded_h"], meta["padded_w"]), meta["pad_h"], meta["pad_w"],
            tensor_format="HWC", mode="bilinear").squeeze(-1)
        return sim_full

    async def extract_features_async(self, **kwargs) -> torch.Tensor:
        return await self.run_in_executor(self.extract_features, **kwargs)

    async def encode_text_async(self, **kwargs) -> torch.Tensor:
        return await self.run_in_executor(self.encode_text, **kwargs)

    async def compute_similarity_async(self, **kwargs) -> torch.Tensor:
        return await self.run_in_executor(self.compute_similarity, **kwargs)


if __name__ == "__main__":
    clip = AsyncMultiWrapper(CLIPFeaturesUnified, num_objects=2)
    test_image_path = "notebooks/images/cars.jpg"

    print("=== Direct CLIP API Demo ===")

    img = Image.open(test_image_path)
    print(f"Input image size: h={img.height}, w={img.width}")
    desc_pca = clip.extract_features(
        image=test_image_path,
        ret_pca=True,
        ret_patches=False,
        load_size=2048,
    )
    print(f"Output shape: {desc_pca.shape}")
    plt.imshow(desc_pca.cpu().numpy())
    plt.title("CLIP Patch PCA Visualization")
    plt.axis("off")
    plt.show()

    # Demo: Text similarity (patch resolution + upsample for viz)
    sim_map = clip.compute_similarity(
        image=test_image_path,
        text="car",
        load_size=2048,
        softmax=0.25,
    )
    plt.imshow(sim_map.cpu().numpy(), cmap="turbo")
    plt.title("Similarity to 'car'")
    plt.axis("off")
    plt.show()

    print("=== Async Demo ===")
    # warmup num_objects times so it doesnt load twice in async mode
    _ = clip.extract_features(image=test_image_path, ret_pca=True, ret_patches=False, load_size=512)
    _ = clip.extract_features(image=test_image_path, ret_pca=True, ret_patches=False, load_size=512)

    feat_tasks = [clip.extract_features_async(image=test_image_path, ret_pca=True, ret_patches=False, load_size=2048) for _ in range(8)]
    feat_results = asyncio.run(AsyncMultiWrapper.async_run_tasks(feat_tasks, desc="CLIP feature tasks"))
    plt.imshow(feat_results[0].cpu().numpy())
    plt.title("CLIP PCA Visualization (async result)")
    plt.axis("off")
    plt.show()

    text_tasks = [clip.encode_text_async(text="car") for _ in range(8)]
    text_results = asyncio.run(AsyncMultiWrapper.async_run_tasks(text_tasks, desc="CLIP text tasks"))
    print(f"Async car embedding shape: {text_results[0].shape}")
    print(f"Async car embedding shape: {text_results[1].shape}")

    sim_tasks = [clip.compute_similarity_async(image=test_image_path, text="car", load_size=2048, softmax=0.25) for _ in range(8)]
    sim_results = asyncio.run(AsyncMultiWrapper.async_run_tasks(sim_tasks, desc="CLIP sim tasks"))
    plt.imshow(sim_results[0].cpu().numpy(), cmap="turbo")
    plt.title("Similarity to 'car' (async)")
    plt.axis("off")
    plt.show()

    print("✅ Direct CLIP API demo completed successfully!")
