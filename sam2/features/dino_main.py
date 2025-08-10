import torch
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
from sam2.features.dino_vit_extractor import ViTExtractor
from sam2.features.utils import preprocess_image, apply_pca_colormap, pad_to_multiple, upsample_and_unpad


class DINOfeatures:
    def __init__(self, model_type="dinov2_vitl14", stride=None, device=None):
        self.model_type = model_type
        self.stride = stride
        self.device = device or torch.device("cpu")

        # Initialize DINO extractor
        self.extractor = ViTExtractor(model_type=self.model_type, stride=self.stride, device=self.device)

        # Get patch size for padding
        self.patch_size = self.extractor.p

        # DINO normalization values
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @torch.inference_mode()
    def extract(self,
                pil_img,
                ret_pca=False,
                ret_patches=True,
                load_size=-1,
                facet="token",
                layer=-1,
                bin=False,
                pca_niter=5,
                pca_q_min=0.01,
                pca_q_max=0.99,
                interpolation_mode="bilinear",
                tensor_format="HWC",
                padding_mode="constant"):
        """
        Extract DINO features from an image.

        Args:
            pil_img: PIL Image to process
            ret_pca: If True, return 3D PCA visualization, else full D-dim features
            ret_patches: If True, return patch grid, else upsample to full image size
            load_size: Target size for resizing (None for no resize, negative for scale smallest side)
            facet: Feature facet to extract ("token", "key", "query", "value", "attn")
            layer: Layer to extract from (-1 for last layer)
            bin: Whether to apply log binning to descriptors
            pca_niter: Number of iterations for PCA computation
            pca_q_min, pca_q_max: Quantile range for PCA normalization
            interpolation_mode: Interpolation mode for upsampling ("bilinear", "nearest", etc.)
            tensor_format: Output tensor format ("HWC", "CHW", etc.)
            padding_mode: Padding mode for pad_to_multiple ("constant", "reflect", etc.)
        """
        # Step 1: Preprocess image
        img_tensor, orig_w, orig_h = preprocess_image(
            pil_img,
            load_size=load_size,
            center_crop=False,
            mean=self.mean,
            std=self.std
        )
        img_tensor = img_tensor.to(self.device)

        # Step 2: Pad to multiple of patch size
        img_tensor, pad_h, pad_w = pad_to_multiple(img_tensor, self.patch_size, format="BCHW", mode=padding_mode)

        # Step 3: Extract patch descriptors
        desc = self.extractor.extract_descriptors(img_tensor, layer=layer, facet=facet, bin=bin)
        desc = desc[0, 0]  # Remove batch and sequence dims: (B, 1, T, D) -> (T, D)

        # Step 4: Reshape to patch grid
        _, _, padded_h, padded_w = img_tensor.shape
        h_p, w_p = padded_h // self.patch_size, padded_w // self.patch_size
        desc = rearrange(desc, "(h w) d -> h w d", h=h_p, w=w_p)

        # Step 5: Optional PCA for visualization
        if ret_pca:
            desc = apply_pca_colormap(desc, niter=pca_niter, q_min=pca_q_min, q_max=pca_q_max)

        # Step 6: Optional upsample to full image size and unpad
        if not ret_patches:
            desc = upsample_and_unpad(desc, (padded_h, padded_w), pad_h, pad_w, tensor_format=tensor_format, mode=interpolation_mode)

        return desc


if __name__ == "__main__":
    img = Image.open("notebooks/images/cars.jpg")
    dino = DINOfeatures(device=torch.device("cuda"))
    desc_pca = dino.extract(img, ret_pca=True, ret_patches=False, load_size=2048)
    print(f"Input image size: h={img.height}, w={img.width}")
    print(f"Output shape: {desc_pca.shape}")
    plt.imshow(desc_pca.cpu().numpy())
    plt.title("DINO Patch PCA Visualization")
    plt.axis("off")
    plt.show()
