import torch
from PIL import Image
import numpy as np
import open_clip
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from einops import rearrange
from sam2.features.utils import preprocess_image, interpolate_positional_embedding, apply_pca_colormap, pad_to_multiple, upsample_and_unpad


class CLIPfeatures:
    def __init__(self, model_name="ViT-L-14-336-quickgelu", model_pretrained="openai", device=None):
        self.model_name = model_name
        self.model_pretrained = model_pretrained
        self.device = device or torch.device("cpu")

        # Initialize CLIP model
        self.model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.model_pretrained, device=self.device)
        self.model.eval()

        # Get patch size for padding
        if self.model_name.startswith("ViT"):
            self.patch_size = self.model.visual.patch_size[0]
        else:
            raise NotImplementedError("Only ViT models supported.")

        # CLIP normalization values
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)

    @torch.inference_mode()
    def extract(self,
                pil_img,
                ret_pca=False,
                ret_patches=True,
                load_size=1024,
                center_crop=False,
                pca_niter=5,
                pca_q_min=0.01,
                pca_q_max=0.99,
                interpolation_mode="bilinear",
                tensor_format="HWC",
                padding_mode="constant"):
        """
        Extract CLIP features from an image.

        Args:
            pil_img: PIL Image to process
            ret_pca: If True, return 3D PCA visualization, else full D-dim features
            ret_patches: If True, return patch grid, else upsample to full image size
            load_size: Target size for resizing (None for no resize, negative for scale smallest side)
            center_crop: Whether to center crop after resizing
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
            center_crop=center_crop,
            mean=self.mean,
            std=self.std
        )
        img_tensor = img_tensor.to(self.device)

        # Step 2: Pad to multiple of patch size
        img_tensor, pad_h, pad_w = pad_to_multiple(img_tensor, self.patch_size, format="BCHW", mode=padding_mode)

        # Step 3: Extract patch descriptors
        patch_desc = self._get_patch_encodings(img_tensor)

        # Step 4: Reshape to patch grid
        _, _, padded_h, padded_w = img_tensor.shape
        h_p, w_p = padded_h // self.patch_size, padded_w // self.patch_size
        patch_desc = rearrange(patch_desc, "b (h w) d -> b h w d", h=h_p, w=w_p)  # (1, h, w, d)
        patch_desc = patch_desc[0]  # Remove batch dim: (h, w, d)

        # Step 5: Optional PCA for visualization
        if ret_pca:
            patch_desc = apply_pca_colormap(patch_desc, niter=pca_niter, q_min=pca_q_min, q_max=pca_q_max)

        # Step 6: Optional upsample to full image size and unpad
        if not ret_patches:
            patch_desc = upsample_and_unpad(patch_desc, (padded_h, padded_w), pad_h, pad_w, tensor_format=tensor_format, mode=interpolation_mode)

        return patch_desc

    @torch.inference_mode()
    def _get_patch_encodings(self, image_batch):
        _, _, w, h = image_batch.shape
        x = self.model.visual.conv1(image_batch)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + interpolate_positional_embedding(self.model.visual.positional_embedding, x, patch_size=self.model.visual.patch_size[0], w=w, h=h)
        x = self.model.visual.ln_pre(x)
        *layers, last_resblock = self.model.visual.transformer.resblocks
        penultimate = torch.nn.Sequential(*layers)
        x = penultimate(x)
        v_in_proj_weight = last_resblock.attn.in_proj_weight[-last_resblock.attn.embed_dim:]
        v_in_proj_bias = last_resblock.attn.in_proj_bias[-last_resblock.attn.embed_dim:]
        v_in = F.linear(last_resblock.ln_1(x), v_in_proj_weight, v_in_proj_bias)
        x = F.linear(v_in, last_resblock.attn.out_proj.weight, last_resblock.attn.out_proj.bias)
        x = x[:, 1:, :]
        x = self.model.visual.ln_post(x)
        if self.model.visual.proj is not None:
            x = x @ self.model.visual.proj
        return x

    @torch.inference_mode()
    def encode_text(self, text):
        # Returns normalized text embedding for a single string as a torch tensor (device preserved)
        tokenizer = open_clip.get_tokenizer(self.model_name)
        tokens = tokenizer([text]).to(self.device)
        text_emb = self.model.encode_text(tokens).squeeze()
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb

    @torch.inference_mode()
    def compute_similarity(self, patch_desc, text_emb, softmax=1.0):
        # patch_desc: (h, w, d) or (h, w, 3) if PCA, text_emb: (d,) or (3,)
        # Returns similarity map (h, w) as numpy (for viz)
        h, w, d = patch_desc.shape
        patch_flat = patch_desc.reshape(-1, d)
        sim = patch_flat @ text_emb
        sim = sim.reshape(h, w)
        if softmax > 0:
            sim_exp = torch.exp(sim * softmax)
            sim = sim_exp / sim_exp.sum()
        return sim


if __name__ == "__main__":
    img = Image.open("notebooks/images/cars.jpg")
    clipper = CLIPfeatures(device=torch.device("cuda"))
    desc_pca = clipper.extract(img, ret_pca=True, ret_patches=False, load_size=2048)
    print(f"Input image size: h={img.height}, w={img.width}")
    print(f"Output shape: {desc_pca.shape}")
    plt.imshow(desc_pca.cpu().numpy())
    plt.title("CLIP Patch PCA Visualization")
    plt.axis("off")
    plt.show()

    # Demo: text similarity
    desc = clipper.extract(img, ret_pca=False, ret_patches=False, load_size=2048)
    print(f"Descriptor shape: {desc.shape}")
    text_emb = clipper.encode_text("car")
    print(f"Text embedding shape: {text_emb.shape}")
    sim_map = clipper.compute_similarity(desc, text_emb, softmax=0.25)
    plt.imshow(sim_map.cpu().numpy(), cmap="turbo")
    plt.title("Similarity to 'car'")
    plt.axis("off")
    plt.show()
