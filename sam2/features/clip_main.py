import torch
from PIL import Image
import numpy as np
import open_clip
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from einops import rearrange
from sam2.features.utils import (
    preprocess_image,
    interpolate_positional_embedding,
    apply_pca_colormap,
    pad_to_multiple,
    upsample_and_unpad,
    repackage_pixels_to_patch_grid,
)
from tqdm import tqdm
from typing import List


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
                padding_mode="constant",
                return_meta: bool = False):
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

        if return_meta:
            meta = {
                "padded_h": padded_h,
                "padded_w": padded_w,
                "pad_h": pad_h,
                "pad_w": pad_w,
            }
            return patch_desc, meta
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
    def extract_agg(self,
                    pil_img,
                    agg_scales: List[float],
                    agg_weights=None,
                    ret_pca: bool = False,
                    ret_patches: bool = True,
                    load_size: int = 1024,
                    center_crop: bool = False,
                    pca_niter: int = 5,
                    pca_q_min: float = 0.01,
                    pca_q_max: float = 0.99,
                    interpolation_mode: str = "bilinear",
                    tensor_format: str = "HWC",
                    padding_mode: str = "constant",
                    return_meta: bool = False):
        """
        Multi-scale wrapper over extract() with patch-space aggregation.
        - For each scale s in agg_scales, runs extract at load_size * s to get a patch grid.
        - Interpolates each patch grid to the reference (scale=1.0) patch grid size and averages in patch space.
        - If agg_weights is provided, uses weighted average (must match agg_scales length); else equal weights.
        - Returns the averaged patch grid (ret_patches=True), or upsamples once to pixel level (ret_patches=False).

        Output shapes match extract() for the given load_size and flags.
        """
        if 1.0 not in agg_scales:
            raise ValueError("agg_scales must include 1.0 as the reference scale")

        # Reference (scale = 1.0) patch grid and meta
        ref_patch, ref_meta = self.extract(
            pil_img,
            ret_pca=False,
            ret_patches=True,
            load_size=load_size,
            center_crop=center_crop,
            pca_niter=pca_niter,
            pca_q_min=pca_q_min,
            pca_q_max=pca_q_max,
            interpolation_mode=interpolation_mode,
            tensor_format=tensor_format,
            padding_mode=padding_mode,
            return_meta=True,
        )
        H_pad_ref, W_pad_ref = ref_meta["padded_h"], ref_meta["padded_w"]
        pad_h_ref, pad_w_ref = ref_meta["pad_h"], ref_meta["pad_w"]

        # Prepare weights
        if agg_weights is not None:
            if not isinstance(agg_weights, (list, tuple)) or len(agg_weights) != len(agg_scales):
                raise ValueError("agg_weights must be a list with the same length as agg_scales")
            weights = [float(w) for w in agg_weights]
        else:
            weights = [1.0] * len(agg_scales)

        # Reference patch grid size (avoid pixel-level tensors to reduce VRAM)
        h_ref, w_ref, d_ref = ref_patch.shape
        agg_t = torch.zeros((d_ref, h_ref, w_ref), device=ref_patch.device, dtype=ref_patch.dtype)

        # Add reference contribution
        ref_idx = agg_scales.index(1.0)
        weight_ref = float(weights[ref_idx])
        if weight_ref > 0.0:
            agg_t += ref_patch.permute(2, 0, 1).contiguous() * weight_ref
        weight_accum = max(weight_ref, 0.0)
        # Free ref tensor early to save VRAM
        del ref_patch
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Exclude the reference scale (1.0) from the loop to avoid duplicate computation
        other = [(i, s) for i, s in enumerate(agg_scales) if i != ref_idx]

        for i, s in other:
            scaled_load = int(round(load_size * float(s)))
            if scaled_load <= 0:
                continue
            patch_s, meta_s = self.extract(
                pil_img,
                ret_pca=False,
                ret_patches=True,
                load_size=scaled_load,
                center_crop=center_crop,
                pca_niter=pca_niter,
                pca_q_min=pca_q_min,
                pca_q_max=pca_q_max,
                interpolation_mode=interpolation_mode,
                tensor_format=tensor_format,
                padding_mode=padding_mode,
                return_meta=True,
            )
            # Align patch_s to the reference patch grid (memory-light compared to pixel-level)
            ps_t = patch_s.permute(2, 0, 1).unsqueeze(0)  # (1, d, h_s, w_s)
            target_size_hw = (int(h_ref), int(w_ref))
            if interpolation_mode in ("bilinear", "bicubic", "trilinear"):
                ps_t = F.interpolate(ps_t, size=target_size_hw, mode=interpolation_mode, align_corners=False)
            else:
                ps_t = F.interpolate(ps_t, size=target_size_hw, mode=interpolation_mode)
            w_i = float(weights[i])
            if w_i > 0.0:
                agg_t += ps_t.squeeze(0).to(dtype=agg_t.dtype) * w_i  # (d, h_ref, w_ref)
                weight_accum += w_i
            # Cleanup per-scale tensors promptly
            del patch_s, meta_s, ps_t
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Average in patch space and convert back to (h_ref, w_ref, d)
        if weight_accum <= 0.0:
            raise ValueError("Sum of weights must be > 0")
        agg_t = agg_t / weight_accum
        agg_patch = agg_t.permute(1, 2, 0)
        # Free accumulator before optional pixel upsample
        del agg_t
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()

        if ret_pca:
            patch_out = apply_pca_colormap(agg_patch, niter=pca_niter, q_min=pca_q_min, q_max=pca_q_max)
        else:
            patch_out = agg_patch

        if ret_patches:
            if return_meta:
                return patch_out, ref_meta
            return patch_out
        else:
            # Produce pixel-level output once at the end to limit VRAM usage
            src = patch_out if ret_pca else agg_patch
            pix = upsample_and_unpad(src, (H_pad_ref, W_pad_ref), pad_h_ref, pad_w_ref, tensor_format=tensor_format, mode=interpolation_mode)
            if return_meta:
                return pix, ref_meta
            return pix

    @torch.inference_mode()
    def encode_text(self, text):
        # Returns normalized text embedding for a single string as a torch tensor (device preserved)
        tokenizer = open_clip.get_tokenizer(self.model_name)
        tokens = tokenizer([text]).to(self.device)
        text_emb = self.model.encode_text(tokens).squeeze()
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb

    @torch.inference_mode()
    def compute_similarity(self, patch_desc, text_emb, neg_text_embs=None, softmax_temp=None, normalize: bool = True):
        """
        Compute per-patch similarity between visual patch descriptors and a text embedding.

        Args:
            patch_desc: (h, w, d) patch descriptors.
            text_emb: (d,) positive text embedding.
            neg_text_embs: optional tensor of negative text embeddings, shape (K, d).
            softmax_temp: temperature T for probability over [pos + negatives].
                - If T is None or <= 0: no softmax is applied.
                - If T > 0 and negatives are provided: per-pixel probabilities are computed as
                  softmax([s_pos, s_neg1..K] / T). The output is P(pos) at each pixel.
                  Interpretation of T:
                    - T < 1: sharper, more peaky probabilities (closer to argmax)
                    - T = 1: standard softmax
                    - T > 1: smoother, more uniform probabilities
            normalize: if True (default), L2-normalize patch and text embeddings to compute cosine similarity;
                if False, use raw dot product.

        Returns:
            - If neg_text_embs is None: (h, w) cosine similarity map (in [-1, 1] when normalize=True).
            - If neg_text_embs is provided and softmax_temp is None or <= 0: (h, w) margin map equal to
              s_pos - mean_k(s_neg,k). This is a contrast score, not a probability.
            - If neg_text_embs is provided and softmax_temp > 0: (h, w) probability map in [0, 1],
              where each pixel is P(pos | {pos, negs}) under softmax with temperature T.
        """
        h, w, d = patch_desc.shape
        patch_flat = patch_desc.reshape(-1, d)
        if normalize:
            patch_flat = patch_flat / (patch_flat.norm(dim=1, keepdim=True) + 1e-6)
            text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-6)
        # Positive similarity
        sim_pos = patch_flat @ text_emb

        # Aggregate negatives if provided (expect tensor of shape (K, d))
        if isinstance(neg_text_embs, torch.Tensor) and neg_text_embs.numel() > 0:
            neg_text_embs = neg_text_embs.to(patch_flat.device, dtype=patch_flat.dtype)
            if normalize:
                neg_text_embs = neg_text_embs / (neg_text_embs.norm(dim=1, keepdim=True) + 1e-6)
            sim_negs = patch_flat @ neg_text_embs.T  # (N, K)
            if softmax_temp is not None and softmax_temp > 0:
                logits = torch.cat([sim_pos.unsqueeze(1), sim_negs], dim=1)  # (N, K+1)
                probs = torch.softmax(logits / softmax_temp, dim=1)
                sim = probs[:, 0]
            else:
                neg_agg = sim_negs.mean(dim=1)
                sim = sim_pos - neg_agg
        else:
            sim = sim_pos

        return sim.reshape(h, w)


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

    # Demo: multi-scale aggregation
    desc_agg = clipper.extract_agg(img, agg_scales=[0.25, 0.5, 1.0, 1.5], agg_weights=[1, 2, 5.5, 2], ret_pca=True, ret_patches=False, load_size=2048)
    print(f"Output shape: {desc_agg.shape}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(desc_pca.cpu().numpy())
    axs[0].set_title("CLIP Patch PCA Visualization")
    axs[0].axis("off")
    axs[1].imshow(desc_agg.cpu().numpy())
    axs[1].set_title("CLIP Patch PCA Visualization (Multi-scale Aggregation)")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

    # Demo: text similarity, do similarity compute on patches for mem efficiency -> then upsample
    desc, meta = clipper.extract(img, ret_pca=False, ret_patches=True, load_size=2048, return_meta=True)
    print(f"Descriptor shape: {desc.shape}")
    text_emb = clipper.encode_text("car")
    print(f"Text embedding shape: {text_emb.shape}")
    sim_map = clipper.compute_similarity(desc, text_emb)
    sim_vis = upsample_and_unpad(sim_map.unsqueeze(-1), (meta["padded_h"], meta["padded_w"]), meta["pad_h"], meta["pad_w"], tensor_format="HWC", mode="bilinear").squeeze()
    plt.imshow(sim_vis.cpu().numpy(), cmap="turbo")
    plt.title("Similarity to 'car'")
    plt.axis("off")
    plt.show()

    # Demo: multi-scale aggregation similarity
    desc, meta = clipper.extract_agg(img, agg_scales=[0.25, 0.5, 1.0, 1.5], agg_weights=[1, 2, 5.5, 2], ret_pca=False, ret_patches=True, load_size=2048, return_meta=True)
    print(f"Descriptor shape: {desc.shape}")
    text_emb = clipper.encode_text("car")
    print(f"Text embedding shape: {text_emb.shape}")
    sim_map = clipper.compute_similarity(desc, text_emb)
    sim_vis = upsample_and_unpad(sim_map.unsqueeze(-1), (meta["padded_h"], meta["padded_w"]), meta["pad_h"], meta["pad_w"], tensor_format="HWC", mode="bilinear").squeeze()
    plt.imshow(sim_vis.cpu().numpy(), cmap="turbo")
    plt.title("Similarity to 'car'")
    plt.axis("off")
    plt.show()
