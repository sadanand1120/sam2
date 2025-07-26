import asyncio
import base64
import io
import logging
import numpy as np
import os
import random
import requests
import torch
from typing import Dict, List, Optional, Any
from PIL import Image

from sam2.features.clip_main import CLIPfeatures
from sam2.features.dino_main import DINOfeatures
from sam2.features.sam2_main import SAM2features

logger = logging.getLogger(__name__)


class BaseService:
    """Base service class with common functionality"""

    def __init__(self):
        self.models = {}  # Cache for model instances
        self.request_counter = 0
        self.gpu_count = torch.cuda.device_count()
        self.worker_id = os.getpid() if self.gpu_count > 0 else 0

    def _get_image(self, image=None, image_url=None):
        """Load image from base64 or URL and return PIL Image"""
        if image:
            if image.startswith('data:image'):
                image = image.split(',')[1]
            image_data = base64.b64decode(image)
            return Image.open(io.BytesIO(image_data))

        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))

        raise ValueError("Either image or image_url must be provided")

    def _get_random_gpu_id(self):
        """Get random GPU ID for load balancing"""
        if self.gpu_count > 0:
            return random.randint(0, self.gpu_count - 1)
        return None

    def _encode_tensor(self, tensor):
        """Encode tensor to base64"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        tensor_bytes = tensor.astype(np.float32).tobytes()
        return base64.b64encode(tensor_bytes).decode('utf-8')

    def get_health(self):
        """Get service health status"""
        total_models = sum(len(gpu_models) for gpu_models in self.models.values())
        model_keys = list(self.models.keys())

        if self.gpu_count > 0:
            gpu_id = 0
            device = f"cuda:{gpu_id}"
        else:
            gpu_id = None
            device = "cpu"

        return {
            'status': 'healthy',
            'device': device,
            'worker_id': self.worker_id,
            'gpu_id': gpu_id,
            'gpu_count': self.gpu_count,
            'models_loaded': model_keys,
            'total_model_instances': total_models,
            'request_counter': self.request_counter
        }


class CLIPService(BaseService):
    """Service for CLIP feature extraction"""

    def _get_model_key(self, model_name, model_pretrained):
        return f"clip_{model_name}_{model_pretrained}"

    def _get_model(self, model_name, model_pretrained, gpu_id):
        key = self._get_model_key(model_name, model_pretrained)

        if key not in self.models:
            self.models[key] = {}

        if gpu_id not in self.models[key]:
            device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
            self.models[key][gpu_id] = CLIPfeatures(
                model_name=model_name,
                model_pretrained=model_pretrained,
                device=device
            )

        return self.models[key][gpu_id]

    def extract_features(self, image=None, image_url=None, model_name="ViT-L-14-336-quickgelu",
                         model_pretrained="openai", **kwargs):
        """Extract CLIP features from image"""
        self.request_counter += 1

        pil_img = self._get_image(image, image_url)
        gpu_id = self._get_random_gpu_id()
        model = self._get_model(model_name, model_pretrained, gpu_id)

        # Extract features
        features = model.extract(pil_img, **kwargs)

        # Clear GPU cache
        if gpu_id is not None:
            torch.cuda.empty_cache()

        return {
            'features': self._encode_tensor(features),
            'shape': list(features.shape),
            'is_pca': kwargs.get('ret_pca', False)
        }

    def encode_text(self, text, model_name="ViT-L-14-336-quickgelu", model_pretrained="openai"):
        """Encode text to embeddings"""
        self.request_counter += 1

        gpu_id = self._get_random_gpu_id()
        model = self._get_model(model_name, model_pretrained, gpu_id)

        text_emb = model.encode_text(text)

        # Clear GPU cache
        if gpu_id is not None:
            torch.cuda.empty_cache()

        return {
            'text_embedding': self._encode_tensor(text_emb),
            'shape': list(text_emb.shape)
        }

    def compute_similarity(self, image=None, image_url=None, text="", model_name="ViT-L-14-336-quickgelu",
                           model_pretrained="openai", softmax=1.0, **kwargs):
        """Compute text-image similarity map"""
        self.request_counter += 1

        pil_img = self._get_image(image, image_url)
        gpu_id = self._get_random_gpu_id()
        model = self._get_model(model_name, model_pretrained, gpu_id)

        # Extract features
        features = model.extract(pil_img, **kwargs)

        # Encode text
        text_emb = model.encode_text(text)

        # Compute similarity
        sim_map = model.compute_similarity(features, text_emb, softmax=softmax)

        # Clear GPU cache
        if gpu_id is not None:
            torch.cuda.empty_cache()

        return {
            'similarity_map': self._encode_tensor(sim_map),
            'shape': list(sim_map.shape)
        }


class DINOService(BaseService):
    """Service for DINO feature extraction"""

    def _get_model_key(self, model_type, stride):
        return f"dino_{model_type}_{stride}"

    def _get_model(self, model_type, stride, gpu_id):
        key = self._get_model_key(model_type, stride)

        if key not in self.models:
            self.models[key] = {}

        if gpu_id not in self.models[key]:
            device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
            self.models[key][gpu_id] = DINOfeatures(
                model_type=model_type,
                stride=stride,
                device=device
            )

        return self.models[key][gpu_id]

    def extract_features(self, image=None, image_url=None, model_type="dinov2_vitl14",
                         stride=None, **kwargs):
        """Extract DINO features from image"""
        self.request_counter += 1

        pil_img = self._get_image(image, image_url)
        gpu_id = self._get_random_gpu_id()
        model = self._get_model(model_type, stride, gpu_id)

        # Extract features
        features = model.extract(pil_img, **kwargs)

        # Clear GPU cache
        if gpu_id is not None:
            torch.cuda.empty_cache()

        return {
            'features': self._encode_tensor(features),
            'shape': list(features.shape),
            'is_pca': kwargs.get('ret_pca', False)
        }


class SAM2Service(BaseService):
    """Service for SAM2 segmentation"""

    def _get_model_key(self, model_cfg, checkpoint_path):
        """Intelligent caching - both params point to same SAM2.1 large model"""
        # For SAM2, both config and checkpoint typically refer to the same model
        # Extract meaningful identifier from paths
        cfg_name = model_cfg.split('/')[-1].replace('.yaml', '') if model_cfg else 'default'
        ckpt_name = checkpoint_path.split('/')[-1].replace('.pt', '') if checkpoint_path else 'default'

        # For SAM2.1, both typically resolve to the same model, so use a unified key
        if 'sam2.1_hiera_l' in cfg_name or 'sam2.1_hiera_large' in ckpt_name:
            return "sam2_hiera_large"
        elif 'sam2.1_hiera_b' in cfg_name or 'sam2.1_hiera_base' in ckpt_name:
            return "sam2_hiera_base"
        elif 'sam2.1_hiera_s' in cfg_name or 'sam2.1_hiera_small' in ckpt_name:
            return "sam2_hiera_small"
        elif 'sam2.1_hiera_t' in cfg_name or 'sam2.1_hiera_tiny' in ckpt_name:
            return "sam2_hiera_tiny"
        else:
            # Fallback for unknown models
            return f"sam2_{cfg_name}_{ckpt_name}"

    def _get_model(self, model_cfg, checkpoint_path, gpu_id):
        key = self._get_model_key(model_cfg, checkpoint_path)

        if key not in self.models:
            self.models[key] = {}

        if gpu_id not in self.models[key]:
            device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
            self.models[key][gpu_id] = SAM2features(
                model_cfg=model_cfg,
                checkpoint_path=checkpoint_path,
                device=device
            )

        return self.models[key][gpu_id]

    def auto_mask(self, image=None, image_url=None, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                  checkpoint_path="checkpoints/sam2.1_hiera_large.pt", **kwargs):
        """Generate automatic masks"""
        self.request_counter += 1

        pil_img = self._get_image(image, image_url)
        gpu_id = self._get_random_gpu_id()
        model = self._get_model(model_cfg, checkpoint_path, gpu_id)

        # Convert point_grids to numpy arrays if provided
        if 'point_grids' in kwargs and kwargs['point_grids'] is not None:
            kwargs['point_grids'] = [np.array(grid) for grid in kwargs['point_grids']]

        # Generate masks
        masks = model.auto_mask(pil_img, **kwargs)

        # Clear GPU cache
        if gpu_id is not None:
            torch.cuda.empty_cache()

        # For auto_mask, masks is a list of dictionaries with metadata
        # Each mask dict contains: segmentation, area, bbox, predicted_iou, etc.
        # We need to encode the 'segmentation' field (the actual mask array)
        # and ensure all numpy arrays are converted to lists for JSON serialization
        encoded_masks = []
        for mask_dict in masks:
            # Create a copy of the mask dict
            encoded_mask = {}

            # Handle each field appropriately
            for key, value in mask_dict.items():
                if key == 'segmentation':
                    # Encode the segmentation mask
                    encoded_mask['segmentation'] = self._encode_tensor(value)
                    encoded_mask['segmentation_shape'] = list(value.shape)
                elif isinstance(value, np.ndarray):
                    # Convert numpy arrays to lists
                    encoded_mask[key] = value.tolist()
                elif isinstance(value, np.integer):
                    # Convert numpy integers to Python int
                    encoded_mask[key] = int(value)
                elif isinstance(value, np.floating):
                    # Convert numpy floats to Python float
                    encoded_mask[key] = float(value)
                else:
                    # Keep other types as-is
                    encoded_mask[key] = value

            encoded_masks.append(encoded_mask)

        return {
            'masks': encoded_masks,
            'num_masks': len(masks)
        }

    def prompt_mask(self, image=None, image_url=None, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                    checkpoint_path="checkpoints/sam2.1_hiera_large.pt", point_coords=None,
                    point_labels=None, box=None, mask_input=None, mask_input_shape=None, **kwargs):
        """Generate prompt-based masks"""
        self.request_counter += 1

        pil_img = self._get_image(image, image_url)
        gpu_id = self._get_random_gpu_id()
        model = self._get_model(model_cfg, checkpoint_path, gpu_id)

        # Convert inputs to numpy arrays if provided
        if point_coords is not None:
            point_coords = np.array(point_coords)
        if point_labels is not None:
            point_labels = np.array(point_labels)
        if box is not None:
            box = np.array(box)
        if mask_input is not None:
            # Decode base64 mask input and reshape properly
            mask_bytes = base64.b64decode(mask_input)
            mask_input = np.frombuffer(mask_bytes, dtype=np.float32)
            if mask_input_shape is not None:
                mask_input = mask_input.reshape(mask_input_shape)
            else:
                # Fallback: assume it's a 3D tensor (C, H, W) - most common case
                total_elements = mask_input.size
                # Try to infer shape: if it's divisible by 3, assume (3, H, W)
                if total_elements % 3 == 0:
                    h = int(np.sqrt(total_elements // 3))
                    if h * h * 3 == total_elements:
                        mask_input = mask_input.reshape(3, h, h)
                    else:
                        # If not a perfect square, try (1, H, W) for single mask
                        h = int(np.sqrt(total_elements))
                        if h * h == total_elements:
                            mask_input = mask_input.reshape(1, h, h)
                        else:
                            raise ValueError(f"Cannot infer mask_input shape from {total_elements} elements. Please provide mask_input_shape.")
                else:
                    # Try (1, H, W) for single mask
                    h = int(np.sqrt(total_elements))
                    if h * h == total_elements:
                        mask_input = mask_input.reshape(1, h, h)
                    else:
                        raise ValueError(f"Cannot infer mask_input shape from {total_elements} elements. Please provide mask_input_shape.")

        # Generate masks
        masks, iou_predictions, low_res_masks = model.prompt_mask(
            pil_img,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            **kwargs
        )

        # Clear GPU cache
        if gpu_id is not None:
            torch.cuda.empty_cache()

        return {
            'masks': self._encode_tensor(masks),
            'masks_shape': list(masks.shape),
            'iou_predictions': self._encode_tensor(iou_predictions),
            'iou_shape': list(iou_predictions.shape),
            'low_res_masks': self._encode_tensor(low_res_masks),
            'low_res_shape': list(low_res_masks.shape)
        }
