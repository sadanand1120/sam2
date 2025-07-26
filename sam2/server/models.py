from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import numpy as np


# ============= Base Models =============
class HealthResponse(BaseModel):
    status: str
    device: str
    worker_id: int = Field(..., description="Worker process ID")
    gpu_id: Optional[int] = Field(None, description="GPU ID in use")
    gpu_count: int = Field(..., description="Total available GPUs")
    models_loaded: List[str] = Field(..., description="Model cache keys loaded")
    total_model_instances: int = Field(..., description="Total cached model instances")
    request_counter: int = Field(..., description="Total requests processed")


# ============= CLIP Models =============
class CLIPExtractRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    model_name: str = Field(default="ViT-L-14-336-quickgelu", description="CLIP model name")
    model_pretrained: str = Field(default="openai", description="CLIP pretrained weights")
    ret_pca: bool = Field(default=False, description="Return 3D PCA visualization, else full D-dim features")
    ret_patches: bool = Field(default=True, description="Return patch grid, else upsample to full image size")
    load_size: Optional[int] = Field(default=1024, description="Target size for resizing (None for no resize, negative for scale smallest side)")
    center_crop: bool = Field(default=False, description="Whether to center crop after resizing")
    pca_niter: int = Field(default=5, description="Number of iterations for PCA computation")
    pca_q_min: float = Field(default=0.01, description="PCA quantile min for normalization")
    pca_q_max: float = Field(default=0.99, description="PCA quantile max for normalization")
    interpolation_mode: str = Field(default="bilinear", description="Interpolation mode for upsampling")
    tensor_format: str = Field(default="HWC", description="Output tensor format")
    padding_mode: str = Field(default="constant", description="Padding mode for pad_to_multiple")


class CLIPTextRequest(BaseModel):
    text: str = Field(..., description="Text to encode")
    model_name: str = Field(default="ViT-L-14-336-quickgelu", description="CLIP model name")
    model_pretrained: str = Field(default="openai", description="CLIP pretrained weights")


class CLIPSimilarityRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    text: str = Field(..., description="Text for similarity computation")
    model_name: str = Field(default="ViT-L-14-336-quickgelu", description="CLIP model name")
    model_pretrained: str = Field(default="openai", description="CLIP pretrained weights")
    ret_pca: bool = Field(default=False, description="Use PCA features for similarity")
    ret_patches: bool = Field(default=True, description="Return patch grid, else upsample to full image size")
    load_size: Optional[int] = Field(default=1024, description="Target size for resizing")
    center_crop: bool = Field(default=False, description="Whether to center crop after resizing")
    pca_niter: int = Field(default=5, description="Number of iterations for PCA computation")
    pca_q_min: float = Field(default=0.01, description="PCA quantile min for normalization")
    pca_q_max: float = Field(default=0.99, description="PCA quantile max for normalization")
    interpolation_mode: str = Field(default="bilinear", description="Interpolation mode for upsampling")
    tensor_format: str = Field(default="HWC", description="Output tensor format")
    padding_mode: str = Field(default="constant", description="Padding mode for pad_to_multiple")
    softmax: float = Field(default=1.0, description="Softmax temperature for similarity")


class CLIPExtractResponse(BaseModel):
    features: str = Field(..., description="Base64 encoded features")
    shape: List[int] = Field(..., description="Feature tensor shape")
    is_pca: bool = Field(..., description="Whether features are PCA-transformed")


class CLIPTextResponse(BaseModel):
    text_embedding: str = Field(..., description="Base64 encoded text embedding")
    shape: List[int] = Field(..., description="Embedding shape")


class CLIPSimilarityResponse(BaseModel):
    similarity_map: str = Field(..., description="Base64 encoded similarity map")
    shape: List[int] = Field(..., description="Similarity map shape")


# ============= DINO Models =============
class DINOExtractRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    model_type: str = Field(default="dinov2_vitl14", description="DINO model type")
    stride: Optional[int] = Field(default=None, description="Stride for ViT extraction")
    ret_pca: bool = Field(default=False, description="Return 3D PCA visualization, else full D-dim features")
    ret_patches: bool = Field(default=True, description="Return patch grid, else upsample to full image size")
    load_size: int = Field(default=-1, description="Target size for resizing")
    facet: str = Field(default="token", description="Feature facet to extract")
    layer: int = Field(default=-1, description="Layer to extract from")
    bin: bool = Field(default=False, description="Whether to apply log binning to descriptors")
    pca_niter: int = Field(default=5, description="Number of iterations for PCA computation")
    pca_q_min: float = Field(default=0.01, description="PCA quantile min for normalization")
    pca_q_max: float = Field(default=0.99, description="PCA quantile max for normalization")
    interpolation_mode: str = Field(default="bilinear", description="Interpolation mode for upsampling")
    tensor_format: str = Field(default="HWC", description="Output tensor format")
    padding_mode: str = Field(default="constant", description="Padding mode for pad_to_multiple")


class DINOExtractResponse(BaseModel):
    features: str = Field(..., description="Base64 encoded features")
    shape: List[int] = Field(..., description="Feature tensor shape")
    is_pca: bool = Field(..., description="Whether features are PCA-transformed")


# ============= SAM2 Models =============
class SAM2AutoMaskRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    model_cfg: str = Field(default="configs/sam2.1/sam2.1_hiera_l.yaml", description="Model configuration file")
    checkpoint_path: str = Field(default="checkpoints/sam2.1_hiera_large.pt", description="Model checkpoint path")
    preset: Optional[str] = Field(None, description="Preset configuration (fine_grained or coarse)")
    # Auto mask generation parameters
    points_per_side: Optional[int] = Field(None, description="Number of points sampled along one side of image")
    points_per_batch: Optional[int] = Field(None, description="Number of points run simultaneously by model")
    pred_iou_thresh: Optional[float] = Field(None, description="Filtering threshold using model's predicted mask quality")
    stability_score_thresh: Optional[float] = Field(None, description="Filtering threshold using mask stability")
    stability_score_offset: Optional[float] = Field(None, description="Amount to shift cutoff for stability score")
    mask_threshold: Optional[float] = Field(None, description="Threshold for binarizing mask logits")
    box_nms_thresh: Optional[float] = Field(None, description="Box IoU cutoff for non-maximal suppression")
    crop_n_layers: Optional[int] = Field(None, description="Number of crop layers")
    crop_nms_thresh: Optional[float] = Field(None, description="Box IoU cutoff for NMS between different crops")
    crop_overlap_ratio: Optional[float] = Field(None, description="Degree to which crops overlap")
    crop_n_points_downscale_factor: Optional[int] = Field(None, description="Points-per-side scaling in crop layers")
    point_grids: Optional[List[List[List[float]]]] = Field(None, description="Explicit grids of points for sampling")
    min_mask_region_area: Optional[float] = Field(None, description="Remove disconnected regions smaller than this")
    use_m2m: Optional[bool] = Field(None, description="Whether to add one step refinement using previous mask predictions")
    multimask_output: Optional[bool] = Field(None, description="Whether to output multimask at each point")
    output_mode: Optional[str] = Field(None, description="Format of returned masks")


class SAM2PromptMaskRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    model_cfg: str = Field(default="configs/sam2.1/sam2.1_hiera_l.yaml", description="Model configuration file")
    checkpoint_path: str = Field(default="checkpoints/sam2.1_hiera_large.pt", description="Model checkpoint path")
    # Prompt parameters - all optional, can use any combination
    point_coords: Optional[List[List[float]]] = Field(None, description="Point coordinates in (X,Y) pixels")
    point_labels: Optional[List[int]] = Field(None, description="Point labels (1=foreground, 0=background)")
    box: Optional[List[float]] = Field(None, description="Box prompt in XYXY format")
    mask_input: Optional[str] = Field(None, description="Base64 encoded low resolution mask input from previous iteration")
    mask_input_shape: Optional[List[int]] = Field(None, description="Shape of mask_input tensor for proper reshaping")
    # Prediction parameters
    multimask_output: bool = Field(default=True, description="If True, return three masks for ambiguous prompts")
    return_logits: bool = Field(default=False, description="If True, return un-thresholded mask logits")
    normalize_coords: bool = Field(default=True, description="If True, normalize coordinates to [0,1] range")
    mask_threshold: float = Field(default=0.0, description="Threshold for binarizing mask logits")
    max_hole_area: float = Field(default=0.0, description="Fill small holes up to this area in masks")
    max_sprinkle_area: float = Field(default=0.0, description="Remove small sprinkles up to this area in masks")


class SAM2AutoMaskResponse(BaseModel):
    masks: List[Dict[str, Any]] = Field(..., description="List of mask dictionaries with metadata")
    num_masks: int = Field(..., description="Total number of masks generated")


class SAM2PromptMaskResponse(BaseModel):
    masks: str = Field(..., description="Base64 encoded masks array")
    masks_shape: List[int] = Field(..., description="Masks array shape")
    iou_predictions: str = Field(..., description="Base64 encoded IoU predictions")
    iou_shape: List[int] = Field(..., description="IoU predictions shape")
    low_res_masks: str = Field(..., description="Base64 encoded low resolution masks for subsequent iterations")
    low_res_shape: List[int] = Field(..., description="Low resolution masks shape")
