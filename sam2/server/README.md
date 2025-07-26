# SAM2 Features API Servers

Production-ready REST API servers for CLIP, DINO, and SAM2 features extraction and segmentation with OpenAI-style authentication.

## Overview

This directory provides three independent REST API servers for the SAM2 features ecosystem:

- **CLIP Server** (Port 8096): Visual-text feature extraction with similarity computation
- **DINO Server** (Port 8097): Self-supervised visual feature extraction  
- **SAM2 Server** (Port 8098): Advanced image segmentation with automatic and prompt-based masking

All servers feature:
- ✅ **Multi-GPU Load Balancing** with intelligent model caching
- ✅ **OpenAI-style Authentication** with Bearer token API keys
- ✅ **Concurrent Request Handling** with semaphore-based limiting (4 concurrent)
- ✅ **Base64 Image Support** and URL loading
- ✅ **Comprehensive Parameter Exposure** for all model options
- ✅ **Production-Ready Deployment** with uvicorn and multiple workers

## Quick Start

### 1. Install Dependencies

```bash
# Server dependencies
pip install -r sam2/server/requirements_server.txt

# Client dependencies (for testing)
pip install -r sam2/server/requirements_client.txt
```

### 2. Start Servers

```bash
# Start CLIP server (port 8096)
python -m sam2.server.run_clip_server --port 8096 --api-key "your-clip-key"

# Start DINO server (port 8097)  
python -m sam2.server.run_dino_server --port 8097 --api-key "your-dino-key"

# Start SAM2 server (port 8098)
python -m sam2.server.run_sam2_server --port 8098 --api-key "your-sam2-key"
```

### 3. Test with Clients

```python
from sam2.server.client.clip_client import extract_clip_features, encode_image

# Extract CLIP features
image_b64 = encode_image("path/to/image.jpg")
result = extract_clip_features(
    image=image_b64,
    base_url="http://localhost:8096",
    api_key="your-clip-key"
)
print(f"CLIP features shape: {result['shape']}")
```

## Server Architecture

### Intelligent Model Caching

Each server implements smart caching strategies:

- **CLIP**: Cache by `model_name` + `model_pretrained` 
- **DINO**: Cache by `model_type` + `stride`
- **SAM2**: **Intelligent caching** - recognizes that different config/checkpoint combinations often point to the same model (e.g., SAM2.1 Large)

### Multi-GPU Support

Automatic GPU detection and round-robin load balancing:
```bash
# Set visible GPUs before starting
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m sam2.server.run_clip_server --port 8096
```

Models are cached per-GPU for optimal memory usage and performance.

### Authentication

**All servers require authentication by default** with the API key "smfeats":
```bash
# Authentication is always enabled
python -m sam2.server.run_clip_server --api-key "smfeats"
```

## API Documentation

### CLIP Server (Port 8096)

#### POST `/extract` - Extract CLIP Features
```json
{
  "image": "base64_encoded_image",
  "model_name": "ViT-L-14-336-quickgelu",
  "model_pretrained": "openai",
  "ret_pca": false,
  "ret_patches": true,
  "load_size": 1024,
  "center_crop": false,
  "pca_niter": 5,
  "pca_q_min": 0.01,
  "pca_q_max": 0.99,
  "interpolation_mode": "bilinear",
  "tensor_format": "HWC",
  "padding_mode": "constant"
}
```

#### POST `/encode_text` - Encode Text
```json
{
  "text": "a photo of a car",
  "model_name": "ViT-L-14-336-quickgelu",
  "model_pretrained": "openai"
}
```

#### POST `/similarity` - Compute Image-Text Similarity
```json
{
  "image": "base64_encoded_image",
  "text": "a photo of a car",
  "model_name": "ViT-L-14-336-quickgelu",
  "model_pretrained": "openai",
  "softmax": 1.0,
  "ret_pca": false,
  "ret_patches": true
}
```

### DINO Server (Port 8097)

#### POST `/extract` - Extract DINO Features
```json
{
  "image": "base64_encoded_image",
  "model_type": "dinov2_vitl14",
  "stride": null,
  "ret_pca": false,
  "ret_patches": true,
  "load_size": -1,
  "facet": "token",
  "layer": -1,
  "bin": false,
  "pca_niter": 5,
  "pca_q_min": 0.01,
  "pca_q_max": 0.99,
  "interpolation_mode": "bilinear",
  "tensor_format": "HWC",
  "padding_mode": "constant"
}
```

### SAM2 Server (Port 8098)

#### POST `/auto_mask` - Automatic Mask Generation
```json
{
  "image": "base64_encoded_image",
  "preset": "fine_grained",
  "points_per_side": 64,
  "points_per_batch": 128,
  "pred_iou_thresh": 0.7,
  "stability_score_thresh": 0.92,
  "stability_score_offset": 0.7,
  "mask_threshold": 0.0,
  "box_nms_thresh": 0.7,
  "crop_n_layers": 1,
  "crop_nms_thresh": 0.7,
  "crop_overlap_ratio": 0.3413333333333333,
  "crop_n_points_downscale_factor": 2,
  "point_grids": null,
  "min_mask_region_area": 25.0,
  "use_m2m": true,
  "multimask_output": true,
  "output_mode": "binary_mask"
}
```

#### POST `/prompt_mask` - Prompt-Based Mask Generation
```json
{
  "image": "base64_encoded_image",
  "point_coords": [[520, 820], [1090, 750]],
  "point_labels": [1, 1],
  "box": [280, 440, 1474, 1225],
  "mask_input": "base64_encoded_low_res_mask",
  "mask_input_shape": [1, 256, 256],
  "multimask_output": true,
  "return_logits": false,
  "normalize_coords": true,
  "mask_threshold": 0.0,
  "max_hole_area": 0.0,
  "max_sprinkle_area": 0.0
}
```

**Note**: SAM2 prompt mask supports **any combination** of points, boxes, and masks. All are optional.

### Health Endpoints

All servers provide `GET /health` endpoints (no authentication required):
```json
{
  "status": "healthy",
  "device": "cuda:0", 
  "worker_id": 12345,
  "gpu_id": 0,
  "gpu_count": 4,
  "models_loaded": ["clip_ViT-L-14-336-quickgelu_openai"],
  "total_model_instances": 1,
  "request_counter": 42
}
```

## Client Libraries

### CLIP Client

```python
from sam2.server.client.clip_client import *

# Extract features
image_b64 = encode_image("image.jpg")
features = extract_clip_features(image=image_b64, ret_pca=True)
features_np = decode_features(features['features'], features['shape'])

# Encode text
text_emb = encode_clip_text("a photo of a car")
text_np = decode_text_embedding(text_emb['text_embedding'], text_emb['shape'])

# Compute similarity
sim_map = compute_clip_similarity(image=image_b64, text="car", softmax=0.25)
sim_np = decode_similarity_map(sim_map['similarity_map'], sim_map['shape'])

# Convenience functions (auto encode/decode)
features_np = extract_clip_features_decoded("image.jpg", ret_pca=True)
text_np = encode_clip_text_decoded("a photo of a car")
sim_np = compute_clip_similarity_decoded("image.jpg", text="car")
```

### DINO Client

```python
from sam2.server.client.dino_client import *

# Extract features
image_b64 = encode_image("image.jpg")
features = extract_dino_features(
    image=image_b64, 
    model_type="dinov2_vitl14",
    facet="token",
    layer=-1,
    ret_pca=True
)
features_np = decode_features(features['features'], features['shape'])

# Convenience function
features_np = extract_dino_features_decoded(
    "image.jpg", 
    model_type="dinov2_vitl14",
    ret_pca=True
)
```

### SAM2 Client

```python
from sam2.server.client.sam2_client import *

# Automatic masks
image_b64 = encode_image("image.jpg")
auto_masks = generate_sam2_auto_masks(image=image_b64, preset="fine_grained")
print(f"Generated {auto_masks['num_masks']} masks")

# Prompt masks - points only
masks, iou_scores, low_res = generate_sam2_prompt_masks(
    image=image_b64,
    point_coords=[[520, 820], [1090, 750]], 
    point_labels=[1, 1]
)

# Prompt masks - box only
masks, iou_scores, low_res = generate_sam2_prompt_masks(
    image=image_b64,
    box=[280, 440, 1474, 1225]
)

# Prompt masks - combined points + box
masks, iou_scores, low_res = generate_sam2_prompt_masks(
    image=image_b64,
    point_coords=[[1398, 752], [520, 820]],
    point_labels=[0, 1],  # negative + positive points
    box=[280, 440, 1474, 1225]  
)

# Convenience functions
auto_masks_list = generate_sam2_auto_masks_decoded("image.jpg", preset="coarse")
masks_np, iou_np, low_res_np = generate_sam2_prompt_masks_decoded(
    "image.jpg", 
    point_coords=[[520, 820]], 
    point_labels=[1]
)

# Specialized convenience functions
masks_np, iou_np, low_res_np = sam2_point_mask(
    "image.jpg", 
    point_coords=[[520, 820]], 
    point_labels=[1]
)

masks_np, iou_np, low_res_np = sam2_box_mask(
    "image.jpg",
    box=[280, 440, 1474, 1225]
)

masks_np, iou_np, low_res_np = sam2_combined_mask(
    "image.jpg",
    point_coords=[[520, 820]],
    point_labels=[1], 
    box=[280, 440, 1474, 1225]
)
```

## Advanced Usage

### Server Configuration

```bash
# Production deployment with multiple workers
python -m sam2.server.run_clip_server \
  --host 0.0.0.0 \
  --port 8096 \
  --workers 4 \
  --api-key "production-clip-key" \
  --log-level info

# Development with custom API key
python -m sam2.server.run_dino_server \
  --host localhost \
  --port 8097 \
  --workers 1 \
  --api-key "dev-dino-key" \
  --log-level debug
```

### Configuration Files

Update `sam2/server/client/servers.yaml`:
```yaml
clip:
  base_url: "http://your-server:8096"
  api_key: "your-clip-key"

dino:
  base_url: "http://your-server:8097" 
  api_key: "your-dino-key"

sam2:
  base_url: "http://your-server:8098"
  api_key: "your-sam2-key"
```

### Load Configuration in Client

```python
import yaml

with open("sam2/server/client/servers.yaml") as f:
    config = yaml.safe_load(f)

clip_config = config['clip']
result = extract_clip_features(
    image=image_b64,
    base_url=clip_config['base_url'],
    api_key=clip_config['api_key']
)
```

## Authentication

### Server-Side

```bash
# Enable authentication (always required)
python -m sam2.server.run_clip_server --api-key "secure-api-key"

# Default API key is "smfeats"
python -m sam2.server.run_clip_server  # Uses "smfeats" by default
```

### Client-Side  

```python
# With authentication (always required)
result = extract_clip_features(
    image=image_b64,
    api_key="secure-api-key"
)

# Using default API key
result = extract_clip_features(
    image=image_b64,
    api_key="smfeats"  # Default key
)
```

## Deployment

### Single Server Deployment

```bash
# Start all servers with authentication
python -m sam2.server.run_clip_server --port 8096 --api-key "clip-key" &
python -m sam2.server.run_dino_server --port 8097 --api-key "dino-key" &  
python -m sam2.server.run_sam2_server --port 8098 --api-key "sam2-key" &
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install dependencies
COPY sam2/server/requirements_server.txt /app/
RUN pip install -r /app/requirements_server.txt

# Copy code
COPY sam2/ /app/sam2/
WORKDIR /app

# Expose ports
EXPOSE 8096 8097 8098

# Start servers
CMD python -m sam2.server.run_clip_server --port 8096 & \
    python -m sam2.server.run_dino_server --port 8097 & \
    python -m sam2.server.run_sam2_server --port 8098 & \
    wait
```

### Multi-GPU Production Setup

```bash
# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start with multiple workers per GPU
python -m sam2.server.run_clip_server \
  --host 0.0.0.0 \
  --port 8096 \
  --workers 8 \
  --api-key "production-key" \
  --log-level info
```

## Testing

### Run Comprehensive Test Suite

```bash
# Test all endpoints and functionality
python sam2/server/client/test_api_clients.py
```

### Individual Test Functions

```python
# Test specific functionality
from sam2.server.client.test_api_clients import *

# Test endpoints
test_clip_endpoints()
test_dino_endpoints() 
test_sam2_endpoints()

# Test authentication
test_authentication()

# Test concurrent requests
test_concurrent_requests()

# Test parameter validation
test_parameter_validation()
```

## Interactive Documentation

Each server provides auto-generated interactive documentation:

- **CLIP**: http://localhost:8096/docs  
- **DINO**: http://localhost:8097/docs
- **SAM2**: http://localhost:8098/docs

## File Structure

```
sam2/server/
├── README.md                  # This file
├── requirements_server.txt    # Server dependencies
├── requirements_client.txt    # Client dependencies  
├── models.py                 # Pydantic request/response models
├── services.py               # Service layer with model management
├── api_servers.py            # FastAPI applications
├── run_clip_server.py        # CLIP server runner
├── run_dino_server.py        # DINO server runner
├── run_sam2_server.py        # SAM2 server runner
└── client/
    ├── __init__.py
    ├── servers.yaml          # Client server configurations
    ├── clip_client.py        # CLIP client functions
    ├── dino_client.py        # DINO client functions
    ├── sam2_client.py        # SAM2 client functions
    └── test_api_clients.py   # Comprehensive test suite
```

## Key Features

- **Model Loading**: First request triggers model loading (may take longer)
- **Memory Management**: GPU cache is cleared after each request
- **Error Handling**: Comprehensive error logging and HTTP status codes
- **Concurrent Requests**: Limited by semaphore (4 concurrent per server)
- **Image Formats**: Supports all PIL-compatible formats via base64 or URL
- **Parameter Validation**: Full Pydantic validation for all requests
- **OpenAI Compatibility**: Authentication follows OpenAI API patterns
- **Base64 Encoding**: All tensor data is base64 encoded for JSON serialization

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use fewer workers
2. **Model loading errors**: Check checkpoint paths and model configurations  
3. **Port conflicts**: Ensure ports 8096-8098 are available
4. **Authentication errors**: Verify API key matches between client and server

### Debugging

```bash
# Enable debug logging
python -m sam2.server.run_clip_server --log-level debug

# Check server health
curl http://localhost:8096/health

# Test with default API key
python -m sam2.server.run_clip_server  # Uses "smfeats" by default
```

### Performance Optimization

```bash
# Multi-GPU setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m sam2.server.run_clip_server --workers 8

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

For additional support, check the interactive documentation at `/docs` endpoints. 