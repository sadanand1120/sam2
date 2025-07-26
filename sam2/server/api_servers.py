import asyncio
import logging
import time
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from sam2.server.models import *
from sam2.server.services import CLIPService, DINOService, SAM2Service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common settings
# Individual semaphores per service for fine-grained control
CLIP_MAX_CONCURRENT_REQUESTS = 1  # Single threaded for CLIP (avoid PCA threading issues)
DINO_MAX_CONCURRENT_REQUESTS = 1  # Single threaded for DINO (avoid hooks threading issues)
SAM2_MAX_CONCURRENT_REQUESTS = 4  # Multi-threaded for SAM2 (thread-safe)

clip_request_semaphore = asyncio.Semaphore(CLIP_MAX_CONCURRENT_REQUESTS)
dino_request_semaphore = asyncio.Semaphore(DINO_MAX_CONCURRENT_REQUESTS)
sam2_request_semaphore = asyncio.Semaphore(SAM2_MAX_CONCURRENT_REQUESTS)

security = HTTPBearer(auto_error=False)

# Global API key - ALWAYS required for authentication
API_KEY = "smfeats"  # Default API key


def set_api_key(api_key: str):
    """Set the global API key"""
    global API_KEY
    API_KEY = api_key


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key - ALWAYS required"""
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")

    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


def create_logging_middleware(app_name: str):
    """Create logging middleware for the app"""
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        logger.info(f"[{app_name}] {client_ip} - \"{request.method} {request.url.path}\" - Processing")
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"[{app_name}] {client_ip} - \"{request.method} {request.url.path}\" {response.status_code} - {process_time:.3f}s")

        return response
    return log_requests


def add_common_middleware(app: FastAPI, app_name: str):
    """Add common middleware to FastAPI app"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        return await create_logging_middleware(app_name)(request, call_next)


# ============= CLIP Server =============
def create_clip_app():
    """Create CLIP FastAPI application"""
    app = FastAPI(title="SAM2 CLIP Features API", version="1.0.0")
    add_common_middleware(app, "CLIP")

    clip_service = CLIPService()

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return clip_service.get_health()

    @app.post("/extract", response_model=CLIPExtractResponse)
    async def extract_features(request: CLIPExtractRequest, _: bool = Depends(verify_api_key)):
        try:
            async with clip_request_semaphore:
                kwargs = dict(
                    ret_pca=request.ret_pca,
                    ret_patches=request.ret_patches,
                    load_size=request.load_size,
                    center_crop=request.center_crop,
                    pca_niter=request.pca_niter,
                    pca_q_min=request.pca_q_min,
                    pca_q_max=request.pca_q_max,
                    interpolation_mode=request.interpolation_mode,
                    tensor_format=request.tensor_format,
                    padding_mode=request.padding_mode
                )
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: clip_service.extract_features(
                        request.image,
                        request.image_url,
                        request.model_name,
                        request.model_pretrained,
                        **kwargs
                    )
                )
            return CLIPExtractResponse(**result)
        except Exception as e:
            logger.error(f"Error in CLIP extract endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @app.post("/encode_text", response_model=CLIPTextResponse)
    async def encode_text(request: CLIPTextRequest, _: bool = Depends(verify_api_key)):
        try:
            async with clip_request_semaphore:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    clip_service.encode_text,
                    request.text,
                    request.model_name,
                    request.model_pretrained
                )
            return CLIPTextResponse(**result)
        except Exception as e:
            logger.error(f"Error in CLIP encode_text endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @app.post("/similarity", response_model=CLIPSimilarityResponse)
    async def compute_similarity(request: CLIPSimilarityRequest, _: bool = Depends(verify_api_key)):
        try:
            async with clip_request_semaphore:
                kwargs = dict(
                    ret_pca=request.ret_pca,
                    ret_patches=request.ret_patches,
                    load_size=request.load_size,
                    center_crop=request.center_crop,
                    pca_niter=request.pca_niter,
                    pca_q_min=request.pca_q_min,
                    pca_q_max=request.pca_q_max,
                    interpolation_mode=request.interpolation_mode,
                    tensor_format=request.tensor_format,
                    padding_mode=request.padding_mode
                )
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: clip_service.compute_similarity(
                        request.image,
                        request.image_url,
                        request.text,
                        request.model_name,
                        request.model_pretrained,
                        request.softmax,
                        **kwargs
                    )
                )
            return CLIPSimilarityResponse(**result)
        except Exception as e:
            logger.error(f"Error in CLIP similarity endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return app


# ============= DINO Server =============
def create_dino_app():
    """Create DINO FastAPI application"""
    app = FastAPI(title="SAM2 DINO Features API", version="1.0.0")
    add_common_middleware(app, "DINO")

    dino_service = DINOService()

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return dino_service.get_health()

    @app.post("/extract", response_model=DINOExtractResponse)
    async def extract_features(request: DINOExtractRequest, _: bool = Depends(verify_api_key)):
        try:
            async with dino_request_semaphore:
                kwargs = dict(
                    ret_pca=request.ret_pca,
                    ret_patches=request.ret_patches,
                    load_size=request.load_size,
                    facet=request.facet,
                    layer=request.layer,
                    bin=request.bin,
                    pca_niter=request.pca_niter,
                    pca_q_min=request.pca_q_min,
                    pca_q_max=request.pca_q_max,
                    interpolation_mode=request.interpolation_mode,
                    tensor_format=request.tensor_format,
                    padding_mode=request.padding_mode
                )
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: dino_service.extract_features(
                        request.image,
                        request.image_url,
                        request.model_type,
                        request.stride,
                        **kwargs
                    )
                )
            return DINOExtractResponse(**result)
        except Exception as e:
            logger.error(f"Error in DINO extract endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return app


# ============= SAM2 Server =============
def create_sam2_app():
    """Create SAM2 FastAPI application"""
    app = FastAPI(title="SAM2 Segmentation API", version="1.0.0")
    add_common_middleware(app, "SAM2")

    sam2_service = SAM2Service()

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return sam2_service.get_health()

    @app.post("/auto_mask", response_model=SAM2AutoMaskResponse)
    async def auto_mask(request: SAM2AutoMaskRequest, _: bool = Depends(verify_api_key)):
        try:
            # Extract auto mask generation parameters
            auto_mask_params = {}
            if request.preset is not None:
                auto_mask_params['preset'] = request.preset

            # Add all optional parameters that are not None
            param_mapping = {
                'points_per_side': request.points_per_side,
                'points_per_batch': request.points_per_batch,
                'pred_iou_thresh': request.pred_iou_thresh,
                'stability_score_thresh': request.stability_score_thresh,
                'stability_score_offset': request.stability_score_offset,
                'mask_threshold': request.mask_threshold,
                'box_nms_thresh': request.box_nms_thresh,
                'crop_n_layers': request.crop_n_layers,
                'crop_nms_thresh': request.crop_nms_thresh,
                'crop_overlap_ratio': request.crop_overlap_ratio,
                'crop_n_points_downscale_factor': request.crop_n_points_downscale_factor,
                'point_grids': request.point_grids,
                'min_mask_region_area': request.min_mask_region_area,
                'use_m2m': request.use_m2m,
                'multimask_output': request.multimask_output,
                'output_mode': request.output_mode
            }

            for key, value in param_mapping.items():
                if value is not None:
                    auto_mask_params[key] = value

            async with sam2_request_semaphore:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: sam2_service.auto_mask(
                        request.image,
                        request.image_url,
                        request.model_cfg,
                        request.checkpoint_path,
                        **auto_mask_params
                    )
                )
            return SAM2AutoMaskResponse(**result)
        except Exception as e:
            logger.error(f"Error in SAM2 auto_mask endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @app.post("/prompt_mask", response_model=SAM2PromptMaskResponse)
    async def prompt_mask(request: SAM2PromptMaskRequest, _: bool = Depends(verify_api_key)):
        try:
            # Extract prompt parameters
            prompt_params = {
                'multimask_output': request.multimask_output,
                'return_logits': request.return_logits,
                'normalize_coords': request.normalize_coords,
                'mask_threshold': request.mask_threshold,
                'max_hole_area': request.max_hole_area,
                'max_sprinkle_area': request.max_sprinkle_area
            }

            async with sam2_request_semaphore:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: sam2_service.prompt_mask(
                        request.image,
                        request.image_url,
                        request.model_cfg,
                        request.checkpoint_path,
                        request.point_coords,
                        request.point_labels,
                        request.box,
                        request.mask_input,
                        request.mask_input_shape,
                        **prompt_params
                    )
                )
            return SAM2PromptMaskResponse(**result)
        except Exception as e:
            logger.error(f"Error in SAM2 prompt_mask endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return app


# App factory functions for use in server runners
clip_app = create_clip_app()
dino_app = create_dino_app()
sam2_app = create_sam2_app()
