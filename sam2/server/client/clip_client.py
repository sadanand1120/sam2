import base64
import io
import numpy as np
import requests
from typing import Dict, Optional, Union
from PIL import Image
import yaml
import os
import matplotlib.pyplot as plt


def encode_image(image: Union[str, Image.Image]) -> str:
    """Encode local image file path or PIL Image to base64"""
    if isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Image must be a file path (str) or a PIL.Image.Image object.")


def decode_features(features_base64: str, shape: list) -> np.ndarray:
    """Decode base64 features to numpy array"""
    features_bytes = base64.b64decode(features_base64)
    features = np.frombuffer(features_bytes, dtype=np.float32).reshape(shape)
    return features


def decode_text_embedding(embedding_base64: str, shape: list) -> np.ndarray:
    """Decode base64 text embedding to numpy array"""
    embedding_bytes = base64.b64decode(embedding_base64)
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(shape)
    return embedding


def decode_similarity_map(similarity_base64: str, shape: list) -> np.ndarray:
    """Decode base64 similarity map to numpy array"""
    similarity_bytes = base64.b64decode(similarity_base64)
    similarity = np.frombuffer(similarity_bytes, dtype=np.float32).reshape(shape)
    return similarity


def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Get headers with optional API key"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_clip_features(image: Optional[Union[str, Image.Image]] = None, image_url: Optional[str] = None,
                          base_url: str = "",
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
                          api_key: str = "") -> Dict:
    """Extract CLIP features from image"""
    payload = {
        'model_name': model_name,
        'model_pretrained': model_pretrained,
        'ret_pca': ret_pca,
        'ret_patches': ret_patches,
        'load_size': load_size,
        'center_crop': center_crop,
        'pca_niter': pca_niter,
        'pca_q_min': pca_q_min,
        'pca_q_max': pca_q_max,
        'interpolation_mode': interpolation_mode,
        'tensor_format': tensor_format,
        'padding_mode': padding_mode
    }

    if image:
        payload['image'] = encode_image(image)
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")

    response = requests.post(
        f"{base_url}/extract",
        json=payload,
        headers=_get_headers(api_key)
    )
    response.raise_for_status()
    return response.json()


def encode_clip_text(text: str,
                     base_url: str = "",
                     model_name: str = "ViT-L-14-336-quickgelu",
                     model_pretrained: str = "openai",
                     api_key: str = "") -> Dict:
    """Encode text using CLIP"""
    payload = {
        'text': text,
        'model_name': model_name,
        'model_pretrained': model_pretrained
    }

    response = requests.post(
        f"{base_url}/encode_text",
        json=payload,
        headers=_get_headers(api_key)
    )
    response.raise_for_status()
    return response.json()


def compute_clip_similarity(image: Optional[Union[str, Image.Image]] = None, image_url: Optional[str] = None,
                            text: str = "",
                            base_url: str = "",
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
                            softmax: float = 1.0,
                            api_key: str = "") -> Dict:
    """Compute CLIP image-text similarity"""
    payload = {
        'text': text,
        'model_name': model_name,
        'model_pretrained': model_pretrained,
        'ret_pca': ret_pca,
        'ret_patches': ret_patches,
        'load_size': load_size,
        'center_crop': center_crop,
        'pca_niter': pca_niter,
        'pca_q_min': pca_q_min,
        'pca_q_max': pca_q_max,
        'interpolation_mode': interpolation_mode,
        'tensor_format': tensor_format,
        'padding_mode': padding_mode,
        'softmax': softmax
    }

    if image:
        payload['image'] = encode_image(image)
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")

    response = requests.post(
        f"{base_url}/similarity",
        json=payload,
        headers=_get_headers(api_key)
    )
    response.raise_for_status()
    return response.json()


def get_clip_health(base_url: str = "") -> Dict:
    """Get CLIP server health status (no authentication required)"""
    headers = {"Content-Type": "application/json"}
    response = requests.get(f"{base_url}/health", headers=headers)
    response.raise_for_status()
    return response.json()


# Convenience functions that combine request + decode
def extract_clip_features_decoded(image: Union[str, Image.Image] = None, **kwargs) -> np.ndarray:
    """Extract CLIP features and return decoded numpy array"""
    if image:
        kwargs['image'] = image

    result = extract_clip_features(**kwargs)
    return decode_features(result['features'], result['shape'])


def encode_clip_text_decoded(text: str, **kwargs) -> np.ndarray:
    """Encode text and return decoded numpy array"""
    result = encode_clip_text(text, **kwargs)
    return decode_text_embedding(result['text_embedding'], result['shape'])


def compute_clip_similarity_decoded(image: Union[str, Image.Image] = None, **kwargs) -> np.ndarray:
    """Compute similarity and return decoded numpy array"""
    if image:
        kwargs['image'] = image

    result = compute_clip_similarity(**kwargs)
    return decode_similarity_map(result['similarity_map'], result['shape'])


if __name__ == "__main__":
    # Load server config
    config_path = "sam2/server/client/servers.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_url = config["clip"]["base_url"]
    api_key = config["clip"]["api_key"]

    test_image_path = "notebooks/images/cars.jpg"
    print("=== CLIP Client Demo ===")

    # Health check (no authentication required)
    health = get_clip_health(base_url=base_url)
    print(f"Server health: {health}")

    # Extract features with PCA (exactly matching main file)
    img = Image.open(test_image_path)
    print(f"Input image size: h={img.height}, w={img.width}")

    desc_pca = extract_clip_features_decoded(
        test_image_path,
        base_url=base_url,
        api_key=api_key,
        ret_pca=True,
        ret_patches=False,
        load_size=2048
    )
    print(f"Output shape: {desc_pca.shape}")
    plt.imshow(desc_pca)
    plt.title("CLIP Patch PCA Visualization")
    plt.axis("off")
    plt.show()

    text_emb = encode_clip_text_decoded(
        "car",
        base_url=base_url,
        api_key=api_key
    )
    print(f"Text embedding shape: {text_emb.shape}")

    sim_map = compute_clip_similarity_decoded(
        test_image_path,
        text="car",
        base_url=base_url,
        api_key=api_key,
        ret_pca=False,
        ret_patches=False,
        load_size=2048,
        softmax=0.25
    )
    plt.imshow(sim_map, cmap="turbo")
    plt.title("Similarity to 'car'")
    plt.axis("off")
    plt.show()

    print("✅ CLIP client demo completed successfully!")
