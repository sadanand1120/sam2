import base64
import numpy as np
import requests
from typing import Dict, Optional
from PIL import Image
import yaml
import os
import matplotlib.pyplot as plt


def encode_image(image_path: str) -> str:
    """Encode local image file to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def decode_features(features_base64: str, shape: list) -> np.ndarray:
    """Decode base64 features to numpy array"""
    features_bytes = base64.b64decode(features_base64)
    features = np.frombuffer(features_bytes, dtype=np.float32).reshape(shape)
    return features


def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Get headers with optional API key"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_dino_features(image: Optional[str] = None, image_url: Optional[str] = None,
                          base_url: str = "",
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
                          padding_mode: str = "constant",
                          api_key: str = "") -> Dict:
    """Extract DINO features from image"""
    payload = {
        'model_type': model_type,
        'stride': stride,
        'ret_pca': ret_pca,
        'ret_patches': ret_patches,
        'load_size': load_size,
        'facet': facet,
        'layer': layer,
        'bin': bin,
        'pca_niter': pca_niter,
        'pca_q_min': pca_q_min,
        'pca_q_max': pca_q_max,
        'interpolation_mode': interpolation_mode,
        'tensor_format': tensor_format,
        'padding_mode': padding_mode
    }

    if image:
        payload['image'] = image
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


def get_dino_health(base_url: str = "") -> Dict:
    """Get DINO server health status (no authentication required)"""
    headers = {"Content-Type": "application/json"}
    response = requests.get(f"{base_url}/health", headers=headers)
    response.raise_for_status()
    return response.json()


# Convenience function that combines request + decode
def extract_dino_features_decoded(image_path: str = None, **kwargs) -> np.ndarray:
    """Extract DINO features and return decoded numpy array"""
    if image_path:
        image_base64 = encode_image(image_path)
        kwargs['image'] = image_base64

    result = extract_dino_features(**kwargs)
    return decode_features(result['features'], result['shape'])


if __name__ == "__main__":
    # Load server config
    config_path = "sam2/server/client/servers.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_url = config["dino"]["base_url"]
    api_key = config["dino"]["api_key"]

    test_image_path = "notebooks/images/cars.jpg"
    print("=== DINO Client Demo ===")

    # Health check (no authentication required)
    health = get_dino_health(base_url=base_url)
    print(f"Server health: {health}")

    # Extract features with PCA (exactly matching main file)
    img = Image.open(test_image_path)
    print(f"Input image size: h={img.height}, w={img.width}")

    desc_pca = extract_dino_features_decoded(
        test_image_path,
        base_url=base_url,
        api_key=api_key,
        ret_pca=True,
        ret_patches=False,
        load_size=2048
    )
    print(f"Output shape: {desc_pca.shape}")
    plt.imshow(desc_pca)
    plt.title("DINO Patch PCA Visualization")
    plt.axis("off")
    plt.show()

    print("✅ DINO client demo completed successfully!")
