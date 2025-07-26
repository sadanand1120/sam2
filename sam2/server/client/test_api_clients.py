#!/usr/bin/env python3
import asyncio
import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import requests
import time
import yaml
import os
from PIL import Image
from tqdm import tqdm

from sam2.server.client.clip_client import *
from sam2.server.client.dino_client import *
from sam2.server.client.sam2_client import *
from sam2.features.utils import SAM2utils


def load_test_image():
    """Load the test image used in client demos"""
    test_image_path = 'notebooks/images/cars.jpg'

    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please ensure the cars.jpg image exists in notebooks/images/")
        return None

    return test_image_path


def test_clip_endpoints():
    """Test CLIP API endpoints exactly matching the client demo"""
    print("🎨 Testing CLIP API endpoints...")

    with open("sam2/server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)

    base_url = servers["clip"]["base_url"]
    api_key = servers["clip"]["api_key"]

    try:
        test_image_path = load_test_image()
        if not test_image_path:
            return False

        img = Image.open(test_image_path)
        print(f"Input image size: h={img.height}, w={img.width}")

        # Test health endpoint (no authentication required)
        health = get_clip_health(base_url)
        print(f"✅ CLIP Health: {health['status']}, GPU count: {health['gpu_count']}")

        # Test extract features with PCA (exactly matching client demo)
        desc_pca = extract_clip_features_decoded(
            test_image_path,
            base_url=base_url,
            api_key=api_key,
            ret_pca=True,
            ret_patches=False,
            load_size=2048
        )
        print(f"✅ CLIP PCA Features: shape {desc_pca.shape}")

        # Test visualization (exactly matching client demo)
        plt.figure(figsize=(8, 6))
        plt.imshow(desc_pca)
        plt.title("CLIP Patch PCA Visualization")
        plt.axis("off")
        plt.show()
        print("✅ CLIP PCA visualization displayed")

        # Test text encoding (exactly matching client demo)
        text_emb = encode_clip_text_decoded(
            "car",
            base_url=base_url,
            api_key=api_key
        )
        print(f"✅ CLIP Text Embedding: shape {text_emb.shape}")

        # Test similarity computation (exactly matching client demo)
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
        print(f"✅ CLIP Similarity Map: shape {sim_map.shape}")

        # Test visualization (exactly matching client demo)
        plt.figure(figsize=(8, 6))
        plt.imshow(sim_map, cmap="turbo")
        plt.title("Similarity to 'car'")
        plt.axis("off")
        plt.show()
        print("✅ CLIP similarity visualization displayed")

        return True

    except Exception as e:
        print(f"❌ CLIP Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dino_endpoints():
    """Test DINO API endpoints exactly matching the client demo"""
    print("\n🦕 Testing DINO API endpoints...")

    with open("sam2/server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)

    base_url = servers["dino"]["base_url"]
    api_key = servers["dino"]["api_key"]

    try:
        test_image_path = load_test_image()
        if not test_image_path:
            return False

        img = Image.open(test_image_path)
        print(f"Input image size: h={img.height}, w={img.width}")

        # Test health endpoint (no authentication required)
        health = get_dino_health(base_url)
        print(f"✅ DINO Health: {health['status']}, GPU count: {health['gpu_count']}")

        # Test extract features with PCA (exactly matching client demo)
        desc_pca = extract_dino_features_decoded(
            test_image_path,
            base_url=base_url,
            api_key=api_key,
            ret_pca=True,
            ret_patches=False,
            load_size=2048
        )
        print(f"✅ DINO PCA Features: shape {desc_pca.shape}")

        # Test visualization (exactly matching client demo)
        plt.figure(figsize=(8, 6))
        plt.imshow(desc_pca)
        plt.title("DINO Patch PCA Visualization")
        plt.axis("off")
        plt.show()
        print("✅ DINO PCA visualization displayed")

        return True

    except Exception as e:
        print(f"❌ DINO Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sam2_endpoints():
    """Test SAM2 API endpoints exactly matching the client demo"""
    print("\n🎭 Testing SAM2 API endpoints...")

    with open("sam2/server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)

    base_url = servers["sam2"]["base_url"]
    api_key = servers["sam2"]["api_key"]

    try:
        test_image_path = load_test_image()
        if not test_image_path:
            return False

        img = Image.open(test_image_path)
        print(f"Input image size: h={img.height}, w={img.width}")

        # Test health endpoint (no authentication required)
        health = get_sam2_health(base_url)
        print(f"✅ SAM2 Health: {health['status']}, GPU count: {health['gpu_count']}")

        print("=== Testing Automatic Mask Generation ===")

        # Test coarse masks (exactly matching client demo)
        print("Generating coarse masks...")
        coarse_masks = generate_sam2_auto_masks_decoded(
            test_image_path,
            base_url=base_url,
            api_key=api_key,
            preset="coarse"
        )
        print(f"✅ Generated {len(coarse_masks)} coarse masks")

        # Test visualization (exactly matching client demo)
        SAM2utils.visualize_masks(img, coarse_masks)
        plt.show()
        print("✅ SAM2 coarse masks visualization displayed")

        # Test fine-grained masks (exactly matching client demo)
        print("Generating fine-grained masks...")
        fine_masks = generate_sam2_auto_masks_decoded(
            test_image_path,
            base_url=base_url,
            api_key=api_key,
            preset="fine_grained"
        )
        print(f"✅ Generated {len(fine_masks)} fine-grained masks")

        # Test visualization (exactly matching client demo)
        SAM2utils.visualize_masks(img, fine_masks)
        plt.show()
        print("✅ SAM2 fine-grained masks visualization displayed")

        print("\n=== Testing Prompt-based Mask Generation ===")

        # Test single point (exactly matching client demo)
        print("\n--- Single Point Example ---")
        point_coords = np.array([[520, 820]])
        point_labels = np.array([1])  # 1 for foreground, 0 for background

        masks, scores, logits = generate_sam2_prompt_masks_decoded(
            test_image_path,
            base_url=base_url,
            api_key=api_key,
            point_coords=point_coords.tolist(),
            point_labels=point_labels.tolist(),
            multimask_output=True  # set to False for single mask output
        )

        print(f"✅ Generated {masks.shape[0]} masks from single point")

        # Test visualization (exactly matching client demo)
        SAM2utils.visualize_prompt_masks(np.array(img), masks, scores,
                                         point_coords, point_labels)
        plt.show()
        print("✅ SAM2 single point visualization displayed")

        # Test multiple points with mask input (exactly matching client demo)
        print("\n--- Multiple Points with Mask Input ---")
        best_mask_idx = np.argmax(scores)
        mask_input = logits[best_mask_idx:best_mask_idx + 1, :, :]

        point_coords_multi = np.array([[520, 820], [1090, 750]])
        point_labels_multi = np.array([1, 1])

        # Encode mask input with shape information
        mask_input_encoded, mask_input_shape = encode_mask_with_shape(mask_input)

        masks_multi, scores_multi, _ = generate_sam2_prompt_masks_decoded(
            test_image_path,
            base_url=base_url,
            api_key=api_key,
            point_coords=point_coords_multi.tolist(),
            point_labels=point_labels_multi.tolist(),
            mask_input=mask_input_encoded,
            mask_input_shape=mask_input_shape,
            multimask_output=False
        )

        print(f"✅ Refined with multiple points and mask input")

        # Test visualization (exactly matching client demo)
        SAM2utils.visualize_prompt_masks(np.array(img), masks_multi, scores_multi,
                                         point_coords_multi, point_labels_multi)
        plt.show()
        print("✅ SAM2 multiple points visualization displayed")

        # Test box prompt (exactly matching client demo)
        print("\n--- Box Prompt Example ---")
        box = np.array([280, 440, 1474, 1225])  # [x1, y1, x2, y2]

        box_masks, box_scores, _ = generate_sam2_prompt_masks_decoded(
            test_image_path,
            base_url=base_url,
            api_key=api_key,
            box=box.tolist(),
            multimask_output=False
        )

        print(f"✅ Generated mask from box prompt")

        # Test visualization (exactly matching client demo)
        SAM2utils.visualize_prompt_masks(np.array(img), box_masks, box_scores, box=box)
        plt.show()
        print("✅ SAM2 box prompt visualization displayed")

        # Test points + box (exactly matching client demo)
        print("\n--- Points + Box Example ---")
        point_coords_4 = np.array([[1398, 752], [520, 820]])
        point_labels_4 = np.array([0, 1])

        combined_masks, combined_scores, _ = generate_sam2_prompt_masks_decoded(
            test_image_path,
            base_url=base_url,
            api_key=api_key,
            point_coords=point_coords_4.tolist(),
            point_labels=point_labels_4.tolist(),  # Negative point
            box=box.tolist(),
            multimask_output=False
        )

        print(f"✅ Generated mask from box + negative point")

        # Test visualization (exactly matching client demo)
        SAM2utils.visualize_prompt_masks(np.array(img), combined_masks, combined_scores,
                                         point_coords=point_coords_4,
                                         point_labels=point_labels_4,
                                         box=box)
        plt.show()
        print("✅ SAM2 combined prompt visualization displayed")

        # Test SAM2utils analysis (exactly matching client demo)
        print("\n--- SAM2utils Analysis Demo ---")

        # Get mask statistics
        stats = SAM2utils.get_mask_statistics(coarse_masks)
        print(f"✅ Coarse mask statistics: {stats}")

        # Filter masks by area
        large_masks = SAM2utils.filter_masks_by_area(coarse_masks, min_area=5000)
        print(f"✅ Large masks (>5000 pixels): {len(large_masks)}")

        # Filter by quality scores
        high_quality_masks = SAM2utils.filter_masks_by_score(coarse_masks, min_iou=0.8)
        print(f"✅ High quality masks (IoU>0.8): {len(high_quality_masks)}")

        # Visualize filtered results
        if large_masks:
            print("✅ Visualizing large masks:")
            SAM2utils.visualize_masks(img, large_masks)
            plt.show()
            print("✅ SAM2 large masks visualization displayed")

        return True

    except Exception as e:
        print(f"❌ SAM2 Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_authentication():
    """Test API key authentication for all servers"""
    print("\n🔐 Testing API Key Authentication...")

    with open("sam2/server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)

    test_image_path = load_test_image()
    if not test_image_path:
        return

    image_base64 = encode_image(test_image_path)

    # Test all three servers
    for service_name, service_config in servers.items():
        print(f"\n--- Testing {service_name.upper()} Authentication ---")
        base_url = service_config["base_url"]
        valid_api_key = service_config["api_key"]

        # Test valid API key
        try:
            if service_name == "clip":
                result = extract_clip_features(
                    image=image_base64,
                    base_url=base_url,
                    ret_pca=True,
                    ret_patches=False,
                    load_size=2048,
                    api_key=valid_api_key
                )
                print(f"✅ {service_name.upper()} valid API key: SUCCESS")
            elif service_name == "dino":
                result = extract_dino_features(
                    image=image_base64,
                    base_url=base_url,
                    ret_pca=True,
                    ret_patches=False,
                    load_size=2048,
                    api_key=valid_api_key
                )
                print(f"✅ {service_name.upper()} valid API key: SUCCESS")
            elif service_name == "sam2":
                result = generate_sam2_auto_masks(
                    image=image_base64,
                    base_url=base_url,
                    preset="coarse",
                    api_key=valid_api_key
                )
                print(f"✅ {service_name.upper()} valid API key: SUCCESS")
        except Exception as e:
            print(f"❌ {service_name.upper()} valid API key failed: {e}")

        # Test invalid API key
        try:
            headers = {"Content-Type": "application/json", "Authorization": "Bearer invalid-key"}
            if service_name == "clip":
                endpoint = "/extract"
                payload = {"image": image_base64, "ret_pca": True, "ret_patches": False, "load_size": 2048}
            elif service_name == "dino":
                endpoint = "/extract"
                payload = {"image": image_base64, "ret_pca": True, "ret_patches": False, "load_size": 2048}
            elif service_name == "sam2":
                endpoint = "/auto_mask"
                payload = {"image": image_base64, "preset": "coarse"}

            response = requests.post(f"{base_url}{endpoint}", json=payload, headers=headers)
            if response.status_code == 401:
                print(f"✅ {service_name.upper()} correctly rejected invalid API key")
            else:
                print(f"❌ {service_name.upper()} should have rejected invalid API key, got {response.status_code}")
        except Exception as e:
            print(f"❌ {service_name.upper()} invalid API key test error: {e}")

        # Test missing API key
        try:
            headers_no_auth = {"Content-Type": "application/json"}
            response_no_auth = requests.post(f"{base_url}{endpoint}", json=payload, headers=headers_no_auth)
            if response_no_auth.status_code == 401:
                print(f"✅ {service_name.upper()} correctly rejected request without API key")
            else:
                print(f"❌ {service_name.upper()} should have rejected request without API key, got {response_no_auth.status_code}")
        except Exception as e:
            print(f"❌ {service_name.upper()} missing API key test error: {e}")


def test_image_url():
    """Test image URL support for all servers"""
    print("\n🌐 Testing Image URL Support...")

    with open("sam2/server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)

    # Use a simple, reliable test image URL
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=224&h=224&fit=crop"

    # Test CLIP with URL
    try:
        clip_config = servers["clip"]
        clip_result = extract_clip_features(
            image_url=image_url,
            base_url=clip_config["base_url"],
            ret_pca=True,
            ret_patches=False,
            load_size=2048,
            api_key=clip_config["api_key"]
        )
        print(f"✅ CLIP Image URL: shape {clip_result['shape']}")
    except Exception as e:
        print(f"❌ CLIP Image URL error: {e}")

    # Test DINO with URL
    try:
        dino_config = servers["dino"]
        dino_result = extract_dino_features(
            image_url=image_url,
            base_url=dino_config["base_url"],
            ret_pca=True,
            ret_patches=False,
            load_size=2048,
            api_key=dino_config["api_key"]
        )
        print(f"✅ DINO Image URL: shape {dino_result['shape']}")
    except Exception as e:
        print(f"❌ DINO Image URL error: {e}")

    # Test SAM2 with URL
    try:
        sam2_config = servers["sam2"]
        sam2_result = generate_sam2_auto_masks(
            image_url=image_url,
            base_url=sam2_config["base_url"],
            preset="coarse",
            api_key=sam2_config["api_key"]
        )
        print(f"✅ SAM2 Image URL: {sam2_result['num_masks']} masks")
    except Exception as e:
        print(f"❌ SAM2 Image URL error: {e}")


def test_concurrent_requests(num_requests=10, timeout_seconds=1800, stagger_seconds=0.5):
    """Test concurrent requests for all servers"""
    print(f"\n⚡ Testing Concurrent Requests (num_requests={num_requests}, timeout={timeout_seconds}s)...")

    with open("sam2/server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)

    test_image_path = load_test_image()
    if not test_image_path:
        return

    image_base64 = encode_image(test_image_path)

    async def make_request(session, request_id, service_name, base_url, api_key, image_base64):
        try:
            if service_name == "clip":
                endpoint = "/extract"
                payload = {
                    "image": image_base64,
                    "ret_pca": True,
                    "ret_patches": False,
                    "load_size": 2048
                }
            elif service_name == "dino":
                endpoint = "/extract"
                payload = {
                    "image": image_base64,
                    "ret_pca": True,
                    "ret_patches": False,
                    "load_size": 2048
                }
            elif service_name == "sam2":
                endpoint = "/auto_mask"
                payload = {
                    "image": image_base64,
                    "preset": "coarse"
                }
            else:
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': f"Unknown service: {service_name}"
                }

            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

            start_time = time.time()
            async with session.post(f"{base_url}{endpoint}", json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    request_time = time.time() - start_time
                    return {
                        'request_id': request_id,
                        'success': True,
                        'time': request_time,
                        'shape': result.get('shape', 'N/A')
                    }
                else:
                    error_text = await response.text()
                    return {
                        'request_id': request_id,
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except asyncio.TimeoutError:
            return {
                'request_id': request_id,
                'success': False,
                'error': "Request timeout"
            }
        except Exception as e:
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e)
            }

    async def run_concurrent_test(service_name, num_requests=10):
        print(f"\n--- Testing {service_name.upper()} Concurrent Requests ---")
        config = servers[service_name]
        base_url = config["base_url"]
        api_key = config["api_key"]

        start_time = time.time()

        # Create session with configurable timeout for long-running requests
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)  # Use timeout in seconds
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for i in tqdm(range(num_requests), desc=f"Sending {service_name} requests"):
                # Add small staggering: delay each request by stagger_seconds to allow workers to spawn
                await asyncio.sleep(stagger_seconds)
                task = asyncio.create_task(make_request(session, i, service_name, base_url, api_key, image_base64))
                tasks.append(task)

            # Track responses with progress bar
            results = []
            with tqdm(total=num_requests, desc=f"Receiving {service_name} responses") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': len([r for r in results if r['success']]),
                        'failed': len([r for r in results if not r['success']])
                    })

            end_time = time.time()
            total_time = end_time - start_time

            successful_requests = [r for r in results if r['success']]
            failed_requests = [r for r in results if not r['success']]

            print(f"📊 {service_name.upper()} Concurrent Test Results:")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Successful requests: {len(successful_requests)}/{num_requests}")
            print(f"   Failed requests: {len(failed_requests)}/{num_requests}")

            if successful_requests:
                avg_time = sum(r['time'] for r in successful_requests) / len(successful_requests)
                print(f"   Average request time: {avg_time:.2f}s")
                print(f"   Throughput: {len(successful_requests)/total_time:.2f} requests/second")

            for result in failed_requests[:3]:  # Show first 3 failures
                print(f"   Failed request {result['request_id']}: {result['error']}")

    # Test concurrent requests for all services in sequence
    for service_name in ["clip", "dino", "sam2"]:
        print(f"\n--- Testing {service_name.upper()} Concurrent Requests ---")
        asyncio.run(run_concurrent_test(service_name, num_requests))


def test_parameter_validation():
    """Test parameter validation and edge cases"""
    print("\n🧪 Testing Parameter Validation...")

    with open("sam2/server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)

    test_image_path = load_test_image()
    if not test_image_path:
        return

    image_base64 = encode_image(test_image_path)

    # Test CLIP parameter validation
    print("--- Testing CLIP Parameter Validation ---")
    clip_config = servers["clip"]

    # Test invalid model name
    try:
        result = extract_clip_features(
            image=image_base64,
            base_url=clip_config["base_url"],
            model_name="invalid-model-name",
            api_key=clip_config["api_key"]
        )
        print("❌ CLIP should have rejected invalid model name")
    except Exception as e:
        print(f"✅ CLIP correctly rejected invalid model name: {type(e).__name__}")

    # Test invalid load_size
    try:
        result = extract_clip_features(
            image=image_base64,
            base_url=clip_config["base_url"],
            load_size=0,  # Invalid size
            api_key=clip_config["api_key"]
        )
        print("❌ CLIP should have rejected invalid load_size")
    except Exception as e:
        print(f"✅ CLIP correctly handled invalid load_size: {type(e).__name__}")

    # Test DINO parameter validation
    print("--- Testing DINO Parameter Validation ---")
    dino_config = servers["dino"]

    # Test invalid model type
    try:
        result = extract_dino_features(
            image=image_base64,
            base_url=dino_config["base_url"],
            model_type="invalid-model-type",
            api_key=dino_config["api_key"]
        )
        print("❌ DINO should have rejected invalid model type")
    except Exception as e:
        print(f"✅ DINO correctly rejected invalid model type: {type(e).__name__}")

    # Test invalid load_size for DINO
    try:
        result = extract_dino_features(
            image=image_base64,
            base_url=dino_config["base_url"],
            load_size=0,  # Invalid size
            api_key=dino_config["api_key"]
        )
        print("❌ DINO should have rejected invalid load_size")
    except Exception as e:
        print(f"✅ DINO correctly handled invalid load_size: {type(e).__name__}")

    # Test SAM2 parameter validation
    print("--- Testing SAM2 Parameter Validation ---")
    sam2_config = servers["sam2"]

    # Test invalid point coordinates
    try:
        result = generate_sam2_prompt_masks(
            image=image_base64,
            base_url=sam2_config["base_url"],
            point_coords=[[1000, 1000]],  # Outside image bounds
            point_labels=[1],
            api_key=sam2_config["api_key"]
        )
        print("✅ SAM2 handled out-of-bounds coordinates")
    except Exception as e:
        print(f"⚠️ SAM2 with out-of-bounds coordinates: {type(e).__name__}")

    # Test mismatched point coords and labels
    try:
        result = generate_sam2_prompt_masks(
            image=image_base64,
            base_url=sam2_config["base_url"],
            point_coords=[[112, 112], [150, 150]],
            point_labels=[1],  # Only one label for two points
            api_key=sam2_config["api_key"]
        )
        print("❌ SAM2 should have rejected mismatched point coords/labels")
    except Exception as e:
        print(f"✅ SAM2 correctly rejected mismatched coords/labels: {type(e).__name__}")


def test_convenience_functions():
    """Test all convenience functions from client modules"""
    print("\n🔧 Testing Convenience Functions...")

    with open("sam2/server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)

    test_image_path = load_test_image()
    if not test_image_path:
        return

    # Test CLIP convenience functions
    print("--- Testing CLIP Convenience Functions ---")
    clip_config = servers["clip"]

    try:
        # Test extract_clip_features_decoded
        features = extract_clip_features_decoded(
            test_image_path,
            base_url=clip_config["base_url"],
            ret_pca=True,
            ret_patches=False,
            load_size=2048,
            api_key=clip_config["api_key"]
        )
        print(f"✅ CLIP extract_clip_features_decoded: shape {features.shape}")

        # Test encode_clip_text_decoded
        text_emb = encode_clip_text_decoded(
            "car",
            base_url=clip_config["base_url"],
            api_key=clip_config["api_key"]
        )
        print(f"✅ CLIP encode_clip_text_decoded: shape {text_emb.shape}")

        # Test compute_clip_similarity_decoded
        sim_map = compute_clip_similarity_decoded(
            test_image_path,
            text="car",
            base_url=clip_config["base_url"],
            ret_pca=False,
            ret_patches=False,
            load_size=2048,
            softmax=0.25,
            api_key=clip_config["api_key"]
        )
        print(f"✅ CLIP compute_clip_similarity_decoded: shape {sim_map.shape}")

    except Exception as e:
        print(f"❌ CLIP convenience functions error: {e}")

    # Test DINO convenience functions
    print("--- Testing DINO Convenience Functions ---")
    dino_config = servers["dino"]

    try:
        features = extract_dino_features_decoded(
            test_image_path,
            base_url=dino_config["base_url"],
            ret_pca=True,
            ret_patches=False,
            load_size=2048,
            api_key=dino_config["api_key"]
        )
        print(f"✅ DINO extract_dino_features_decoded: shape {features.shape}")

    except Exception as e:
        print(f"❌ DINO convenience functions error: {e}")

    # Test SAM2 convenience functions
    print("--- Testing SAM2 Convenience Functions ---")
    sam2_config = servers["sam2"]

    try:
        # Test generate_sam2_auto_masks_decoded
        auto_masks = generate_sam2_auto_masks_decoded(
            test_image_path,
            base_url=sam2_config["base_url"],
            preset="coarse",
            api_key=sam2_config["api_key"]
        )
        print(f"✅ SAM2 generate_sam2_auto_masks_decoded: {len(auto_masks)} masks")

        # Test generate_sam2_prompt_masks_decoded
        masks, iou, low_res = generate_sam2_prompt_masks_decoded(
            test_image_path,
            base_url=sam2_config["base_url"],
            point_coords=[[520, 820]],
            point_labels=[1],
            api_key=sam2_config["api_key"]
        )
        print(f"✅ SAM2 generate_sam2_prompt_masks_decoded: masks {masks.shape}, IoU {iou.shape}")

        # Test specialized convenience functions
        point_result = sam2_point_mask(
            test_image_path,
            point_coords=[[520, 820]],
            point_labels=[1],
            base_url=sam2_config["base_url"],
            api_key=sam2_config["api_key"]
        )
        print(f"✅ SAM2 sam2_point_mask: masks {point_result[0].shape}")

        box_result = sam2_box_mask(
            test_image_path,
            box=[280, 440, 1474, 1225],
            base_url=sam2_config["base_url"],
            api_key=sam2_config["api_key"]
        )
        print(f"✅ SAM2 sam2_box_mask: masks {box_result[0].shape}")

        combined_result = sam2_combined_mask(
            test_image_path,
            point_coords=[[520, 820]],
            point_labels=[1],
            box=[280, 440, 1474, 1225],
            base_url=sam2_config["base_url"],
            api_key=sam2_config["api_key"]
        )
        print(f"✅ SAM2 sam2_combined_mask: masks {combined_result[0].shape}")

    except Exception as e:
        print(f"❌ SAM2 convenience functions error: {e}")


if __name__ == "__main__":
    # test_clip_endpoints()
    # test_dino_endpoints()
    # test_sam2_endpoints()
    # test_authentication()
    # test_image_url()
    # test_parameter_validation()
    # test_convenience_functions()
    test_concurrent_requests(200, timeout_seconds=7200, stagger_seconds=1.0)
