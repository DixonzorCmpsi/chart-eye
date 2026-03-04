#!/usr/bin/env python3
"""
Test script for Chart Eye backend
"""
import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test the health check endpoint"""
    print("🧪 Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running on http://127.0.0.1:8000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze():
    """Test the /analyze endpoint with a minimal request"""
    print("\n🧪 Testing /analyze endpoint...")
    
    # Create a minimal 1x1 pixel PNG (valid but tiny)
    import base64
    tiny_png = base64.b64encode(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x054\xc0\x1e\xb5\x00\x00\x00\x00IEND\xaeB`\x82').decode()
    
    payload = {
        "image": tiny_png,
        "query": "What do you see?",
        "iteration": 0,
        "context": ""
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=30)
        if response.status_code == 200:
            print("✅ /analyze endpoint responded!")
            result = response.json()
            print(f"   Action: {result.get('action')}")
            if 'prediction' in result:
                print(f"   Prediction: {result['prediction'][:100]}...")
            return True
        else:
            print(f"❌ /analyze failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print("⏱️  Request timed out (model analysis is slow)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Chart Eye Backend Test Suite")
    print("=" * 60)
    
    # Give server time to start if just launched
    print("\n⏳ Waiting for server to initialize (checking for 5 seconds)...")
    for i in range(5):
        time.sleep(1)
        try:
            requests.get(f"{BASE_URL}/", timeout=1)
            print(f"✅ Server is now ready!")
            break
        except:
            print(f"   Attempt {i+1}/5...")
    
    health_ok = test_health()
    
    if health_ok:
        test_analyze()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
