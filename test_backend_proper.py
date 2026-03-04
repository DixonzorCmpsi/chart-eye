#!/usr/bin/env python3
"""
Generate a test chart image and test the backend
"""
import requests
import time
import base64
from PIL import Image, ImageDraw
from io import BytesIO

BASE_URL = "http://127.0.0.1:8000"

def create_test_chart():
    """Create a simple test chart image"""
    # Create a 400x300 image with some lines to simulate a chart
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some chart elements (lines and grid)
    draw.line([(50, 250), (350, 250)], fill='black', width=2)  # x-axis
    draw.line([(50, 50), (50, 250)], fill='black', width=2)    # y-axis
    
    # Draw a simple ascending candlestick pattern
    for i in range(10):
        x = 100 + i * 20
        y_low = 200 - i * 5
        y_high = 100 - i * 8
        # Draw candle body
        draw.rectangle([x, y_high, x+10, y_low], outline='green', width=2)
        # Draw wicks
        draw.line([(x+5, y_high-20), (x+5, y_low+10)], fill='green', width=1)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64

def test_health():
    """Test health endpoint"""
    print("🧪 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model: {data.get('model')}")
            print(f"   Auth: {data.get('auth')}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze(image_b64):
    """Test analyze endpoint"""
    print("\n🧪 Testing /analyze with iteration=0 (region detection)...")
    
    payload = {
        "image": image_b64,
        "query": "Where is the chart region located?",
        "iteration": 0,
        "context": ""
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            print("✅ /analyze iteration=0 passed!")
            print(f"   Action: {data.get('action')}")
            if 'region' in data:
                print(f"   Region: {data.get('context')}")
            if 'prediction' in data:
                print(f"   Prediction: {data.get('prediction')[:100]}...")
            return data
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print("⏱️  Request timed out (expected for heavy model inference)")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("=" * 70)
    print("Chart Eye Backend Test Suite")
    print("=" * 70)
    
    # Wait for server
    print("\n⏳ Waiting for server to be ready...")
    for i in range(10):
        try:
            requests.get(f"{BASE_URL}/", timeout=1)
            print("✅ Server is ready!\n")
            break
        except:
            if i < 9:
                time.sleep(0.5)
            else:
                print("❌ Server not responding after 5 seconds")
                return
    
    # Test health
    if not test_health():
        print("⚠️  Health check failed, but continuing with tests...")
    
    # Generate test image
    print("\n📊 Generating test chart image...")
    test_image = create_test_chart()
    print(f"   Image size: {len(test_image)} bytes (base64)")
    
    # Test analyze
    result = test_analyze(test_image)
    
    print("\n" + "=" * 70)
    if result:
        print("✅ Backend tests PASSED!")
    else:
        print("⚠️  Some tests failed - check logs in backend terminal")
    print("=" * 70)

if __name__ == "__main__":
    main()
