import requests
import base64
import json
from datetime import datetime
import cv2

def detect_celestial_bodies(image_path, roll, pitch, yaw, latitude, longitude, timestamp=None):
    """
    Call the celestial body detection API
    
    Parameters:
    - image_path: Path to the image file
    - roll, pitch, yaw: Orientation in degrees
    - latitude, longitude: GPS coordinates in degrees
    - timestamp: Optional datetime object (uses current time if not provided)
    
    Returns:
    - Dictionary with API response including detected bodies and marked image
    """
    # API endpoint
    url = "http://localhost:5000/detect"
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare the request data
    request_data = {
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "latitude": latitude,
        "longitude": longitude,
        "image": image_base64
    }
    
    # Add timestamp if provided
    if timestamp:
        request_data["timestamp"] = timestamp.isoformat()
    
    # Make the API request
    response = requests.post(url, json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        
        # Decode and save the result image
        if result.get('status') == 'success':
            result_image_base64 = result.get('result_image')
            result_image_bytes = base64.b64decode(result_image_base64)
            
            # Save the result image
            with open("result_image.jpg", "wb") as f:
                f.write(result_image_bytes)
            
            print(f"Result image saved as 'result_image.jpg'")
            
            # Print detected bodies
            print("Detected celestial bodies:")
            for body in result.get('detected_bodies', []):
                print(f"- {body['name']} at position ({body['x']}, {body['y']}), " 
                      f"Angular distance: {body['angular_distance']:.2f}°, "
                      f"Magnitude: {body['magnitude']:.2f}")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage
if __name__ == "__main__":
    # Example parameters
    image_path = "test.jpg"  # Replace with your image path
    roll = 0.0
    pitch = 45.0  # Looking up 45 degrees
    yaw = 0.0     # Facing North
    latitude = 40.7128  # New York City latitude
    longitude = -74.0060  # New York City longitude
    
    # Current time
    current_time = datetime.now()
    
    # Call the API
    result = detect_celestial_bodies(
        image_path, roll, pitch, yaw, latitude, longitude, current_time
    )
    
    # You can now use the 'result' dictionary as needed