import numpy as np
import cv2
import math
from datetime import datetime
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

class CelestialBodyDetector:
    def __init__(self):
        # Hard-coded database of celestial bodies
        # Format: name, right ascension (hours), declination (degrees), apparent magnitude
        self.celestial_bodies = [
            ("Sun", 0.0, 0.0, -26.7),  # RA and Dec will be calculated based on date
            ("Moon", 0.0, 0.0, -12.6),  # RA and Dec will be calculated based on date
            ("Mercury", 2.5, 15.2, -0.2),
            ("Venus", 4.3, 22.1, -4.1),
            ("Mars", 18.2, -24.3, 1.8),
            ("Jupiter", 12.5, -3.2, -2.7),
            ("Saturn", 20.7, -18.9, 0.5),
            ("Sirius", 6.75, -16.72, -1.46),
            ("Canopus", 6.4, -52.7, -0.72),
            ("Alpha Centauri", 14.66, -60.83, -0.27),
            ("Arcturus", 14.27, 19.18, -0.05),
            ("Vega", 18.62, 38.78, 0.03),
            ("Capella", 5.28, 46.0, 0.08),
            ("Rigel", 5.24, -8.2, 0.13),
            ("Procyon", 7.65, 5.21, 0.34),
            ("Betelgeuse", 5.92, 7.41, 0.5),
            ("Achernar", 1.63, -57.24, 0.46),
            ("Hadar", 14.06, -60.37, 0.61),
            ("Altair", 19.85, 8.87, 0.76),
            ("Acrux", 12.44, -63.1, 0.77)
        ]
        
        # Field of view for typical smartphone camera (in degrees)
        self.fov_horizontal = 66
        self.fov_vertical = 45
    
    def pixel_to_view_angle(self, x, y, image_width, image_height):
        """Convert pixel coordinates to viewing angles"""
        # Convert pixel coordinates to normalized coordinates (-1 to 1)
        x_norm = (x / image_width) * 2 - 1
        y_norm = (y / image_height) * 2 - 1
        
        # Convert normalized coordinates to angles
        angle_x = x_norm * (self.fov_horizontal / 2)
        angle_y = y_norm * (self.fov_vertical / 2)
        
        return angle_x, angle_y
    
    def rotate_vector(self, vec, roll, pitch, yaw):
        """Rotate a vector based on roll, pitch, yaw (in degrees)"""
        # Convert to radians
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        
        # Create rotation matrices
        # Roll (rotation around x-axis)
        R_roll = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])
        
        # Pitch (rotation around y-axis)
        R_pitch = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])
        
        # Yaw (rotation around z-axis)
        R_yaw = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations (order matters: yaw, then pitch, then roll)
        R = R_roll @ R_pitch @ R_yaw
        
        # Apply rotation
        rotated_vec = R @ vec
        
        return rotated_vec
    
    def view_direction_to_celestial(self, direction_vector, latitude, longitude, timestamp):
        """Convert a viewing direction to celestial coordinates (RA and Dec)"""
        # Simple approximation for demo purposes
        # A full implementation would need accurate astronomical calculations
        
        # Calculate Local Sidereal Time (LST)
        # This is a simplified approximation
        utc_hours = timestamp.hour + timestamp.minute / 60.0
        days_since_j2000 = (timestamp - datetime(2000, 1, 1)).total_seconds() / 86400.0
        lst_hours = (100.46 + 0.985647 * days_since_j2000 + longitude + 15 * utc_hours) % 24
        
        # Convert direction vector to horizontal coordinates (azimuth and altitude)
        azimuth = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
        altitude = math.degrees(math.asin(direction_vector[2]))
        
        # Convert azimuth to standard form (0 to 360, measured from North)
        azimuth = (azimuth + 360) % 360
        
        # Convert horizontal coordinates to equatorial coordinates (RA and Dec)
        lat_rad = math.radians(latitude)
        alt_rad = math.radians(altitude)
        az_rad = math.radians(azimuth)
        
        # Calculate hour angle
        sin_dec = math.sin(alt_rad) * math.sin(lat_rad) + math.cos(alt_rad) * math.cos(lat_rad) * math.cos(az_rad)
        dec = math.degrees(math.asin(sin_dec))
        
        cos_h = (math.sin(alt_rad) - math.sin(lat_rad) * math.sin(math.radians(dec))) / (math.cos(lat_rad) * math.cos(math.radians(dec)))
        cos_h = max(min(cos_h, 1.0), -1.0)  # Clamp to [-1, 1]
        hour_angle = math.degrees(math.acos(cos_h))
        
        # Adjust sign of hour angle based on azimuth
        if azimuth > 180:
            hour_angle = -hour_angle
        
        # Calculate Right Ascension from LST and hour angle
        ra_hours = (lst_hours - hour_angle / 15) % 24
        
        return ra_hours, dec
    
    def is_celestial_body_visible(self, ra_hours, dec, tolerance=3.0):
        """Check if any celestial body is at the given RA and Dec"""
        visible_bodies = []
        
        for body in self.celestial_bodies:
            name, body_ra, body_dec, magnitude = body
            
            # Calculate angular distance between points
            ra_diff = min(abs(ra_hours - body_ra), 24 - abs(ra_hours - body_ra)) * 15  # Convert hours to degrees
            dec_diff = abs(dec - body_dec)
            
            # Use Pythagorean approximation for small angles
            angular_distance = math.sqrt(ra_diff**2 + dec_diff**2)
            
            # Adjust tolerance based on brightness (brighter objects are easier to see)
            adjusted_tolerance = tolerance * (1 + 0.1 * abs(magnitude)) if magnitude > 0 else tolerance
            
            if angular_distance < adjusted_tolerance:
                visible_bodies.append((name, angular_distance, magnitude))
        
        # Return the closest/brightest body if multiple are found
        if visible_bodies:
            # Sort by angular distance and brightness
            visible_bodies.sort(key=lambda x: (x[1], x[2]))
            return visible_bodies[0]
        
        return None
    
    def process_image(self, image, roll, pitch, yaw, latitude, longitude, timestamp):
        """Process the image and identify celestial bodies"""
        height, width = image.shape[:2]
        result_image = image.copy()
        
        # Define sample points (grid or specific regions of interest)
        # For demonstration, we'll use a grid of points
        num_points_x = 20
        num_points_y = 15
        step_x = width // num_points_x
        step_y = height // num_points_y
        
        # Camera pointing direction (assuming camera aligned with phone)
        # In a camera's coordinate system, typically:
        # x-axis points right, y-axis points down, z-axis points forward
        camera_direction = np.array([0, 0, 1])  # Looking forward
        
        # Rotate camera direction based on phone orientation
        rotated_direction = self.rotate_vector(camera_direction, roll, pitch, yaw)
        
        # Store detected celestial bodies for displaying
        detected_bodies = []
        
        # Check each point in the grid
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                # Convert pixel to view angle relative to center
                angle_x, angle_y = self.pixel_to_view_angle(x, y, width, height)
                
                # Create a unit vector for this pixel's view direction
                # Starting with the camera's forward direction
                view_direction = np.array([0, 0, 1])
                
                # Rotate the view for this specific pixel (relative to camera orientation)
                # First, create rotation matrices for the pixel's angular offset
                cosa_x = math.cos(math.radians(angle_x))
                sina_x = math.sin(math.radians(angle_x))
                cosa_y = math.cos(math.radians(angle_y))
                sina_y = math.sin(math.radians(angle_y))
                
                # Rotation around vertical axis (for horizontal angle)
                R_x = np.array([
                    [cosa_x, 0, sina_x],
                    [0, 1, 0],
                    [-sina_x, 0, cosa_x]
                ])
                
                # Rotation around horizontal axis (for vertical angle)
                R_y = np.array([
                    [1, 0, 0],
                    [0, cosa_y, -sina_y],
                    [0, sina_y, cosa_y]
                ])
                
                # Apply pixel-specific rotation
                pixel_direction = R_y @ R_x @ view_direction
                
                # Then apply the phone's orientation
                final_direction = self.rotate_vector(pixel_direction, roll, pitch, yaw)
                
                # Convert to celestial coordinates
                ra, dec = self.view_direction_to_celestial(final_direction, latitude, longitude, timestamp)
                
                # Check if any celestial body is at these coordinates
                celestial_object = self.is_celestial_body_visible(ra, dec)
                
                if celestial_object:
                    name, distance, magnitude = celestial_object
                    detected_bodies.append((name, x, y, distance, magnitude))
        
        # Draw circles for detected bodies
        for name, x, y, distance, magnitude in detected_bodies:
            # Scale circle size based on brightness (brighter = larger)
            radius = int(max(5, 20 - magnitude))
            # Color based on type (stars, planets, etc.) - for simplicity, we'll use brightness
            if name in ["Sun", "Moon"]:
                color = (0, 255, 255)  # Yellow
            elif name in ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"]:
                color = (0, 0, 255)    # Red
            else:
                color = (255, 255, 255)  # White for stars
                
            cv2.circle(result_image, (x, y), radius, color, 2)
            cv2.putText(result_image, name, (x + radius, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image, detected_bodies

# Create an instance of the detector
detector = CelestialBodyDetector()

@app.route('/detect', methods=['POST'])
def detect_celestial_bodies():
    try:
        # Get data from request
        data = request.json
        
        # Extract roll, pitch, yaw
        roll = float(data.get('roll', 0))
        pitch = float(data.get('pitch', 0))
        yaw = float(data.get('yaw', 0))
        
        # Extract GPS coordinates
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        
        # Extract timestamp or use current time
        timestamp_data = data.get('timestamp', None)
        if timestamp_data:
            timestamp = datetime.fromisoformat(timestamp_data)
        else:
            timestamp = datetime.now()
        
        # Extract and decode the image
        image_b64 = data.get('image', '')
        image_data = base64.b64decode(image_b64)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'error': 'Invalid image data',
                'status': 'error'
            }), 400
        
        # Process the image
        result_image, detected_bodies = detector.process_image(
            image, roll, pitch, yaw, latitude, longitude, timestamp
        )
        
        # Prepare the response
        detections = []
        for name, x, y, distance, magnitude in detected_bodies:
            detections.append({
                'name': name,
                'x': int(x),
                'y': int(y),
                'angular_distance': float(distance),
                'magnitude': float(magnitude)
            })
        
        # Encode the result image
        _, buffer = cv2.imencode('.jpg', result_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'detected_bodies': detections,
            'result_image': img_str
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# Optional route to check if the API is running
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Celestial Body Detection API is running'
    })

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)