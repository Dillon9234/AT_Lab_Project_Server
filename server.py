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
from skyfield.api import load, wgs84
from skyfield.data import hipparcos
from skyfield.api import utc
import pandas as pd

app = Flask(__name__)

class CelestialBodyDetector:
    def __init__(self):
        self.latitude = 0
        self.longitude = 0
        self.celestial_bodies = []  # Will be populated later
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
        """
        Convert a viewing direction to celestial coordinates (RA and Dec)
        
        Parameters:
        -----------
        direction_vector : list or array-like
            A normalized 3D vector representing the viewing direction in local coordinates
            [x, y, z] where x points East, y points North, and z points up
        latitude : float
            Observer's latitude in degrees (positive for North, negative for South)
        longitude : float
            Observer's longitude in degrees (positive for East, negative for West)
        timestamp : datetime
            Time of observation (UTC)
        
        Returns:
        --------
        tuple
            (right_ascension, declination) where:
            - right_ascension is in hours (0 to 24)
            - declination is in degrees (-90 to +90)
        """
        import math
        from datetime import datetime, timedelta
        from skyfield.api import utc
        
        # Normalize the direction vector to ensure it's a unit vector
        magnitude = math.sqrt(sum(x*x for x in direction_vector))
        if abs(magnitude - 1.0) > 1e-6:  # If not already normalized
            direction_vector = [x/magnitude for x in direction_vector]
        
        # Extract the direction vector components
        # Note: assuming x=East, y=North, z=Up in the local horizontal frame
        x, y, z = direction_vector
        
        # Convert direction vector to horizontal coordinates (azimuth and altitude)
        # Azimuth: angle measured eastward from north (standard astronomical definition)
        azimuth = math.degrees(math.atan2(x, y))
        # Ensure azimuth is in the range [0, 360)
        azimuth = (azimuth + 360) % 360
        
        # Altitude: angle above the horizon
        altitude = math.degrees(math.asin(z))
        
        # Calculate Greenwich Mean Sidereal Time (GMST)
        # Use more accurate formula for GMST calculation
        # First, calculate J2000 date
        # Make sure both datetimes have timezone info for consistent subtraction
        j2000_date = datetime(2000, 1, 1, 12, 0, 0, tzinfo=utc)  # J2000 epoch with timezone
        dt_j2000 = timestamp - j2000_date
        days_since_j2000 = dt_j2000.total_seconds() / 86400.0
        
        # Calculate GMST in hours
        # This formula gives GMST at 0h UTC
        gmst_at_0h = (18.697374558 + 24.06570982441908 * days_since_j2000) % 24
        
        # Correct for the current UTC time
        utc_hours = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
        gmst = (gmst_at_0h + utc_hours * 1.002737909) % 24
        
        # Calculate Local Sidereal Time (LST)
        # Convert longitude to hours (15 degrees = 1 hour)
        lst = (gmst + longitude/15.0) % 24
        
        # Convert horizontal coordinates to equatorial coordinates
        lat_rad = math.radians(latitude)
        alt_rad = math.radians(altitude)
        az_rad = math.radians(azimuth)
        
        # Calculate declination
        sin_dec = math.sin(alt_rad) * math.sin(lat_rad) + math.cos(alt_rad) * math.cos(lat_rad) * math.cos(az_rad)
        dec_rad = math.asin(sin_dec)
        dec = math.degrees(dec_rad)
        
        # Calculate hour angle
        cos_ha = (math.sin(alt_rad) - math.sin(lat_rad) * sin_dec) / (math.cos(lat_rad) * math.cos(dec_rad))
        
        # Handle edge cases where floating-point errors might lead to values outside [-1, 1]
        cos_ha = max(min(cos_ha, 1.0), -1.0)
        ha_rad = math.acos(cos_ha)
        
        # Adjust sign of hour angle based on azimuth
        if azimuth > 180 and azimuth < 360:
            ha_rad = -ha_rad
        
        # Convert hour angle from radians to hours
        ha = math.degrees(ha_rad) / 15.0
        
        # Calculate Right Ascension from LST and hour angle
        ra = (lst - ha) % 24
            
        return ra, dec

    def update_celestial_bodies(self, timestamp):
        # Load the necessary data files
        planets = load('de421.bsp')  # Ephemeris file for solar system bodies
        ts = load.timescale()
        
        # Convert timestamp to Skyfield time
        t = ts.from_datetime(timestamp)
        
        # Set up observer location
        earth = planets['earth']
        observer = earth + wgs84.latlon(self.latitude, self.longitude)
        
        # Calculate positions for solar system bodies
        sun = planets['sun']
        moon = planets['moon']
        mercury = planets['mercury barycenter']
        venus = planets['venus barycenter']
        mars = planets['mars barycenter']
        jupiter = planets['jupiter barycenter']
        saturn = planets['saturn barycenter']
        
        # Get positions relative to observer
        sun_astrometric = observer.at(t).observe(sun).apparent()
        moon_astrometric = observer.at(t).observe(moon).apparent()
        mercury_astrometric = observer.at(t).observe(mercury).apparent()
        venus_astrometric = observer.at(t).observe(venus).apparent()
        mars_astrometric = observer.at(t).observe(mars).apparent()
        jupiter_astrometric = observer.at(t).observe(jupiter).apparent()
        saturn_astrometric = observer.at(t).observe(saturn).apparent()
        
        # Convert to RA/Dec
        sun_ra, sun_dec, _ = sun_astrometric.radec()
        moon_ra, moon_dec, _ = moon_astrometric.radec()
        mercury_ra, mercury_dec, _ = mercury_astrometric.radec()
        venus_ra, venus_dec, _ = venus_astrometric.radec()
        mars_ra, mars_dec, _ = mars_astrometric.radec()
        jupiter_ra, jupiter_dec, _ = jupiter_astrometric.radec()
        saturn_ra, saturn_dec, _ = saturn_astrometric.radec()
        
        # Load bright star data
        star_data = []
        try:
            with load.open(hipparcos.URL) as f:
                df = hipparcos.load_dataframe(f)
                
                # Filter for bright stars - using actual column names from debug output
                bright_stars = df[df['magnitude'] <= 1.5]  # Filter for bright stars
                
                # Use the columns we know exist from debugging output
                for _, star in bright_stars.iterrows():
                    # Use index as name since there's no name/proper column
                    name = f"Star {_}"
                    
                    # Use confirmed column names from debug output
                    ra_hours = star['ra_hours']
                    dec_deg = star['dec_degrees']
                    mag = star['magnitude']
                    
                    star_data.append((name, ra_hours, dec_deg, mag))
        except Exception as e:
            print(f"Error loading star data: {e}")
            # Continue with just planets if star data fails
        
        # Update the celestial bodies with calculated positions
        self.celestial_bodies = [
            ("Sun", sun_ra.hours, sun_dec.degrees, -26.7),
            ("Moon", moon_ra.hours, moon_dec.degrees, -12.6),
            ("Mercury", mercury_ra.hours, mercury_dec.degrees, -0.2),
            ("Venus", venus_ra.hours, venus_dec.degrees, -4.1),
            ("Mars", mars_ra.hours, mars_dec.degrees, 1.8),
            ("Jupiter", jupiter_ra.hours, jupiter_dec.degrees, -2.7),
            ("Saturn", saturn_ra.hours, saturn_dec.degrees, 0.5)
        ] + star_data
    
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
        self.latitude = latitude
        self.longitude = longitude
        self.update_celestial_bodies(timestamp)
            
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
            # Add timezone information if missing
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=utc)
        else:
            timestamp = datetime.now(tz=utc)  # Create with timezone
        
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
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error processing request: {e}")
        print(error_traceback)
        return jsonify({
            'error': str(e),
            'traceback': error_traceback,
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