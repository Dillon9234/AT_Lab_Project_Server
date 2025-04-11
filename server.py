from flask import Flask, request, jsonify
from skyfield.api import load, wgs84
from datetime import datetime

app = Flask(__name__)

# Load the ephemeris data (planetary position data)
eph = load('de421.bsp')
ts = load.timescale()

# Define the celestial bodies we want to track
sun = eph['sun']
mercury = eph['mercury']
venus = eph['venus']
earth = eph['earth']
moon = eph['moon']
mars = eph['mars']
jupiter = eph['jupiter barycenter']
saturn = eph['saturn barycenter']
uranus = eph['uranus barycenter']
neptune = eph['neptune barycenter']

@app.route('/celestial', methods=['POST'])
def get_celestial_coordinates():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract latitude, longitude, altitude, and timestamp
        latitude = data.get('latitude', 0.0)
        longitude = data.get('longitude', 0.0)
        altitude = data.get('altitude', 0.0)
        
        # Parse timestamp or use current time
        timestamp = data.get('timestamp')
        if timestamp:
            dt = datetime.fromtimestamp(timestamp)
        else:
            dt = datetime.now()
        
        # Convert to Skyfield time
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        
        # Create observer location
        location = earth + wgs84.latlon(latitude, longitude, altitude)
        
        # Calculate observations from the location
        observer = location.at(t)
        
        # Get positions of celestial bodies
        result = {
            'timestamp': dt.isoformat(),
            'observer': {
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude
            },
            'celestial_bodies': {}
        }
        
        # Calculate positions for each body
        bodies = {
            'sun': sun,
            'mercury': mercury,
            'venus': venus,
            'moon': moon,
            'mars': mars,
            'jupiter': jupiter,
            'saturn': saturn,
            'uranus': uranus,
            'neptune': neptune
        }
        
        for name, body in bodies.items():
            apparent = observer.observe(body).apparent()
            
            # Get equatorial coordinates (RA/Dec)
            ra, dec, _ = apparent.radec()
            
            # Add to result
            result['celestial_bodies'][name] = {
                'ra': {
                    'hours': ra.hours,
                    'degrees': ra.degrees,
                    'string': str(ra)
                },
                'dec': {
                    'degrees': dec.degrees,
                    'string': str(dec)
                }
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
