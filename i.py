from flask import Flask, jsonify, request, render_template
import google.generativeai as genai
import pandas as pd
import re
from dataclasses import dataclass
from enum import Enum
from flask_cors import CORS
import requests
from typing import Tuple, Optional

app = Flask(__name__)
CORS(app)

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LocationType(Enum):
    POINT = "point"
    AREA = "area"

class SearchScope(Enum):
    NEARBY = "nearby"
    LOCAL = "local"
    CITY = "city"
    METRO = "metro"
    REGION = "region"

@dataclass
class SearchRegion:
    center_lat: float
    center_lon: float
    radius_km: float
    name: str

class LocationService:
    def __init__(self):
        self.default_location = (19.0549, 72.8258)  # Bandra Mount Mary coordinates
        self.regions = {}
        logger.info(f"LocationService initialized with default location: {self.default_location}")

    def get_user_location(self) -> Tuple[float, float]:
        try:
            ip = request.headers.get('X-Forwarded-For', request.remote_addr)
            logger.debug(f"Attempting to get location for IP: {ip}")
            
            response = requests.get(f'http://ip-api.com/json/{ip}')
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"IP API Response: {data}")
                
                if data.get('status') == 'success':
                    location = (data['lat'], data['lon'])
                    logger.info(f"Successfully got user location: {location}")
                    return location
        except Exception as e:
            logger.error(f"Error getting user location: {e}")
        
        logger.warning(f"Using default location: {self.default_location}")
        return self.default_location

    def initialize_regions(self, center_lat: float, center_lon: float):
        logger.info(f"Initializing regions around center: ({center_lat}, {center_lon})")
        self.regions = {
            SearchScope.NEARBY.value: SearchRegion(center_lat, center_lon, 5, "Nearby Area"),
            SearchScope.LOCAL.value: SearchRegion(center_lat, center_lon, 15, "Local Area"),
            SearchScope.CITY.value: SearchRegion(center_lat, center_lon, 30, "City Area"),
            SearchScope.METRO.value: SearchRegion(center_lat, center_lon, 50, "Metropolitan Area"),
            SearchScope.REGION.value: SearchRegion(center_lat, center_lon, 200, "Regional Area")
        }

    def get_search_region(self, scope: SearchScope) -> SearchRegion:
        """Get the SearchRegion for the specified scope."""
        if not self.regions:
            raise ValueError("Regions not initialized. Call initialize_regions first.")
        
        region = self.regions.get(scope.value)
        if not region:
            region = self.regions[SearchScope.LOCAL.value]
        
        return region

def determine_search_scope(query: str) -> SearchScope:
    query_lower = query.lower()
    scope = SearchScope.LOCAL  # default scope
    
    if any(word in query_lower for word in ['college', 'university', 'hospital', 'mall']):
        scope = SearchScope.METRO
    elif any(word in query_lower for word in ['park', 'beach', 'forest', 'hill', 'mountain', 'island']):
        scope = SearchScope.REGION
    elif any(word in query_lower for word in ['restaurant', 'shop', 'cafe', 'store', 'salon']):
        scope = SearchScope.NEARBY
    elif 'city' in query_lower:
        scope = SearchScope.CITY
    
    logger.info(f"Determined search scope for query '{query}': {scope.value}")
    return scope

def create_structured_data(response_text: str, region: SearchRegion) -> pd.DataFrame:
    logger.debug(f"Processing raw response text:\n{response_text}")
    sections = response_text.split('\n\n')
    records = []

    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        logger.debug(f"Processing section {i + 1}:\n{section}")
        
        name_match = re.search(r'(?:name|location|place):\s*([^\n]+)', section, re.I)
        type_match = re.search(r'(?:type|category):\s*([^\n]+)', section, re.I)
        description_match = re.search(r'description:\s*([^\n]+)', section, re.I)
        # Updated regex to be more precise in capturing coordinates
        coordinates_match = re.search(r'coordinates:\s*([-+]?\d*\.\d+|\d+)\s*,\s*([-+]?\d*\.\d+|\d+)', section, re.I)

        if name_match and coordinates_match:
            try:
                # Extract coordinates and convert to float with full precision
                lat = float(coordinates_match.group(1))
                lon = float(coordinates_match.group(2))
                
                record = {
                    'name': name_match.group(1).strip(),
                    'type': type_match.group(1).strip() if type_match else 'unspecified',
                    'latitude': lat,
                    'longitude': lon,
                    'description': description_match.group(1).strip() if description_match else None
                }
                records.append(record)
                logger.info(f"Added location: {record['name']} at ({lat}, {lon})")
            except ValueError as e:
                logger.error(f"Error parsing coordinates in section {i + 1}: {e}")
                continue

    df = pd.DataFrame(records)
    # Ensure coordinates are stored as float64 to maintain precision
    if not df.empty:
        df['latitude'] = df['latitude'].astype('float64')
        df['longitude'] = df['longitude'].astype('float64')
    
    return df

def process_location_query(query: str, location_service: LocationService, api_key: str) -> pd.DataFrame:
    logger.info(f"Processing query: {query}")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    scope = determine_search_scope(query)
    region = location_service.get_search_region(scope)
    
    prompt = f"""
    User Query: {query}
    Search Region: {region.name} (Radius: {region.radius_km}km)
    Center Location: Latitude {region.center_lat}, Longitude {region.center_lon}
    
    Provide relevant minimum 1, max 25 locations recommendations within the specified search region. Include both well-known and lesser-known places that match the query.
    
    Format each recommendation as:

    Name: [location name]
    Type: [specific category]
    Description: [brief description including notable features]
    Coordinates: [latitude longitude]
    """
    
    try:
        response = model.generate_content(prompt)
        results_df = create_structured_data(response.text, region)
        results_df['search_scope'] = scope.value
        results_df['search_region'] = region.name
        
        return results_df
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/user-location', methods=['GET'])
def get_user_location():
    location_service = LocationService()
    lat, lon = location_service.get_user_location()
    return jsonify({"latitude": lat, "longitude": lon})

@app.route('/api/places', methods=['GET'])
def get_places():
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    location_service = LocationService()
    user_lat, user_lon = location_service.get_user_location()
    location_service.initialize_regions(user_lat, user_lon)
    
    api_key = "AIzaSyBr0SlLEH4eLM9AHyMtItKpN5nvZ1APGjM"
    
    try:
        results_df = process_location_query(query, location_service, api_key)
        if results_df.empty:
            return jsonify({"error": "No results found for the query"}), 404
        
        places = results_df.to_dict(orient='records')
        return jsonify({
            "user_location": {"latitude": user_lat, "longitude": user_lon},
            "places": places
        })
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)




























