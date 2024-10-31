
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

    @property
    def bounds(self):
        # For the region scope, use fixed Mumbai metropolitan bounds
        if self.radius_km >= 200:  # Region scope
            return (
                19.5,  # North bound
                18.5,  # South bound
                73.5,  # East bound
                72.5   # West bound
            )
        
        # For other scopes, calculate based on radius
        deg_change = self.radius_km / 111
        return (
            self.center_lat + deg_change,
            self.center_lat - deg_change,
            self.center_lon + deg_change,
            self.center_lon - deg_change
        )

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
            SearchScope.REGION.value: SearchRegion(center_lat, center_lon, 200, "Mumbai Metropolitan Region")
        }
        for scope, region in self.regions.items():
            logger.debug(f"Region {scope}: center=({region.center_lat}, {region.center_lon}), radius={region.radius_km}km")

    def get_search_region(self, scope: SearchScope) -> SearchRegion:
        """Get the SearchRegion for the specified scope."""
        if not self.regions:
            raise ValueError("Regions not initialized. Call initialize_regions first.")
        
        region = self.regions.get(scope.value)
        if not region:
            logger.warning(f"Invalid scope {scope.value}, falling back to local scope")
            region = self.regions[SearchScope.LOCAL.value]
        
        return region

# Rest of the code remains the same...

def determine_search_scope(query: str) -> SearchScope:
    query_lower = query.lower()
    scope = SearchScope.LOCAL  # default scope
    
    if any(word in query_lower for word in ['college', 'university', 'hospital', 'mall']):
        scope = SearchScope.METRO
    elif any(word in query_lower for word in ['park', 'beach', 'forest', 'hill', 'mountain', 'island']):
        scope = SearchScope.REGION
    elif any(word in query_lower for word in ['restaurant', 'shop', 'cafe', 'store', 'salon']):
        scope = SearchScope.NEARBY
    elif 'mumbai' in query_lower:
        scope = SearchScope.CITY
    elif 'maharashtra' in query_lower:
        scope = SearchScope.REGION
        
    logger.info(f"Determined search scope for query '{query}': {scope.value}")
    return scope

def is_valid_coordinate(lat: float, lon: float, region: SearchRegion) -> bool:
    """Validate if coordinates are within reasonable bounds of the search region."""
    north, south, east, west = region.bounds
    valid = (
        -90 <= lat <= 90 and
        -180 <= lon <= 180 and
        south <= lat <= north and
        west <= lon <= east
    )
    if not valid:
        logger.warning(f"Invalid coordinates: ({lat}, {lon}) for region bounds: N:{north}, S:{south}, E:{east}, W:{west}")
    return valid

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
        coordinates_match = re.search(r'coordinates:\s*([\d.-]+)\s*([\d.-]+)', section, re.I)

        if name_match and coordinates_match:
            try:
                lat, lon = map(float, coordinates_match.groups())
                logger.debug(f"Found coordinates for {name_match.group(1)}: ({lat}, {lon})")
                
                if is_valid_coordinate(lat, lon, region):
                    record = {
                        'name': name_match.group(1).strip(),
                        'type': type_match.group(1).strip() if type_match else 'unspecified',
                        'latitude': lat,
                        'longitude': lon,
                        'description': description_match.group(1).strip() if description_match else None
                    }
                    records.append(record)
                    logger.info(f"Added location: {record['name']} at ({lat}, {lon})")
                else:
                    logger.warning(f"Skipping location with invalid coordinates: {name_match.group(1)} at ({lat}, {lon})")
            except ValueError as e:
                logger.error(f"Error parsing coordinates in section {i + 1}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Created DataFrame with {len(df)} valid locations")
    if not df.empty:
        logger.debug("Sample of processed data:")
        logger.debug(df.head().to_string())
    return df

def process_location_query(query: str, location_service: LocationService, api_key: str) -> pd.DataFrame:
    logger.info(f"Processing query: {query}")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    scope = determine_search_scope(query)
    region = location_service.get_search_region(scope)
    
    logger.debug(f"Search region: {region.name}, Center: ({region.center_lat}, {region.center_lon}), Radius: {region.radius_km}km")

    prompt = f"""
    User Query: {query}
    Search Region: {region.name} (Radius: {region.radius_km}km)
    Center Location: Latitude {region.center_lat}, Longitude {region.center_lon}
    
    Provide relevant minimum 1, max 25 locations recommendations within the specified search region. Include both well-known and lesser-known places that match the query.  
    Also if there's only 1 relevant result like largest church, biggest park then only return its values.
    if question asks for biggest largest smallest return only single result
    IMPORTANT: IMPORTANT: All coordinates must be valid numbers within the Mumbai region bounds (approximately 18.5°N to 19.5°N and 72.5°E to 73.5°E). Ensure that the coordinates are in the format of two decimal numbers for latitude and longitude, e.g., (19.00, 72.00).

    
    Format each recommendation as:

    Name: [location name]
    Type: [specific category]
    Description: [brief description including notable features]
    Coordinates: [latitude longitude]
    
    Ensure all coordinates are valid numbers within the Mumbai region bounds.
    """
    
    try:
        logger.debug(f"Sending prompt to Gemini:\n{prompt}")
        response = model.generate_content(prompt)
        logger.debug(f"Received response from Gemini:\n{response.text}")
        
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
    logger.info(f"Returning user location: ({lat}, {lon})")
    return jsonify({"latitude": lat, "longitude": lon})

@app.route('/api/places', methods=['GET'])
def get_places():
    query = request.args.get('query', '').strip()
    logger.info(f"Received places query: '{query}'")
    
    if not query:
        logger.warning("Empty query received")
        return jsonify({"error": "No query provided"}), 400

    location_service = LocationService()
    user_lat, user_lon = location_service.get_user_location()
    location_service.initialize_regions(user_lat, user_lon)
    
    api_key = "AIzaSyBr0SlLEH4eLM9AHyMtItKpN5nvZ1APGjM"
    
    try:
        results_df = process_location_query(query, location_service, api_key)
        
        if results_df.empty:
            logger.warning("No results found for query")
            return jsonify({"error": "No results found for the query"}), 404
        
        places = results_df.to_dict(orient='records')
        response_data = {
            "user_location": {"latitude": user_lat, "longitude": user_lon},
            "places": places
        }
        
        logger.info(f"Returning {len(places)} places")
        logger.debug(f"Response data: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)