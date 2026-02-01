"""
Landmark Detection and Management Module

This module handles:
1. Loading landmarks from local JSON database
2. Fetching landmarks from external APIs (Google Places, OpenStreetMap)
3. Calculating distances and visibility based on GPS and orientation
"""

import math
import requests
import os
import json
from typing import List, Dict, Optional


def load_landmarks():
    """Load landmarks from JSON file"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    json_file = os.path.join(data_dir, 'landmarks.json')

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data['landmarks']
    except (FileNotFoundError, KeyError) as e:
        # Fallback to empty list if file not found
        print(f"Warning: landmarks.json not found: {e}, using empty database")
        return []


LANDMARKS = load_landmarks()


def get_nearby_landmarks_google_places(latitude: float, longitude: float, radius: int = 2000) -> List[Dict]:
    """Fetch nearby landmarks using Google Places API"""
    api_key = os.getenv('GOOGLE_PLACES_API_KEY')
    if not api_key:
        return []

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': f'{latitude},{longitude}',
        'radius': radius,
        'type': 'tourist_attraction|museum|park|stadium|church|synagogue|hindu_temple|mosque',
        'key': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        landmarks = []
        for place in data.get('results', []):
            # Get place details for more information
            details = get_place_details_google(place['place_id'], api_key)
            if details:
                landmarks.append({
                    'id': place['place_id'],
                    'name': place['name'],
                    'latitude': place['geometry']['location']['lat'],
                    'longitude': place['geometry']['location']['lng'],
                    'facts': details.get('facts', [f"Nearby attraction: {place['name']}"]),
                    'rating': place.get('rating', 0),
                    'types': place.get('types', [])
                })

        return landmarks[:10]  # Limit to 10 results

    except Exception as e:
        return []


def get_place_details_google(place_id: str, api_key: str) -> Optional[Dict]:
    """Get detailed information about a place from Google Places API"""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        'place_id': place_id,
        'fields': 'name,formatted_address,website,rating,reviews,types,editorial_summary',
        'key': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'OK':
            result = data['result']
            facts = []

            if result.get('editorial_summary'):
                facts.append(result['editorial_summary']['overview'])

            if result.get('website'):
                facts.append(f"Website: {result['website']}")

            if result.get('rating'):
                facts.append(f"Rating: {result['rating']}/5")

            if not facts:
                facts = [f"Popular attraction: {result['name']}"]

            return {'facts': facts}

    except Exception as e:
        return None


def get_nearby_landmarks_openstreetmap(latitude: float, longitude: float, radius: int = 2000) -> List[Dict]:
    """Fetch nearby landmarks using OpenStreetMap Overpass API"""
    # Overpass API query for tourist attractions
    query = f"""
    [out:json][timeout:25];
    (
      node["tourism"~"attraction|museum|viewpoint|artwork"](around:{radius},{latitude},{longitude});
      way["tourism"~"attraction|museum|viewpoint|artwork"](around:{radius},{latitude},{longitude});
      relation["tourism"~"attraction|museum|viewpoint|artwork"](around:{radius},{latitude},{longitude});
    );
    out center meta;
    """

    url = "https://overpass-api.de/api/interpreter"

    try:
        response = requests.post(url, data={'data': query}, timeout=10)
        response.raise_for_status()
        data = response.json()

        landmarks = []
        for element in data.get('elements', []):
            if 'center' in element:
                lat = element['center']['lat']
                lon = element['center']['lon']
            elif 'lat' in element and 'lon' in element:
                lat = element['lat']
                lon = element['lon']
            else:
                continue

            name = element.get('tags', {}).get('name', 'Unnamed landmark')
            description = element.get('tags', {}).get('description', '')

            facts = [f"Tourist attraction: {name}"]
            if description:
                facts.append(description)

            landmarks.append({
                'id': f"osm_{element['id']}",
                'name': name,
                'latitude': lat,
                'longitude': lon,
                'facts': facts,
                'source': 'openstreetmap'
            })

        return landmarks[:10]  # Limit to 10 results

    except Exception as e:
        return []


def get_combined_landmarks(latitude: float, longitude: float, radius: int = 2000) -> List[Dict]:
    """Get landmarks from multiple APIs and combine with local database"""
    all_landmarks = []

    # Add local landmarks first
    for landmark in LANDMARKS:
        distance = calculate_distance(
            latitude, longitude, landmark['latitude'], landmark['longitude'])
        if distance <= radius:
            all_landmarks.append(landmark)

    # Try Google Places API
    google_landmarks = get_nearby_landmarks_google_places(
        latitude, longitude, radius)
    all_landmarks.extend(google_landmarks)

    # Try OpenStreetMap API
    osm_landmarks = get_nearby_landmarks_openstreetmap(
        latitude, longitude, radius)
    all_landmarks.extend(osm_landmarks)

    # Remove duplicates based on name similarity (simple approach)
    unique_landmarks = []
    seen_names = set()

    for landmark in all_landmarks:
        name_lower = landmark['name'].lower()
        if name_lower not in seen_names:
            unique_landmarks.append(landmark)
            seen_names.add(name_lower)

    return unique_landmarks


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in meters"""
    R = 6371000  # Earth's radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2) * math.sin(delta_lat/2) + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * \
        math.sin(delta_lon/2) * math.sin(delta_lon/2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c

    return distance


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2 in degrees"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def find_visible_landmarks(user_lat, user_lon, user_heading, max_distance=50000, use_api=False):
    """Find landmarks that should be visible based on GPS and orientation"""
    if use_api:
        # Use API data combined with local database
        landmarks = get_combined_landmarks(user_lat, user_lon, max_distance)
    else:
        # Use only local database
        landmarks = LANDMARKS

    visible_landmarks = []

    for landmark in landmarks:
        # Calculate distance
        distance = calculate_distance(
            user_lat, user_lon, landmark['latitude'], landmark['longitude'])

        if distance > max_distance:
            continue

        # Calculate bearing to landmark
        bearing = calculate_bearing(
            user_lat, user_lon, landmark['latitude'], landmark['longitude'])

        # Calculate angle difference (how far off the landmark is from user's heading)
        angle_diff = abs(bearing - user_heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Check if landmark is within ~90 degree field of view
        if angle_diff <= 90:
            visible_landmarks.append({
                **landmark,
                'distance': distance,
                'bearing': bearing,
                'angle_diff': angle_diff
            })

    # Sort by distance (closest first)
    visible_landmarks.sort(key=lambda x: x['distance'])

    # If no visible landmarks but some within distance, return the closest one anyway for debugging
    if not visible_landmarks and landmarks:
        closest = min(landmarks, key=lambda l: calculate_distance(
            user_lat, user_lon, l['latitude'], l['longitude']))
        dist = calculate_distance(
            user_lat, user_lon, closest['latitude'], closest['longitude'])
        if dist <= max_distance:
            bearing = calculate_bearing(
                user_lat, user_lon, closest['latitude'], closest['longitude'])
            angle_diff = abs(bearing - user_heading)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            visible_landmarks = [{
                **closest,
                'distance': dist,
                'bearing': bearing,
                'angle_diff': angle_diff
            }]

    return visible_landmarks[:3]  # Return top 3 closest visible landmarks
