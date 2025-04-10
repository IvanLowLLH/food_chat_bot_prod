import os
from typing import Dict, Union, Tuple, List
import requests
import json
import googlemaps
from geopy.distance import geodesic
from operator import itemgetter


class NearbyPlacesFinder:
    def __init__(self):
        """Initialize the NearbyPlacesFinder with Google Maps client."""
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        serper_api_key = os.getenv('SERPER_API')

        if not api_key:
            raise ValueError("Google Maps API key not found in environment variables")
        self.gmaps = googlemaps.Client(key=api_key)
        self.serper_url = "https://google.serper.dev/maps"
        self.serper_headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

    def get_coordinates(self, location: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
        """
        Get coordinates from either an address string or existing coordinates.

        Args:
            location: Either a string address or tuple of (latitude, longitude)

        Returns:
            Tuple of (latitude, longitude)
        """
        if isinstance(location, str):
            # Geocode the address
            result = self.gmaps.geocode(location)
            if not result:
                raise ValueError(f"Could not find coordinates for address: {location}")
            lat = result[0]['geometry']['location']['lat']
            lng = result[0]['geometry']['location']['lng']
            return (lat, lng)
        elif isinstance(location, (tuple, list)) and len(location) == 2:
            return location
        else:
            raise ValueError("Location must be either an address string or (latitude, longitude) tuple")

    def find_nearby_places(self, location: Union[str, Tuple[float, float]],
                           radius: int = 1000) -> Dict[str, Dict]:
        """
        Find the nearest places of different types (MRT stations, shopping malls, landmarks) to a given location.

        Args:
            location: Either an address string or (latitude, longitude) tuple
            radius: Search radius in meters (default: 500)

        Returns:
            Dictionary containing nearest places by type
        """
        coordinates = self.get_coordinates(location)

        # Define place types to search for
        # (generic term: search term)
        place_types = {
            'MRT/Subway Station': 'MRT',
            'Shopping Mall': 'Shopping Mall'
        }

        results = {}

        # Search for each place type
        for place_type, place_search in place_types.items():
            payload = json.dumps({
                "q": place_search,
                "ll": f"@{coordinates[0]}, {coordinates[1]}, 16z"
            })
            response = requests.request("POST", self.serper_url, headers=self.serper_headers, data=payload)
            places_result = json.loads(response.text)

            if places_result.get('places'):
                # Calculate distances for all places and sort them
                places_with_distances = []
                for place in places_result['places']:
                    place_lat = place['latitude']
                    place_lng = place['longitude']
                    distance = geodesic(coordinates, (place_lat, place_lng)).meters
                    if place_type == 'MRT/Subway Station':
                        place_info = place.get('type', "")
                        if place_info == "Subway station":
                            places_with_distances.append({
                                'name': place['title'],
                                'distance': distance,
                                'lat': place_lat,
                                'lng': place_lng
                            })
                    else:
                        places_with_distances.append({
                            'name': place['title'],
                            'distance': distance,
                            'lat': place_lat,
                            'lng': place_lng
                        })

                # Sort places by distance and get the closest one
                if not places_with_distances:
                    continue
                closest_place = sorted(places_with_distances, key=itemgetter('distance'))[0]
                if closest_place['distance'] > radius:
                    continue
                results[place_type] = closest_place

        return results

def get_building_name_onemap(postal_code: str) -> list:
    """
    Get the building name of an address in Singapore using the OneMap API.

    Args:
        address (str): The address to search.

    Returns:
        str: The building name if found, otherwise "Building name not found."
    """
    api_token = os.getenv('ONEMAP_API')
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={postal_code}&returnGeom=N&getAddrDetails=N"
    headers = {
        'Authorization': api_token,
    }

    response = requests.get(url, headers=headers)
    response_json = json.loads(response.text)
    building_names = []
    for result in response_json['results']:
        place_name = result["SEARCHVAL"]
        if not place_name.startswith("ATM"):
            building_names.append(place_name.title())

    return building_names