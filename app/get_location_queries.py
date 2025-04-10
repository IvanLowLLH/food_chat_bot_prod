import difflib
import json
from typing import Dict

class GetLocationSubzone:
    def __init__(self, area_file="area_to_subzone.json", subzone_file="sub_zone_nearby.json", match_cutoff=0.6):
        """"
        area_to_subzone is a dictionary where the key are Planning Areas, MRTs, shopping malls in Singapore and the value
        is the relevant subzone
        subzone_nearby is a dictionary where the keys are the sub-zones in Singapore and the values contain the
        sub-zones surrounding it
        """
        with open(area_file, "r", encoding="utf-8") as file:
            self.area_to_subzone = json.load(file)
        with open(subzone_file, "r", encoding="utf-8") as file:
            self.subzone_nearby = json.load(file)
        self.match_cutoff = match_cutoff

    def _find_closest_match(self, word: str, word_list: list) -> str:
        """Given a query word and a list of words, find a word in the list that matches the query word"""
        matches = difflib.get_close_matches(word, word_list, n=1, cutoff=self.match_cutoff)
        return matches[0] if matches else None

    def subzone_distance(self, base_zone, compare_zone):
        """Given a base-zone and another zone (compare-zone), get the distance between them that
        is already in subzone_nearby
        """
        base_zone = base_zone.lower()
        compare_zone = compare_zone.lower()
        nearby_base_zone = self.subzone_nearby[base_zone]['nearest_subzone']
        for nearby_zone_info in nearby_base_zone:
            name = nearby_zone_info[0]
            distance = nearby_zone_info[1]
            if name == compare_zone:
                return distance
        return None


    def find_subzones(self, location_query: str, max_dist: float ) -> Dict:
        """Given a location in Singapore, find the subzone it is in and nearby subzones"""
        location_query = location_query.lower()
        areas_places_list = self.area_to_subzone.keys()
        subzone_list = self.subzone_nearby.keys()
        # Find match in list of areas/places
        area_place_match = self._find_closest_match(location_query, areas_places_list)
        if area_place_match:
            subzone = self.area_to_subzone[area_place_match]
        else:  # If no match, find direct from list of sub-zones
            subzone = self._find_closest_match(location_query, subzone_list)

        result = {}
        if subzone:
            subzone_data = self.subzone_nearby[subzone]
            nearby_subzones = subzone_data["nearest_subzone"]
            result['nearby_subzones'] = []
            for nearby_subzone in nearby_subzones:
                name = nearby_subzone[0]
                distance = nearby_subzone[1]
                if distance <= max_dist:
                    result['nearby_subzones'].append(name.title())
        # Add original query at end. If no subzone match at all, then only return original query
        result['others'] = location_query

        return result
