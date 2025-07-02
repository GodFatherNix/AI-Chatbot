#!/usr/bin/env python3
"""Geospatial query service for MOSDAC satellite data.

Handles bounding box queries, region-based filtering, and spatial context
for satellite missions and products.
"""

import json
from typing import List, Dict, Optional, Tuple
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import geojson
from dataclasses import dataclass


@dataclass
class SpatialQuery:
    """Represents a spatial query with bounding box or polygon."""
    geometry: Dict
    mission: Optional[str] = None
    product_type: Optional[str] = None
    date_range: Optional[Tuple[str, str]] = None


@dataclass
class SpatialResult:
    """Spatial query result with metadata."""
    product_id: str
    mission: str
    product_type: str
    coverage_area: Dict
    download_url: str
    spatial_resolution: str
    temporal_coverage: str


class GeospatialService:
    """Service for handling geospatial queries and filtering."""

    def __init__(self):
        # Sample MOSDAC mission coverage data
        self.mission_coverage = {
            "INSAT-3D": {
                "coverage": [-180, -60, 180, 60],  # [minx, miny, maxx, maxy]
                "resolution": "4km",
                "products": ["Temperature", "Humidity", "Cloud Imagery"]
            },
            "OCEANSAT-2": {
                "coverage": [-180, -90, 180, 90],
                "resolution": "360m",
                "products": ["Ocean Color", "SST", "Chlorophyll"]
            },
            "SCATSAT-1": {
                "coverage": [-180, -90, 180, 90],
                "resolution": "25km",
                "products": ["Wind Speed", "Wind Direction"]
            },
            "MEGHA-TROPIQUES": {
                "coverage": [-180, -30, 180, 30],
                "resolution": "10km",
                "products": ["Precipitation", "Water Vapor", "Cloud Properties"]
            }
        }

    def query_by_bbox(self, bbox: List[float], **filters) -> List[SpatialResult]:
        """Query satellite products by bounding box.
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            **filters: mission, product_type, date_range
        """
        query_polygon = box(*bbox)
        results = []
        
        for mission, data in self.mission_coverage.items():
            # Check if mission filter matches
            if filters.get('mission') and mission != filters['mission']:
                continue
                
            # Check spatial overlap
            mission_bbox = box(*data['coverage'])
            if not query_polygon.intersects(mission_bbox):
                continue
                
            # Generate sample results for intersecting missions
            for product in data['products']:
                if filters.get('product_type') and product != filters['product_type']:
                    continue
                    
                result = SpatialResult(
                    product_id=f"{mission}_{product}_{hash(str(bbox)) % 1000}",
                    mission=mission,
                    product_type=product,
                    coverage_area={
                        "type": "Polygon",
                        "coordinates": [[
                            [data['coverage'][0], data['coverage'][1]],
                            [data['coverage'][2], data['coverage'][1]],
                            [data['coverage'][2], data['coverage'][3]],
                            [data['coverage'][0], data['coverage'][3]],
                            [data['coverage'][0], data['coverage'][1]]
                        ]]
                    },
                    download_url=f"https://mosdac.gov.in/data/{mission}/{product}",
                    spatial_resolution=data['resolution'],
                    temporal_coverage="2020-01-01 to 2024-12-31"
                )
                results.append(result)
                
        return results

    def query_by_location(self, lat: float, lon: float, **filters) -> List[SpatialResult]:
        """Query products covering a specific point location."""
        point = Point(lon, lat)
        results = []
        
        for mission, data in self.mission_coverage.items():
            mission_bbox = box(*data['coverage'])
            if point.within(mission_bbox):
                for product in data['products']:
                    if filters.get('product_type') and product != filters['product_type']:
                        continue
                        
                    result = SpatialResult(
                        product_id=f"{mission}_{product}_point_{int(lat*100)}_{int(lon*100)}",
                        mission=mission,
                        product_type=product,
                        coverage_area={
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        download_url=f"https://mosdac.gov.in/data/{mission}/{product}",
                        spatial_resolution=data['resolution'],
                        temporal_coverage="2020-01-01 to 2024-12-31"
                    )
                    results.append(result)
                    
        return results

    def get_mission_coverage_geojson(self) -> Dict:
        """Return mission coverage as GeoJSON for map visualization."""
        features = []
        
        for mission, data in self.mission_coverage.items():
            bbox = data['coverage']
            feature = {
                "type": "Feature",
                "properties": {
                    "mission": mission,
                    "resolution": data['resolution'],
                    "products": data['products']
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]]
                    ]]
                }
            }
            features.append(feature)
            
        return {
            "type": "FeatureCollection",
            "features": features
        }

    def extract_spatial_context(self, query_text: str) -> Optional[Dict]:
        """Extract spatial information from natural language query."""
        import re
        
        # Simple regex patterns for location extraction
        patterns = {
            'coordinates': r'(\-?\d+\.?\d*),\s*(\-?\d+\.?\d*)',
            'bbox': r'bbox\s*:\s*(\[[\d\.,\s\-]+\])',
            'region': r'(India|Arabian Sea|Bay of Bengal|Indian Ocean)',
            'state': r'(Kerala|Tamil Nadu|Karnataka|Andhra Pradesh|Odisha|West Bengal)'
        }
        
        spatial_info = {}
        
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            if matches:
                spatial_info[pattern_type] = matches
                
        return spatial_info if spatial_info else None


# Example usage
if __name__ == "__main__":
    service = GeospatialService()
    
    # Query by bounding box (Bay of Bengal region)
    bbox = [80, 10, 95, 22]  # [min_lon, min_lat, max_lon, max_lat]
    results = service.query_by_bbox(bbox, mission="OCEANSAT-2")
    
    print(f"Found {len(results)} products for Bay of Bengal region:")
    for result in results:
        print(f"- {result.mission}: {result.product_type} ({result.spatial_resolution})")
    
    # Query by point location (Chennai)
    results = service.query_by_location(13.0827, 80.2707)
    print(f"\nProducts covering Chennai: {len(results)}")
    
    # Get coverage map
    coverage = service.get_mission_coverage_geojson()
    print(f"\nMission coverage GeoJSON has {len(coverage['features'])} features")