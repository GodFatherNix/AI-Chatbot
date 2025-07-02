"""
Geospatial metadata parser for MOSDAC satellite data.
"""
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import math

from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

@dataclass
class GeospatialInfo:
    """Represents geospatial information extracted from content."""
    coordinates: List[Tuple[float, float]]
    bounding_box: Optional[Dict[str, float]]
    coordinate_system: str
    location_names: List[str]
    satellite_info: Dict[str, Any]
    temporal_info: Dict[str, Any]
    spatial_resolution: Optional[str]
    coverage_area: Optional[str]
    orbit_info: Dict[str, Any]

class GeospatialParser:
    """
    Parser for extracting geospatial metadata from MOSDAC content.
    Handles satellite data metadata, coordinates, and spatial information.
    """
    
    def __init__(self):
        self.coordinate_systems = config.get('geospatial.coordinate_systems', ['WGS84', 'UTM'])
        
        # Coordinate patterns
        self.coordinate_patterns = {
            'decimal_degrees': r'(-?\d+\.?\d*)\s*[°]?\s*([NS])\s*,?\s*(-?\d+\.?\d*)\s*[°]?\s*([EW])',
            'dms': r'(\d+)[°]\s*(\d+)[\'′]\s*(\d+\.?\d*)[\"″]?\s*([NS])\s*,?\s*(\d+)[°]\s*(\d+)[\'′]\s*(\d+\.?\d*)[\"″]?\s*([EW])',
            'simple_decimal': r'(-?\d+\.?\d+)\s*,\s*(-?\d+\.?\d+)',
            'bounded_coords': r'(?:lat|latitude):\s*(-?\d+\.?\d+).*?(?:lon|longitude):\s*(-?\d+\.?\d+)'
        }
        
        # Satellite name patterns
        self.satellite_patterns = {
            'indian_satellites': [
                r'\b(INSAT-?\w*)\b',
                r'\b(IRS-?\w*)\b', 
                r'\b(Oceansat-?\w*)\b',
                r'\b(ResourceSat-?\w*)\b',
                r'\b(CartoSat-?\w*)\b',
                r'\b(RISAT-?\w*)\b',
                r'\b(SARAL)\b',
                r'\b(Astrosat)\b',
                r'\b(Chandrayaan-?\w*)\b',
                r'\b(Mangalyaan)\b'
            ],
            'international_satellites': [
                r'\b(Landsat-?\w*)\b',
                r'\b(MODIS)\b',
                r'\b(AVHRR)\b',
                r'\b(VIIRS)\b',
                r'\b(Sentinel-?\w*)\b',
                r'\b(SPOT-?\w*)\b',
                r'\b(NOAA-?\w*)\b',
                r'\b(Terra)\b',
                r'\b(Aqua)\b'
            ]
        }
        
        # Spatial resolution patterns
        self.resolution_patterns = [
            r'(\d+\.?\d*)\s*(?:meter|metre|m)\s*(?:resolution|pixel)',
            r'(\d+\.?\d*)\s*(?:km|kilometer|kilometre)\s*(?:resolution|pixel)',
            r'resolution.*?(\d+\.?\d*)\s*(?:m|meter|metre)',
            r'pixel.*?(\d+\.?\d*)\s*(?:m|meter|metre)'
        ]
        
        # Temporal patterns
        self.temporal_patterns = {
            'date_iso': r'\b(\d{4}-\d{2}-\d{2})\b',
            'date_slash': r'\b(\d{2}/\d{2}/\d{4})\b',
            'date_dot': r'\b(\d{2}\.\d{2}\.\d{4})\b',
            'year': r'\b(19|20)\d{2}\b',
            'temporal_frequency': r'\b(daily|weekly|monthly|yearly|annual|seasonal)\b'
        }
        
        # Orbit patterns
        self.orbit_patterns = [
            r'orbit.*?(\d+\.?\d*)\s*(?:km|degree)',
            r'(?:ascending|descending)\s*(?:pass|node)',
            r'(?:polar|sun-synchronous|geostationary|geosynchronous)\s*orbit'
        ]
        
        # Location name patterns (common Indian locations)
        self.location_patterns = [
            r'\b(India|Bharat)\b',
            r'\b(Kashmir|Punjab|Haryana|Rajasthan|Gujarat|Maharashtra|Karnataka|Kerala|Tamil Nadu|Andhra Pradesh|Telangana|Odisha|West Bengal|Bihar|Jharkhand|Uttarakhand|Himachal Pradesh|Jammu)\b',
            r'\b(Delhi|Mumbai|Kolkata|Chennai|Bangalore|Hyderabad|Pune|Ahmedabad|Surat|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Bhopal|Visakhapatnam|Pimpri)\b',
            r'\b(Arabian Sea|Bay of Bengal|Indian Ocean|Himalaya|Western Ghats|Eastern Ghats|Deccan Plateau|Indo-Gangetic Plain)\b'
        ]
    
    def _parse_coordinates(self, text: str) -> List[Tuple[float, float]]:
        """Extract coordinate pairs from text."""
        coordinates = []
        
        for pattern_name, pattern in self.coordinate_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    if pattern_name == 'decimal_degrees':
                        lat, lat_dir, lon, lon_dir = match.groups()
                        lat_val = float(lat) * (-1 if lat_dir.upper() == 'S' else 1)
                        lon_val = float(lon) * (-1 if lon_dir.upper() == 'W' else 1)
                        
                    elif pattern_name == 'dms':
                        lat_d, lat_m, lat_s, lat_dir, lon_d, lon_m, lon_s, lon_dir = match.groups()
                        lat_val = int(lat_d) + int(lat_m)/60 + float(lat_s)/3600
                        lon_val = int(lon_d) + int(lon_m)/60 + float(lon_s)/3600
                        lat_val *= (-1 if lat_dir.upper() == 'S' else 1)
                        lon_val *= (-1 if lon_dir.upper() == 'W' else 1)
                        
                    elif pattern_name == 'simple_decimal':
                        lat_val, lon_val = map(float, match.groups())
                        
                    elif pattern_name == 'bounded_coords':
                        lat_val, lon_val = map(float, match.groups())
                    
                    # Validate coordinates
                    if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                        coordinates.append((lat_val, lon_val))
                        
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing coordinates: {e}")
                    continue
        
        return coordinates
    
    def _calculate_bounding_box(self, coordinates: List[Tuple[float, float]]) -> Optional[Dict[str, float]]:
        """Calculate bounding box from coordinate list."""
        if not coordinates:
            return None
        
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        return {
            'min_latitude': min(lats),
            'max_latitude': max(lats),
            'min_longitude': min(lons),
            'max_longitude': max(lons),
            'center_latitude': sum(lats) / len(lats),
            'center_longitude': sum(lons) / len(lons)
        }
    
    def _extract_satellite_info(self, text: str) -> Dict[str, Any]:
        """Extract satellite information from text."""
        satellite_info = {
            'names': [],
            'missions': [],
            'sensors': [],
            'agencies': []
        }
        
        # Extract satellite names
        for category, patterns in self.satellite_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    satellite_info['names'].append({
                        'name': match,
                        'category': category
                    })
        
        # Extract agencies
        agency_patterns = [r'\b(ISRO|NASA|ESA|NOAA|JAXA|CSA)\b']
        for pattern in agency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            satellite_info['agencies'].extend(matches)
        
        # Extract sensors
        sensor_patterns = [
            r'\b(LISS|WiFS|AWiFS|PAN|MSIL|SWIR|TIR|MIR)\b',
            r'\b(MODIS|AVHRR|VIIRS|OLI|TIRS|ETM\+|TM)\b'
        ]
        for pattern in sensor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            satellite_info['sensors'].extend(matches)
        
        # Remove duplicates
        for key in satellite_info:
            if isinstance(satellite_info[key], list):
                satellite_info[key] = list(set(satellite_info[key]))
        
        return satellite_info
    
    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from text."""
        temporal_info = {
            'dates': [],
            'years': [],
            'frequency': [],
            'time_range': None
        }
        
        # Extract dates and years
        for pattern_name, pattern in self.temporal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            if pattern_name in ['date_iso', 'date_slash', 'date_dot']:
                temporal_info['dates'].extend(matches)
            elif pattern_name == 'year':
                temporal_info['years'].extend([match[0] + match[1] for match in matches])
            elif pattern_name == 'temporal_frequency':
                temporal_info['frequency'].extend(matches)
        
        # Determine time range
        if temporal_info['dates']:
            try:
                dates = [datetime.strptime(date, '%Y-%m-%d') for date in temporal_info['dates']
                        if re.match(r'\d{4}-\d{2}-\d{2}', date)]
                if dates:
                    temporal_info['time_range'] = {
                        'start': min(dates).isoformat(),
                        'end': max(dates).isoformat()
                    }
            except ValueError:
                pass
        
        # Remove duplicates
        for key in temporal_info:
            if isinstance(temporal_info[key], list):
                temporal_info[key] = list(set(temporal_info[key]))
        
        return temporal_info
    
    def _extract_spatial_resolution(self, text: str) -> Optional[str]:
        """Extract spatial resolution information."""
        for pattern in self.resolution_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def _extract_orbit_info(self, text: str) -> Dict[str, Any]:
        """Extract orbit information from text."""
        orbit_info = {
            'type': [],
            'altitude': [],
            'inclination': [],
            'pass_type': []
        }
        
        # Orbit type
        orbit_types = re.findall(
            r'\b(polar|sun-synchronous|geostationary|geosynchronous|elliptical)\s*orbit',
            text, re.IGNORECASE
        )
        orbit_info['type'] = list(set(orbit_types))
        
        # Pass type
        pass_types = re.findall(
            r'\b(ascending|descending)\s*(?:pass|node)',
            text, re.IGNORECASE
        )
        orbit_info['pass_type'] = list(set(pass_types))
        
        # Altitude
        altitude_matches = re.findall(
            r'(?:altitude|height).*?(\d+\.?\d*)\s*(?:km|kilometer)',
            text, re.IGNORECASE
        )
        orbit_info['altitude'] = [float(alt) for alt in altitude_matches]
        
        return orbit_info
    
    def _extract_location_names(self, text: str) -> List[str]:
        """Extract geographic location names."""
        locations = []
        
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            locations.extend(matches)
        
        return list(set(locations))
    
    def _determine_coordinate_system(self, text: str) -> str:
        """Determine the coordinate system used."""
        coordinate_systems = ['WGS84', 'UTM', 'Geographic', 'Projected']
        
        for system in coordinate_systems:
            if system.lower() in text.lower():
                return system
        
        return 'WGS84'  # Default assumption
    
    def _calculate_coverage_area(self, coordinates: List[Tuple[float, float]]) -> Optional[str]:
        """Estimate coverage area from coordinates."""
        if len(coordinates) < 2:
            return None
        
        # Simple bounding box area calculation
        bbox = self._calculate_bounding_box(coordinates)
        if not bbox:
            return None
        
        lat_diff = bbox['max_latitude'] - bbox['min_latitude']
        lon_diff = bbox['max_longitude'] - bbox['min_longitude']
        
        # Rough area calculation (not accurate for large areas)
        area_degrees = lat_diff * lon_diff
        
        if area_degrees < 1:
            return "Local"
        elif area_degrees < 100:
            return "Regional"
        elif area_degrees < 1000:
            return "National"
        else:
            return "Continental"
    
    def parse_geospatial_content(self, content: str, metadata: Dict = None) -> GeospatialInfo:
        """
        Parse geospatial information from content.
        
        Args:
            content: Text content to parse
            metadata: Additional metadata dictionary
            
        Returns:
            GeospatialInfo object with extracted spatial information
        """
        if metadata is None:
            metadata = {}
        
        # Combine content and metadata for parsing
        full_text = content
        if metadata:
            full_text += " " + json.dumps(metadata, default=str)
        
        # Extract various geospatial components
        coordinates = self._parse_coordinates(full_text)
        bounding_box = self._calculate_bounding_box(coordinates)
        coordinate_system = self._determine_coordinate_system(full_text)
        location_names = self._extract_location_names(full_text)
        satellite_info = self._extract_satellite_info(full_text)
        temporal_info = self._extract_temporal_info(full_text)
        spatial_resolution = self._extract_spatial_resolution(full_text)
        coverage_area = self._calculate_coverage_area(coordinates)
        orbit_info = self._extract_orbit_info(full_text)
        
        return GeospatialInfo(
            coordinates=coordinates,
            bounding_box=bounding_box,
            coordinate_system=coordinate_system,
            location_names=location_names,
            satellite_info=satellite_info,
            temporal_info=temporal_info,
            spatial_resolution=spatial_resolution,
            coverage_area=coverage_area,
            orbit_info=orbit_info
        )
    
    def validate_geospatial_info(self, geo_info: GeospatialInfo) -> Dict[str, Any]:
        """Validate extracted geospatial information."""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'completeness_score': 0
        }
        
        # Check coordinates
        if geo_info.coordinates:
            validation['completeness_score'] += 20
            for lat, lon in geo_info.coordinates:
                if not (-90 <= lat <= 90):
                    validation['errors'].append(f"Invalid latitude: {lat}")
                    validation['is_valid'] = False
                if not (-180 <= lon <= 180):
                    validation['errors'].append(f"Invalid longitude: {lon}")
                    validation['is_valid'] = False
        else:
            validation['warnings'].append("No coordinates found")
        
        # Check satellite info
        if geo_info.satellite_info.get('names'):
            validation['completeness_score'] += 20
        else:
            validation['warnings'].append("No satellite information found")
        
        # Check temporal info
        if geo_info.temporal_info.get('dates'):
            validation['completeness_score'] += 20
        else:
            validation['warnings'].append("No temporal information found")
        
        # Check spatial resolution
        if geo_info.spatial_resolution:
            validation['completeness_score'] += 20
        else:
            validation['warnings'].append("No spatial resolution found")
        
        # Check location names
        if geo_info.location_names:
            validation['completeness_score'] += 20
        else:
            validation['warnings'].append("No location names found")
        
        return validation
    
    def to_dict(self, geo_info: GeospatialInfo) -> Dict[str, Any]:
        """Convert GeospatialInfo to dictionary."""
        return {
            'coordinates': geo_info.coordinates,
            'bounding_box': geo_info.bounding_box,
            'coordinate_system': geo_info.coordinate_system,
            'location_names': geo_info.location_names,
            'satellite_info': geo_info.satellite_info,
            'temporal_info': geo_info.temporal_info,
            'spatial_resolution': geo_info.spatial_resolution,
            'coverage_area': geo_info.coverage_area,
            'orbit_info': geo_info.orbit_info
        }