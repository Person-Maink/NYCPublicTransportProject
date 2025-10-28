import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from haversine import haversine, Unit
from typing import Tuple, List, Dict, Optional
import warnings


class StationFinder:
    """
    Utility class for working with Citibike and NYC Subway station networks.
    Provides functionality to:
    - Map Citibike stations to their closest subway stations
    - Find all stations within a radius of a given point
    - Generate random points within a radius
    """
    
    def __init__(self, citibike_graph_path: str, subway_graph_path: str, 
                 subway_coords_path: Optional[str] = None):
        """
        Initialize the StationFinder with graph paths.
        
        Parameters:
        -----------
        citibike_graph_path : str
            Path to the Citibike network GML file
        subway_graph_path : str
            Path to the subway network GML file
        subway_coords_path : str, optional
            Path to CSV file with subway coordinates (columns: station_name, latitude, longitude)
            If not provided, will attempt to download from MTA GTFS
        """
        self.citibike_graph_path = citibike_graph_path
        self.subway_graph_path = subway_graph_path
        self.subway_coords_path = subway_coords_path
        
        # Load graphs
        self.citibike_graph = self._load_citibike_graph()
        self.subway_graph = self._load_subway_graph()
        
        # Extract station coordinates
        self.citibike_stations = self._extract_citibike_coords()
        self.subway_stations = self._extract_subway_coords()
        self.original_subway_station_names = set(self.subway_stations['station_name'].values)
        
    def _load_citibike_graph(self) -> nx.DiGraph:
        """Load the Citibike graph from GML file."""
        return nx.read_gml(self.citibike_graph_path)
    
    def _load_subway_graph(self) -> nx.Graph:
        """Load the subway graph from GML file."""
        return nx.read_gml(self.subway_graph_path)
    
    def _extract_citibike_coords(self) -> pd.DataFrame:
        """
        Extract Citibike station coordinates from the graph.
        Returns DataFrame with columns: station_name, latitude, longitude
        """
        stations = []
        for node, data in self.citibike_graph.nodes(data=True):
            stations.append({
                'station_name': node,
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude')
            })
        
        df = pd.DataFrame(stations)
        # Remove any stations with missing coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        return df
    
    def _extract_subway_coords(self) -> pd.DataFrame:
        """
        Extract subway station coordinates.
        If subway_coords_path is provided, load from CSV.
        Otherwise, attempt to download from MTA GTFS or use embedded dataset.
        """
        if self.subway_coords_path and Path(self.subway_coords_path).exists():
            # Load from provided CSV
            df = pd.read_csv(self.subway_coords_path)
            # Ensure required columns exist
            required_cols = ['station_name', 'latitude', 'longitude']
            if not all(col in df.columns for col in required_cols):
                # Try alternative column names
                if 'stop_name' in df.columns and 'stop_lat' in df.columns and 'stop_lon' in df.columns:
                    df = df.rename(columns={
                        'stop_name': 'station_name',
                        'stop_lat': 'latitude',
                        'stop_lon': 'longitude'
                    })
                else:
                    raise ValueError(f"CSV must contain columns: {required_cols}")
            return df[['station_name', 'latitude', 'longitude']].drop_duplicates()
        else:
            # Attempt to download from MTA GTFS
            return self._download_subway_coords()
    
    def _download_subway_coords(self) -> pd.DataFrame:
        """
        Download NYC subway station coordinates from MTA GTFS feed.
        Falls back to embedded dataset if download fails.
        """
        try:
            # MTA GTFS URL for subway stops
            url = "http://web.mta.info/developers/data/nyct/subway/google_transit.zip"
            
            import io
            import zipfile
            import urllib.request
            
            print("Downloading NYC subway coordinates from MTA GTFS...")
            response = urllib.request.urlopen(url, timeout=30)
            zip_data = io.BytesIO(response.read())
            
            with zipfile.ZipFile(zip_data) as z:
                with z.open('stops.txt') as f:
                    stops_df = pd.read_csv(f)
            
            # Filter for stations only (not individual platforms)
            # Station IDs without directional suffix (N/S)
            stops_df = stops_df[~stops_df['stop_id'].str.endswith(('N', 'S'))]
            
            df = stops_df[['stop_name', 'stop_lat', 'stop_lon']].copy()
            df.columns = ['station_name', 'latitude', 'longitude']
            df = df.drop_duplicates(subset=['station_name'])
            
            print(f"Successfully loaded {len(df)} subway stations")
            return df
            
        except Exception as e:
            warnings.warn(f"Could not download subway coordinates: {e}. "
                        "Please provide subway_coords_path parameter.")
            raise ValueError(
                "Subway coordinates not available. Please provide a CSV file with "
                "columns: station_name, latitude, longitude via subway_coords_path parameter."
            )
    
    def create_citibike_to_subway_mapping(self, output_path: str = 'src/outputs/citibike_to_subway_mapping.csv') -> pd.DataFrame:
        """
        Create a lookup table mapping each Citibike station to its closest subway station.
        
        Parameters:
        -----------
        output_path : str
            Path to save the mapping CSV file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: citibike_station, citibike_lat, citibike_lng,
                                   closest_subway_station, subway_lat, subway_lng, distance_km
        """
        mappings = []
        
        for _, citibike_row in self.citibike_stations.iterrows():
            cb_point = (citibike_row['latitude'], citibike_row['longitude'])
            
            # Calculate distance to all subway stations
            min_distance = float('inf')
            closest_subway = None
            closest_coords = None
            
            for _, subway_row in self.subway_stations.iterrows():
                subway_point = (subway_row['latitude'], subway_row['longitude'])
                distance = haversine(cb_point, subway_point, unit=Unit.KILOMETERS)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_subway = subway_row['station_name']
                    closest_coords = subway_point
            
            mappings.append({
                'citibike_station': citibike_row['station_name'],
                'citibike_lat': citibike_row['latitude'],
                'citibike_lng': citibike_row['longitude'],
                'closest_subway_station': closest_subway,
                'subway_lat': closest_coords[0] if closest_coords else None,
                'subway_lng': closest_coords[1] if closest_coords else None,
                'distance_km': min_distance
            })
        
        mapping_df = pd.DataFrame(mappings)
        
        # Save to CSV
        mapping_df.to_csv(output_path, index=False)
        print(f"Saved Citibike to Subway mapping to {output_path}")
        print(f"Total mappings: {len(mapping_df)}")
        print(f"Average distance to closest subway: {mapping_df['distance_km'].mean():.3f} km")
        print(f"Max distance to closest subway: {mapping_df['distance_km'].max():.3f} km")
        
        return mapping_df
    
    def find_stations_within_radius(self, latitude: float, longitude: float, 
                                    radius_km: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Find all Citibike and subway stations within a given radius of a point.
        
        Parameters:
        -----------
        latitude : float
            Latitude of the center point
        longitude : float
            Longitude of the center point
        radius_km : float
            Radius in kilometers (default: 1.0 km)
            
        Returns:
        --------
        dict
            Dictionary with keys 'citibike' and 'subway', each containing a DataFrame
            with columns: station_name, latitude, longitude, distance_km
        """
        center_point = (latitude, longitude)
        
        # Find Citibike stations within radius
        citibike_within = []
        for _, row in self.citibike_stations.iterrows():
            station_point = (row['latitude'], row['longitude'])
            distance = haversine(center_point, station_point, unit=Unit.KILOMETERS)
            
            if distance <= radius_km:
                citibike_within.append({
                    'station_name': row['station_name'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'distance_km': distance
                })
        
        # Find subway stations within radius
        subway_within = []
        for _, row in self.subway_stations.iterrows():
            station_point = (row['latitude'], row['longitude'])
            distance = haversine(center_point, station_point, unit=Unit.KILOMETERS)
            
            if distance <= radius_km:
                subway_within.append({
                    'station_name': row['station_name'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'distance_km': distance
                })
        
        citibike_df = pd.DataFrame(citibike_within).sort_values('distance_km') if citibike_within else pd.DataFrame()
        subway_df = pd.DataFrame(subway_within).sort_values('distance_km') if subway_within else pd.DataFrame()
        
        return {
            'citibike': citibike_df,
            'subway': subway_df
        }
    
    def generate_random_points_in_circle(self, latitude: float, longitude: float, 
                                         radius_km: float = 1.0, 
                                         num_points: int = 10) -> pd.DataFrame:
        """
        Generate random points uniformly distributed within a circle.
        
        Parameters:
        -----------
        latitude : float
            Latitude of the center point
        longitude : float
            Longitude of the center point
        radius_km : float
            Radius in kilometers (default: 1.0 km)
        num_points : int
            Number of random points to generate (default: 10)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: point_id, latitude, longitude, distance_from_center_km
        """
        # Earth's radius in km
        earth_radius_km = 6371.0
        
        # Convert radius to degrees (approximate)
        radius_deg = (radius_km / earth_radius_km) * (180 / np.pi)
        
        points = []
        for i in range(num_points):
            # Generate random radius and angle
            # Use sqrt for uniform distribution in circle
            r = radius_deg * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()
            
            # Convert polar to Cartesian, accounting for latitude compression
            # At higher latitudes, longitude degrees are shorter
            lat_offset = r * np.cos(theta)
            lng_offset = r * np.sin(theta) / np.cos(np.radians(latitude))
            
            point_lat = latitude + lat_offset
            point_lng = longitude + lng_offset
            
            # Calculate actual distance
            distance = haversine((latitude, longitude), (point_lat, point_lng), unit=Unit.KILOMETERS)
            
            points.append({
                'point_id': i + 1,
                'latitude': point_lat,
                'longitude': point_lng,
                'distance_from_center_km': distance
            })
        
        return pd.DataFrame(points)


def main():
    """
    Example usage of the StationFinder class.
    """
    import sys
    
    # Paths to graph files
    citibike_weekday = 'citibike_weekday_network.gml'
    citibike_weekend = 'citibike_weekend_network.gml'
    subway_graph = 'subway_graph_weekday_weekend.gml'
    
    # Check which Citibike graph to use
    citibike_path = citibike_weekday if Path(citibike_weekday).exists() else citibike_weekend
    
    if not Path(citibike_path).exists():
        print("Error: Citibike graph not found. Please run citibike_processor.py first.")
        sys.exit(1)
    
    if not Path(subway_graph).exists():
        print(f"Error: Subway graph not found at {subway_graph}")
        sys.exit(1)
    
    print("Initializing StationFinder...")
    try:
        finder = StationFinder(
            citibike_graph_path=citibike_path,
            subway_graph_path=subway_graph
        )
    except Exception as e:
        print(f"Error initializing StationFinder: {e}")
        print("\nIf subway coordinates are not available, you can:")
        print("1. Provide a CSV file with subway coordinates")
        print("2. Download MTA GTFS data manually from: http://web.mta.info/developers/data/nyct/subway/google_transit.zip")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("TASK 1: Creating Citibike to Subway station mapping")
    print("="*80)
    mapping = finder.create_citibike_to_subway_mapping()
    print("\nFirst 10 mappings:")
    print(mapping.head(10))
    
    print("\n" + "="*80)
    print("TASK 2: Finding stations within 1 km radius")
    print("="*80)
    # Example: Times Square area (approximate coordinates)
    lat, lng = 40.7580, -73.9855
    print(f"Center point: ({lat}, {lng})")
    stations = finder.find_stations_within_radius(lat, lng, radius_km=1.0)
    
    print(f"\nCitibike stations within 1 km: {len(stations['citibike'])}")
    if len(stations['citibike']) > 0:
        print(stations['citibike'].head())
    
    print(f"\nSubway stations within 1 km: {len(stations['subway'])}")
    if len(stations['subway']) > 0:
        print(stations['subway'].head())
    
    print("\n" + "="*80)
    print("TASK 3: Generating random points within 1 km radius")
    print("="*80)
    random_points = finder.generate_random_points_in_circle(lat, lng, radius_km=1.0, num_points=10)
    print(random_points)
    
    # Save random points to CSV
    random_points.to_csv('random_points_example.csv', index=False)
    print("\nSaved random points to random_points_example.csv")


if __name__ == "__main__":
    main()

