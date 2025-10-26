import sys
import folium
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from haversine import haversine, Unit
from typing import Tuple, List, Dict, Optional
import warnings
import haversine as hs
from tqdm import tqdm
import googlemaps
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

from station_utilities import StationFinder

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


class Simulator:
    """
    A class to simulate travel times and analyze public transport networks in NYC.

    This simulator can find nearby stations, generate random points for analysis,
    and calculate travel times between two points using different modes of transport
    like walking, biking, and the subway system.
    
    Optimized version with:
    - Pre-calculated nearest stations
    - Batched API calls using Distance Matrix API
    - Parallelization support
    """
    
    def __init__(self, finder: StationFinder):
        """
        Initializes the Simulator, loading station data and setting constants.
        """
        self.gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        self.WALK_SPEED = 5  # Kilometers per Hour
        self.BIKE_SPEED = 18 # Kilometers per Hour
        self.METRO_SPEED = 28 # Kilometers per Hour
        
        # Initialize StationFinder to load graph data
        self.finder = finder
        
        # Extract station coordinates from the finder for easy access
        self.citibike_stations = self.finder.citibike_stations
        self.subway_stations = self.finder.subway_stations

    
    def get_nearby_stations(self, location: Tuple[float, float], radius_km: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Finds all citibike and subway stations within a given radius of a location.

        Args:
            location (Tuple[float, float]): The (latitude, longitude) of the center point.
            radius_km (float, optional): The search radius in kilometers. Defaults to 1.0.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing DataFrames for 'citibike'
                                     and 'subway' stations found within the radius.
        """
        lat, lng = location[0], location[1]
        stations = self.finder.find_stations_within_radius(lat, lng, radius_km=radius_km)
        return stations
    

    def pre_calculate_nearest_stations(self, points: pd.DataFrame, mode: str, radius_km: float = 1.0) -> List[Optional[Tuple[float, float]]]:
        """
        Pre-calculates the nearest station for all points.

        Args:
            points (pd.DataFrame): DataFrame with 'latitude' and 'longitude' columns.
            mode (str): The mode of transportation ('citibike' or 'subway').
            radius_km (float): The search radius for nearby stations.

        Returns:
            List[Optional[Tuple[float, float]]]: List of nearest station coordinates for each point.
                                                   None if no station is found.
        """
        nearest_stations = []
        
        for _, row in points.iterrows():
            location = (row['latitude'], row['longitude'])
            try:
                station = self.get_nearby_stations(location, radius_km=radius_km)[mode].iloc[0]
                station_coord = (station['latitude'], station['longitude'])
                nearest_stations.append(station_coord)
            except IndexError:
                print(f"Warning: No {mode} station found within {radius_km}km for point {location}")
                nearest_stations.append(None)
        
        return nearest_stations
    
    
    def generate_random_points(self, location: Tuple[float, float], radius_km: float = 1.0, num_points: int = 10) -> pd.DataFrame:
        """
        Generates a specified number of random geographic points within a radius of a location.

        Args:
            location (Tuple[float, float]): The (latitude, longitude) of the center point.
            radius_km (float, optional): The radius in kilometers. Defaults to 1.0.
            num_points (int, optional): The number of random points to generate. Defaults to 10.

        Returns:
            pd.DataFrame: A DataFrame with the generated random points' coordinates.
        """
        random_points = self.finder.generate_random_points_in_circle(location[0], location[1], radius_km=radius_km, num_points=num_points)
        return random_points
    
    
    def insert_station(self, station: Tuple[str, Tuple[float, float]]) -> None:
        """
        Inserts a station into the stations dataframe.

        Args:
            station (tuple): A tuple containing the station name and its (latitude, longitude).
        """
        station_name, station_coords = station
        if station_name not in self.subway_stations:
            print(f"Inserting new station: {station_name} at {station_coords}")
            self.subway_stations.loc[len(self.subway_stations)] = [station_name, station_coords[0], station_coords[1]]
    
    
    def batch_distance_matrix_call(self, origins: List[Tuple[float, float]], 
                                   destinations: List[Tuple[float, float]], 
                                   mode: str = 'walking',
                                   max_elements: int = 100) -> np.ndarray:
        """
        Makes batched calls to Google Maps Distance Matrix API.
        
        The API has limits: max 25 origins x 25 destinations = 625 elements per request.
        We use a conservative default of 100 elements and batch accordingly.

        Args:
            origins (List[Tuple[float, float]]): List of origin coordinates.
            destinations (List[Tuple[float, float]]): List of destination coordinates.
            mode (str): Travel mode ('walking', 'bicycling', 'driving', 'transit').
            max_elements (int): Maximum elements per API call (origins * destinations).

        Returns:
            np.ndarray: Matrix of travel times in seconds (shape: len(origins) x len(destinations)).
        """
        num_origins = len(origins)
        num_destinations = len(destinations)
        time_matrix = np.full((num_origins, num_destinations), np.nan)
        
        # Calculate batch sizes
        max_origins_per_batch = min(25, int(np.sqrt(max_elements)))
        max_dests_per_batch = min(25, max_elements // max_origins_per_batch)
        
        # Batch the requests
        for i in range(0, num_origins, max_origins_per_batch):
            for j in range(0, num_destinations, max_dests_per_batch):
                batch_origins = origins[i:i + max_origins_per_batch]
                batch_dests = destinations[j:j + max_dests_per_batch]
                
                try:
                    result = self.gmaps.distance_matrix(
                        origins=batch_origins,
                        destinations=batch_dests,
                        mode=mode
                    )
                    
                    # Parse the results
                    for oi, origin_result in enumerate(result['rows']):
                        for di, element in enumerate(origin_result['elements']):
                            if element['status'] == 'OK':
                                time_matrix[i + oi, j + di] = element['duration']['value']
                            else:
                                print(f"Warning: No route found for origin {i+oi} to dest {j+di}")
                                
                except Exception as e:
                    print(f"Error in batch Distance Matrix call: {e}")
                    
        return time_matrix
    
    
    def calculate_travel_time(self, src: Tuple[float, float], dst: Tuple[float, float], 
                             mode: str = 'citibike', radius_km: float = 1.0) -> Optional[Tuple[float, List[Tuple[float, float]]]]:
        """
        Calculate the total travel time between source and destination with the selected mode of transportation.

        The travel time includes:
        1. Walking from the source to the nearest station (via Google Maps).
        2. Traveling between the nearest source and destination stations (via Google Maps for bike, via haversine for subway).
        3. Walking from the destination station to the final destination (via Google Maps).

        Args:
            src (tuple): The (latitude, longitude) of the starting point.
            dst (tuple): The (latitude, longitude) of the destination point.
            mode (str): The mode of transportation ('citibike' or 'subway').
            radius_km (float): Search radius for stations.

        Returns:
            Optional[tuple]: A tuple containing:
                                - The total travel time (in seconds).
                                - A list of coordinates representing the route waypoints.
                                Returns None if the mode is invalid or stations/routes cannot be found.
        """
        if mode not in ('citibike', 'subway'):
            warnings.warn("Invalid mode of transportation specified. Choose 'citibike' or 'subway'.")
            return None
        
        try:
            # Get the nearest stations
            src_station = self.get_nearby_stations(src, radius_km=radius_km)[mode].iloc[0]
            dst_station = self.get_nearby_stations(dst, radius_km=radius_km)[mode].iloc[0]
        except IndexError:
            print(f"Could not find a start or end station for mode '{mode}' within {radius_km}km. Aborting.")
            return None

        src_station_coord = (src_station['latitude'], src_station['longitude'])
        dst_station_coord = (dst_station['latitude'], dst_station['longitude'])

        # The key waypoints of the journey
        route_points = [src, src_station_coord, dst_station_coord, dst]

        total_time_seconds = 0
        
        try:
            # --- Leg 1: Walk from src to src_station ---
            walk_1_result = self.gmaps.directions(src, src_station_coord, mode="walking")
            if not walk_1_result:
                raise Exception(f"Google Maps could not find walking route for Leg 1: {src} to {src_station_coord}")
            total_time_seconds += walk_1_result[0]['legs'][0]['duration']['value']

            # --- Leg 2: Transit (Bike or Subway) ---
            if mode == 'citibike':
                bike_result = self.gmaps.directions(src_station_coord, dst_station_coord, mode="bicycling")
                if not bike_result:
                    raise Exception(f"Google Maps could not find biking route for Leg 2: {src_station_coord} to {dst_station_coord}")
                total_time_seconds += bike_result[0]['legs'][0]['duration']['value']
                
            elif mode == 'subway':
                transit_dist = hs.haversine(src_station_coord, dst_station_coord, unit=Unit.KILOMETERS)
                transit_time = (transit_dist / self.METRO_SPEED) * 3600
                total_time_seconds += transit_time

            # --- Leg 3: Walk from dst_station to dst ---
            walk_2_result = self.gmaps.directions(dst_station_coord, dst, mode="walking")
            if not walk_2_result:
                raise Exception(f"Google Maps could not find walking route for Leg 3: {dst_station_coord} to {dst}")
            total_time_seconds += walk_2_result[0]['legs'][0]['duration']['value']

            return total_time_seconds, route_points
        
        except Exception as e:
            print(f"Error calculating travel time: {e}")
            return None
    

    def calculate_travel_time_matrix(self, start_points: pd.DataFrame, end_points: pd.DataFrame, 
                                    mode: str, radius_km: float = 1.0,
                                    use_batch_api: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates matrices of travel times and routes using optimized methods.

        Optimizations:
        1. Pre-calculates nearest stations for all points (O(1) lookup vs O(N) search)
        2. Uses batched Distance Matrix API calls (reduces API calls from N*M*3 to ~3)
        3. Can use parallelization for remaining calculations

        Args:
            start_points (pd.DataFrame): A DataFrame with 'latitude' and 'longitude' for starting locations.
            end_points (pd.DataFrame): A DataFrame with 'latitude' and 'longitude' for ending locations.
            mode (str): The mode of transportation ('citibike' or 'subway').
            radius_km (float): The search radius for nearby stations.
            use_batch_api (bool): Whether to use Distance Matrix API for batched calls.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - A 2D NumPy array of travel times in seconds.
                - A 2D NumPy array (dtype=object) of route coordinate lists.
        """
        num_starts = len(start_points)
        num_ends = len(end_points)
        
        travel_time_matrix = np.zeros((num_starts, num_ends))
        route_matrix = np.empty((num_starts, num_ends), dtype=object)
        
        print(f"Pre-calculating nearest {mode} stations...")
        # Pre-calculate nearest stations
        src_stations = self.pre_calculate_nearest_stations(start_points, mode, radius_km)
        dst_stations = self.pre_calculate_nearest_stations(end_points, mode, radius_km)
        
        # Check for any None values (points without nearby stations)
        valid_src_indices = [i for i, s in enumerate(src_stations) if s is not None]
        valid_dst_indices = [j for j, s in enumerate(dst_stations) if s is not None]
        
        if not valid_src_indices or not valid_dst_indices:
            print("Warning: Some points have no nearby stations. Results will contain NaN values.")
            return travel_time_matrix, route_matrix
        
        # Extract valid coordinates
        start_coords = [(start_points.iloc[i]['latitude'], start_points.iloc[i]['longitude']) 
                       for i in valid_src_indices]
        end_coords = [(end_points.iloc[j]['latitude'], end_points.iloc[j]['longitude']) 
                     for j in valid_dst_indices]
        valid_src_stations = [src_stations[i] for i in valid_src_indices]
        valid_dst_stations = [dst_stations[j] for j in valid_dst_indices]
        
        if use_batch_api:
            print("Using batched Distance Matrix API calls...")
            
            # Leg 1: Walk from start points to their nearest stations
            print("  - Calculating walk times to origin stations...")
            walk_to_src_matrix = self.batch_distance_matrix_call(
                origins=start_coords,
                destinations=valid_src_stations,
                mode='walking'
            )
            
            # Leg 2: Transit between stations
            if mode == 'citibike':
                print("  - Calculating bike times between stations...")
                # Get unique station pairs to avoid duplicate API calls
                unique_station_pairs = list(set(itertools.product(valid_src_stations, valid_dst_stations)))
                unique_src = [pair[0] for pair in unique_station_pairs]
                unique_dst = [pair[1] for pair in unique_station_pairs]
                
                transit_time_lookup = self.batch_distance_matrix_call(
                    origins=unique_src,
                    destinations=unique_dst,
                    mode='bicycling'
                )
                
                # Build lookup dictionary
                transit_dict = {}
                for idx, (src, dst) in enumerate(unique_station_pairs):
                    transit_dict[(src, dst)] = transit_time_lookup[idx // len(unique_dst), idx % len(unique_dst)]
                
            elif mode == 'subway':
                print("  - Calculating subway times using Haversine...")
                # Use Haversine for subway (fast, local calculation)
                transit_dict = {}
                for src_station in valid_src_stations:
                    for dst_station in valid_dst_stations:
                        dist = hs.haversine(src_station, dst_station, unit=Unit.KILOMETERS)
                        time = (dist / self.METRO_SPEED) * 3600
                        transit_dict[(src_station, dst_station)] = time

            # Leg 3: Walk from destination stations to end points
            print("  - Calculating walk times from destination stations...")
            walk_from_dst_matrix = self.batch_distance_matrix_call(
                origins=valid_dst_stations,
                destinations=end_coords,
                mode='walking'
            )
            
            # Combine all legs
            print(" - Combining travel time legs...")
            for i_idx, i in enumerate(valid_src_indices):
                for j_idx, j in enumerate(valid_dst_indices):
                    src_station = valid_src_stations[i_idx]
                    dst_station = valid_dst_stations[j_idx]
                    
                    # Extract diagonal elements (start[i] -> its nearest station)
                    walk_to = walk_to_src_matrix[i_idx, i_idx]
                    walk_from = walk_from_dst_matrix[j_idx, j_idx]
                    transit = transit_dict.get((src_station, dst_station), np.nan)
                    
                    if not np.isnan(walk_to) and not np.isnan(walk_from) and not np.isnan(transit):
                        travel_time_matrix[i, j] = walk_to + transit + walk_from
                        
                        # Build route points
                        src = (start_points.iloc[i]['latitude'], start_points.iloc[i]['longitude'])
                        dst = (end_points.iloc[j]['latitude'], end_points.iloc[j]['longitude'])
                        route_matrix[i, j] = [src, src_station, dst_station, dst]
                    else:
                        travel_time_matrix[i, j] = np.nan
                        route_matrix[i, j] = []
        
        else:
            # Option to use Directions API instead
            print("Using original method with pre-calculated stations...")
            for i in tqdm(range(num_starts), desc=f"Calculating {mode} travel times"):
                for j in range(num_ends):
                    if src_stations[i] is None or dst_stations[j] is None:
                        travel_time_matrix[i, j] = np.nan
                        route_matrix[i, j] = []
                        continue
                    
                    src = (start_points.iloc[i]['latitude'], start_points.iloc[i]['longitude'])
                    dst = (end_points.iloc[j]['latitude'], end_points.iloc[j]['longitude'])
                    src_station_coord = src_stations[i]
                    dst_station_coord = dst_stations[j]
                    
                    try:
                        total_time = 0
                        
                        # Leg 1: Walk to station
                        walk_1 = self.gmaps.directions(src, src_station_coord, mode="walking")
                        if walk_1:
                            total_time += walk_1[0]['legs'][0]['duration']['value']
                        else:
                            raise Exception("No walking route found")
                        
                        # Leg 2: Transit
                        if mode == 'citibike':
                            bike = self.gmaps.directions(src_station_coord, dst_station_coord, mode="bicycling")
                            if bike:
                                total_time += bike[0]['legs'][0]['duration']['value']
                            else:
                                raise Exception("No biking route found")
                        else:
                            dist = hs.haversine(src_station_coord, dst_station_coord, unit=Unit.KILOMETERS)
                            total_time += (dist / self.METRO_SPEED) * 3600
                        
                        # Leg 3: Walk from station
                        walk_2 = self.gmaps.directions(dst_station_coord, dst, mode="walking")
                        if walk_2:
                            total_time += walk_2[0]['legs'][0]['duration']['value']
                        else:
                            raise Exception("No walking route found")
                        
                        travel_time_matrix[i, j] = total_time
                        route_matrix[i, j] = [src, src_station_coord, dst_station_coord, dst]
                        
                    except Exception as e:
                        travel_time_matrix[i, j] = np.nan
                        route_matrix[i, j] = []

        return travel_time_matrix, route_matrix
    

    def map_simulation_results(self, bike_route_matrix, subway_route_matrix, 
                               src_hub_points, dst_hub_points):
        """
        Creates a Folium map with simulation results, including routes, and saves it to a file.

        Args:
            bike_route_matrix (np.ndarray): Matrix of bike route coordinates.
            subway_route_matrix (np.ndarray): Matrix of subway route coordinates.
            src_hub_points (pd.DataFrame): DataFrame of starting points.
            dst_hub_points (pd.DataFrame): DataFrame of destination points.
        """
        m = folium.Map(location=[40.7347, -73.9903], zoom_start=12)
        
        # --- Create Layers for Map Objects ---
        bike_station_layer = folium.FeatureGroup(name="Bike Stations")
        subway_station_layer = folium.FeatureGroup(name="Subway Stations")
        src_layer = folium.FeatureGroup(name="Starting Points")
        dst_layer = folium.FeatureGroup(name="Destination Points")
        bike_route_layer = folium.FeatureGroup(name="Bike Routes", show=False)
        subway_route_layer = folium.FeatureGroup(name="Subway Routes", show=False)

        # --- Add Station and Point Markers ---
        for _, row in self.citibike_stations.iterrows():
            folium.Marker(location=(row['latitude'], row['longitude']), 
                         tooltip=row['station_name'], 
                         icon=folium.Icon(color='blue', icon='bicycle', prefix='fa')).add_to(bike_station_layer)
        for _, row in self.subway_stations.iterrows():
            folium.Marker(location=(row['latitude'], row['longitude']), 
                         tooltip=row['station_name'], 
                         icon=folium.Icon(color='red', icon='train', prefix='fa')).add_to(subway_station_layer)
        for _, row in src_hub_points.iterrows():
            folium.Marker(location=(row['latitude'], row['longitude']), 
                         tooltip=f'Start {row["point_id"]}', 
                         icon=folium.Icon(color='green', icon='flag-checkered', prefix='fa')).add_to(src_layer)
        for _, row in dst_hub_points.iterrows():
            folium.Marker(location=(row['latitude'], row['longitude']), 
                         tooltip=f'End {row["point_id"]}', 
                         icon=folium.Icon(color='orange', icon='flag-checkered', prefix='fa')).add_to(dst_layer)

        # --- Add Routes to the Map ---
        for routes in bike_route_matrix.flatten():
            if routes and len(routes) > 0:
                folium.PolyLine(locations=routes, color='blue', weight=3, opacity=0.7).add_to(bike_route_layer)
        for routes in subway_route_matrix.flatten():
            if routes and len(routes) > 0:
                folium.PolyLine(locations=routes, color='green', weight=3, opacity=0.7).add_to(subway_route_layer)

        # --- Add Layers to Map ---
        m.add_child(bike_station_layer)
        m.add_child(subway_station_layer)
        m.add_child(src_layer)
        m.add_child(dst_layer)
        m.add_child(bike_route_layer)
        m.add_child(subway_route_layer)
        m.add_child(folium.LayerControl())

        return m


def main():
    """
    Main function to demonstrate the Simulator's capabilities.
    """
    print("="*80)
    print("Initializing Optimized Public Transport Simulator")
    print("="*80)
    
    # The location to center the analysis around
    center_location = (40.6535720712609, -73.931131331664)
    
    try:
        # Initialize StationFinder first, as it's needed by the Simulator
        finder = StationFinder(
            citibike_graph_path='../citibike_weekday_network.gml',
            subway_graph_path='../subway_graph_weekday_weekend.gml'
        )
        simulator = Simulator(finder)
    except (SystemExit, FileNotFoundError) as e:
        print(f"Failed to initialize simulator: {e}")
        return

    # --- 1. Generate random start and end points for simulation ---
    num_points = 10  # Increased for better testing
    start_points = simulator.generate_random_points(center_location, radius_km=1.6, num_points=num_points)
    end_points = simulator.generate_random_points(center_location, radius_km=1.6, num_points=num_points)
    
    # Add point IDs for easier identification on the map
    start_points['point_id'] = range(num_points)
    end_points['point_id'] = range(num_points)

    print(f"\nSimulating travel for {num_points} start/end pairs...")
    print("-"*80)

    # --- 2. Calculate travel time and route matrices with optimizations ---
    bike_time_matrix, bike_route_matrix = simulator.calculate_travel_time_matrix(
        start_points, end_points, mode='citibike', use_batch_api=True
    )
    subway_time_matrix, subway_route_matrix = simulator.calculate_travel_time_matrix(
        start_points, end_points, mode='subway', use_batch_api=True
    )

    # --- 3. Map the simulation results ---
    map_obj = simulator.map_simulation_results(
        bike_route_matrix=bike_route_matrix,
        subway_route_matrix=subway_route_matrix,
        src_hub_points=start_points,
        dst_hub_points=end_points
    )
    
    # Save the map
    output_path = "../map_visualizations/simulation_routes_optimized.html"
    map_obj.save(output_path)
    print(f"\nMap saved to: {output_path}")

    # --- 4. Print average travel times ---
    avg_bike_time = np.nanmean(bike_time_matrix)
    avg_subway_time = np.nanmean(subway_time_matrix)
    print(f"\nAverage Bike Travel Time: {avg_bike_time / 60:.2f} minutes")
    print(f"Average Subway Travel Time: {avg_subway_time / 60:.2f} minutes")
    
    # Print time saved comparison
    valid_comparisons = ~(np.isnan(bike_time_matrix) | np.isnan(subway_time_matrix))
    if valid_comparisons.any():
        bike_faster = np.sum((bike_time_matrix < subway_time_matrix) & valid_comparisons)
        subway_faster = np.sum((subway_time_matrix < bike_time_matrix) & valid_comparisons)
        print(f"\nBike faster in {bike_faster} routes, Subway faster in {subway_faster} routes")


if __name__ == "__main__":
    main()