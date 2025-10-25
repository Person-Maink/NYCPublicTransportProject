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

from station_utilities import StationFinder


class Simulator:
    """
    A class to simulate travel times and analyze public transport networks in NYC.

    This simulator can find nearby stations, generate random points for analysis,
    and calculate travel times between two points using different modes of transport
    like walking, biking, and the subway system.
    """
    
    def __init__(self, finder: StationFinder):
        """
        Initializes the Simulator, loading station data and setting constants.
        """
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
        # print("\n" + "="*80)
        # print(f"Finding stations within a {radius_km}km radius")
        # print("="*80)
        lat, lng = location[0], location[1]
        # print(f"Center point: ({lat}, {lng})")
        stations = self.finder.find_stations_within_radius(lat, lng, radius_km=radius_km)
        # print(f"Found {len(stations['citibike'])} Citibike stations and {len(stations['subway'])} Subway stations within {radius_km} km radius.")

        return stations
    
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
        print("" + "="*80)
        print(f"Generating {num_points} random points within a {radius_km} km radius")
        print("="*80)
        random_points = self.finder.generate_random_points_in_circle(location[0], location[1], radius_km=radius_km, num_points=num_points)

        return random_points
    
    
    def insert_station(self, station: Tuple[str, Tuple[float, float]]) -> None:
        """
        Inserts a station into the stations dataframe.

        Args:
            station (tuple): A tuple containing the station name and its (latitude, longitude).
            stations (dict): The dictionary of stations to insert into.
        """
        station_name, station_coords = station
        if station_name not in self.subway_stations:
            print(f"\nInserting new station: {station_name} at {station_coords}")
            self.subway_stations.loc[len(self.subway_stations)] = [station_name, station_coords[0], station_coords[1]]
            
    
    
    def calculate_travel_time(self, src: Tuple[float, float], dst: Tuple[float, float], mode: str = 'citibike', radius_km: float = 1.0) -> Optional[Tuple[float, List[Tuple[float, float]]]]:
        """
        Calculate the total travel time between source and destination with the selected mode of transportation.

        The travel time includes:
        1. Walking from the source to the nearest station.
        2. Traveling between the nearest source and destination stations via the chosen mode.
        3. Walking from the destination station to the final destination.

        Args:
            src (tuple): The (latitude, longitude) of the starting point.
            dst (tuple): The (latitude, longitude) of the destination point.
            mode (str): The mode of transportation ('citibike' or 'subway').

        Returns:
            Optional[tuple]: A tuple containing:
                             - The total travel time (in seconds).
                             - A list of coordinates representing the route points.
                             Returns None if the mode is invalid or stations cannot be found.
        """
        if mode == 'citibike':
            transit_speed = self.BIKE_SPEED
        elif mode == 'subway':
            transit_speed = self.METRO_SPEED
        else:
            warnings.warn("Invalid mode of transportation specified. Choose 'citibike' or 'subway'.")
            return None
        
        src_station = self.get_nearby_stations(src, radius_km=radius_km)[mode].iloc[0] # nearest station to source
        dst_station = self.get_nearby_stations(dst, radius_km=radius_km)[mode].iloc[0] # nearest station to destination

        src_station_name = src_station['station_name']
        src_station_coord = (src_station['latitude'], src_station['longitude'])
        dst_station_name = dst_station['station_name']
        dst_station_coord = (dst_station['latitude'], dst_station['longitude'])

        if src_station_name is None or dst_station_name is None:
            print(f"Could not find start/end station for mode '{mode}'. Aborting.")
            return None

        route_points = [
            src,
            src_station_coord,
            dst_station_coord,
            dst
        ]

        # Calculate distances in kilometers
        walk_dist_1 = hs.haversine(src, src_station_coord, unit=Unit.KILOMETERS)
        walk_dist_2 = hs.haversine(dst_station_coord, dst, unit=Unit.KILOMETERS)
        transit_dist = hs.haversine(src_station_coord, dst_station_coord, unit=Unit.KILOMETERS)

        # Calculate time in hours and convert to seconds
        walking_time = (walk_dist_1 + walk_dist_2) / self.WALK_SPEED * 3600
        transit_time = transit_dist / transit_speed * 3600

        return walking_time + transit_time, route_points
    

    def calculate_travel_time_matrix(self, start_points, end_points, mode, radius_km=1.0):
        """
        Calculates matrices of travel times and routes by iterating through each start/end pair.

        Args:
            start_points (pd.DataFrame): A DataFrame with 'latitude' and 'longitude' for starting locations.
            end_points (pd.DataFrame): A DataFrame with 'latitude' and 'longitude' for ending locations.
            mode (str): The mode of transportation ('citibike' or 'subway').
            radius_km (float): The search radius for nearby stations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - A 2D NumPy array of travel times in seconds.
                - A 2D NumPy array (dtype=object) of route coordinate lists.
        """
        num_starts = len(start_points)
        num_ends = len(end_points)
        
        travel_time_matrix = np.zeros((num_starts, num_ends))
        route_matrix = np.empty((num_starts, num_ends), dtype=object)
        
        for i, row_i in tqdm(start_points.iterrows(), total=start_points.shape[0], desc=f"Calculating {mode} travel times"):
            for j, row_j in end_points.iterrows():
                src = (row_i['latitude'], row_i['longitude'])
                dst = (row_j['latitude'], row_j['longitude'])
                result = self.calculate_travel_time(src=src, dst=dst, mode=mode, radius_km=radius_km)
                
                if result:
                    time, route = result
                    travel_time_matrix[i, j] = time
                    route_matrix[i, j] = route
                else:
                    travel_time_matrix[i, j] = np.nan
                    route_matrix[i, j] = []

        return travel_time_matrix, route_matrix
    

    def map_simulation_results(self, bike_route_matrix, subway_route_matrix, src_hub_points, dst_hub_points):
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
            folium.Marker(location=(row['latitude'], row['longitude']), tooltip=row['station_name'], icon=folium.Icon(color='blue', icon='bicycle', prefix='fa')).add_to(bike_station_layer)
        for _, row in self.subway_stations.iterrows():
            folium.Marker(location=(row['latitude'], row['longitude']), tooltip=row['station_name'], icon=folium.Icon(color='red', icon='train', prefix='fa')).add_to(subway_station_layer)
        for _, row in src_hub_points.iterrows():
            folium.Marker(location=(row['latitude'], row['longitude']), tooltip=f'Start {row["point_id"]}', icon=folium.Icon(color='green', icon='flag-checkered', prefix='fa')).add_to(src_layer)
        for _, row in dst_hub_points.iterrows():
            folium.Marker(location=(row['latitude'], row['longitude']), tooltip=f'End {row["point_id"]}', icon=folium.Icon(color='orange', icon='flag-checkered', prefix='fa')).add_to(dst_layer)

        # --- Add Routes to the Map ---
        for routes in bike_route_matrix.flatten():
            if routes:
                folium.PolyLine(locations=routes, color='blue', weight=3, opacity=0.7).add_to(bike_route_layer)
        for routes in subway_route_matrix.flatten():
            if routes:
                folium.PolyLine(locations=routes, color='green', weight=3, opacity=0.7).add_to(subway_route_layer)

        # --- Add Layers to Map and Save ---
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
    print("Initializing Public Transport Simulator")
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
    num_points = 3
    start_points = simulator.generate_random_points(center_location, radius_km=2, num_points=num_points)
    end_points = simulator.generate_random_points(center_location, radius_km=5, num_points=num_points)
    
    # Add point IDs for easier identification on the map
    start_points['point_id'] = range(num_points)
    end_points['point_id'] = range(num_points)

    print(f"\nSimulating travel for {num_points} start/end pairs...")
    print("-"*80)

    # --- 2. Calculate travel time and route matrices ---
    bike_time_matrix, bike_route_matrix = simulator.calculate_travel_time_matrix(start_points, end_points, mode='citibike')
    subway_time_matrix, subway_route_matrix = simulator.calculate_travel_time_matrix(start_points, end_points, mode='subway')

    # --- 3. Map the simulation results ---
    simulator.map_simulation_results(
        bike_route_matrix=bike_route_matrix,
        subway_route_matrix=subway_route_matrix,
        src_hub_points=start_points,
        dst_hub_points=end_points,
        output_filename="../map_visualizations/simulation_routes.html"
    )

    # --- 4. Optional: Print average travel times ---
    avg_bike_time = np.nanmean(bike_time_matrix)
    avg_subway_time = np.nanmean(subway_time_matrix)
    print(f"\nAverage Bike Travel Time: {avg_bike_time / 60:.2f} minutes")
    print(f"Average Subway Travel Time: {avg_subway_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()
