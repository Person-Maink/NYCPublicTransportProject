import pandas as pd
import networkx as nx
from pathlib import Path
import glob

def load_and_filter_data(csv_files):
    """
    Load and filter Citibike trip data from multiple CSV files.
    Only includes member rides and filters outliers.
    """
    dfs = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        # Filter for member rides only
        df = df[df['member_casual'] == 'member']
        
        # Filter latitude and longitude outliers (NYC approximate bounds)
        lat_bounds = (40.4774, 40.9176)  # NYC latitude bounds
        lon_bounds = (-74.2590, -73.7004)  # NYC longitude bounds
        
        df = df[
            (df['start_lat'].between(*lat_bounds)) &
            (df['start_lng'].between(*lon_bounds)) &
            (df['end_lat'].between(*lat_bounds)) &
            (df['end_lng'].between(*lon_bounds))
        ]
        
        # Filter trip duration outliers
        # Convert ended_at and started_at to datetime if they aren't already
        df['ended_at'] = pd.to_datetime(df['ended_at'])
        df['started_at'] = pd.to_datetime(df['started_at'])
        
        # Calculate trip duration in minutes
        df['duration_minutes'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
        
        # Filter out unreasonable durations (less than 1 minute or more than 12 hours)
        df = df[(df['duration_minutes'] >= 1) & (df['duration_minutes'] <= 720)]
        
        # Add day type (weekend/weekday)
        df['is_weekend'] = df['started_at'].dt.dayofweek.isin([5, 6])
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def create_network_graphs(df):
    """
    Create two NetworkX graphs (weekday and weekend) from the filtered data.
    Edges contain trip count and separate average durations for electric and classic bikes.
    Excludes self-edges (trips that start and end at the same station).
    """
    # Create separate dataframes for weekday and weekend
    weekday_df = df[~df['is_weekend']]
    weekend_df = df[df['is_weekend']]
    
    graphs = {}
    for day_type, data in [('weekday', weekday_df), ('weekend', weekend_df)]:
        # Remove self-edges (where start and end stations are the same)
        data = data[data['start_station_name'] != data['end_station_name']]
        
        # Separate calculations for electric and classic bikes
        grouped = data.groupby(
            ['start_station_name', 'start_lat', 'start_lng',
            'end_station_name', 'end_lat', 'end_lng', 'rideable_type']
        ).agg({
            'duration_minutes': 'mean',
            'ride_id': 'count'
        }).reset_index()
        
        # Pivot the data to get separate columns for each bike type
        pivoted = grouped.pivot_table(
            index=['start_station_name', 'start_lat', 'start_lng',
                    'end_station_name', 'end_lat', 'end_lng'],
            columns='rideable_type',
            values=['duration_minutes', 'ride_id'],
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        pivoted.columns = [f"{'' if col[0] == '' else col[0]}_{col[1]}" if col[1] != '' 
                            else col[0] for col in pivoted.columns]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes with positions
        start_stations = pivoted[['start_station_name', 'start_lat', 'start_lng']].drop_duplicates()
        end_stations = pivoted[['end_station_name', 'end_lat', 'end_lng']].drop_duplicates()
        
        for _, row in start_stations.iterrows():
            G.add_node(row['start_station_name'],
                      latitude=row['start_lat'],
                      longitude=row['start_lng'])
            
        for _, row in end_stations.iterrows():
            G.add_node(row['end_station_name'],
                      latitude=row['end_lat'],
                      longitude=row['end_lng'])
        
        # Add edges with properties
        for _, row in pivoted.iterrows():
            # Sum up all bike type trips for total count
            total_trips = sum(row[col] for col in pivoted.columns if col.startswith('ride_id_'))
            
            edge_properties = {
                'trip_count': int(total_trips),
                'electric_bike_duration': float(row.get('duration_minutes_electric_bike', 0)),
                'classic_bike_duration': float(row.get('duration_minutes_classic_bike', 0))
            }
            
            G.add_edge(
                row['start_station_name'],
                row['end_station_name'],
                **edge_properties
            )
        
        graphs[day_type] = G
    
    return graphs

def main():
    # Get all CSV files in the data directory
    data_dir = Path('202408-citibike-tripdata')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in the data directory!")
        return
    
    print(f"Loading and processing {len(csv_files)} CSV files...")
    df = load_and_filter_data(csv_files)
    print(f"Processed {len(df)} valid trips")
    
    print("Creating network graphs...")
    graphs = create_network_graphs(df)
    
    # Save graphs
    for day_type, G in graphs.items():
        output_path = f'citibike_{day_type}_network.gml'
        nx.write_gml(G, output_path)
        print(f"Saved {day_type} graph to {output_path}")
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

if __name__ == "__main__":
    main()