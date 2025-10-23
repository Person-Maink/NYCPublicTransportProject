import networkx as nx
import pandas as pd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_hubs(G, hubs_df, title, save_path):
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot all stations
    xs = [G.nodes[n]['longitude'] for n in G.nodes()]
    ys = [G.nodes[n]['latitude'] for n in G.nodes()]
    plt.scatter(xs, ys, c='lightgray', s=15, label='All Stations', alpha=0.3)
    
    # Calculate marker sizes based on total_strength
    min_size = 100
    max_size = 2000
    sizes = min_size + (hubs_df['total_strength'] - hubs_df['total_strength'].min()) * \
            (max_size - min_size) / (hubs_df['total_strength'].max() - hubs_df['total_strength'].min())
    
    # Plot top hubs with a more visible color scheme
    scatter = plt.scatter(
        hubs_df['longitude'], hubs_df['latitude'],
        c=hubs_df['total_strength'],
        s=sizes,
        cmap='YlOrRd',
        alpha=0.6,
        label='Hub Stations'
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Trip Count', fontsize=10)
    
    # Add labels for top 5 hubs only (to reduce clutter)
    top_5 = hubs_df.head(5)
    for _, row in top_5.iterrows():
        # Create shortened station name
        station_name = row['station'].split('/')[-1].strip()
        station_name = station_name.replace(' & ', '\n').replace(' - ', '\n')
        
        # Add text with white outline for better visibility
        text = plt.text(row['longitude'], row['latitude'], station_name,
                       fontsize=8, ha='center', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                       zorder=3)
    
    # Improve the appearance
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Longitude", fontsize=10)
    plt.ylabel("Latitude", fontsize=10)
    
    # Add gridlines
    plt.grid(True, alpha=0.3)
    
    # Adjust legend
    plt.legend(loc='upper right', framealpha=0.9)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def analyze_hubs(G, top_n=10):
    """
    Compute various centrality measures to identify Citibike hubs.
    Returns a DataFrame with combined scores.
    """

    print("Begin analysis:")
    # Weighted degrees (trip count)
    out_strength = dict(G.out_degree(weight='trip_count'))
    in_strength = dict(G.in_degree(weight='trip_count'))
    total_strength = {n: out_strength.get(n, 0) + in_strength.get(n, 0) for n in G.nodes()}

    # Centralities (optional — can be slow on large graphs)
    betweenness = nx.betweenness_centrality(G, weight='trip_count', normalized=True)
    closeness = nx.closeness_centrality(G, distance=lambda u, v, d: 1 / (d['trip_count'] + 1e-6))

    # Combine into a DataFrame
    df = pd.DataFrame({
        'station': list(G.nodes()),
        'in_strength': [in_strength[n] for n in G.nodes()],
        'out_strength': [out_strength[n] for n in G.nodes()],
        'total_strength': [total_strength[n] for n in G.nodes()],
        'betweenness': [betweenness[n] for n in G.nodes()],
        'closeness': [closeness[n] for n in G.nodes()],
        'latitude': [G.nodes[n]['latitude'] for n in G.nodes()],
        'longitude': [G.nodes[n]['longitude'] for n in G.nodes()]
    })

    print("Dataframe created")
    # Sort by total activity (or other metric)
    df = df.sort_values('total_strength', ascending=False).head(top_n)

    return df



# Create output directories if they don't exist
Path('../plots').mkdir(exist_ok=True)
Path('outputs').mkdir(exist_ok=True)

# Load networks
G_weekday = nx.read_gml('../citibike_weekday_network.gml')
G_weekend = nx.read_gml('../citibike_weekend_network.gml')

# Analyze weekday data
weekday_hubs = analyze_hubs(G_weekday)
weekday_hubs.to_csv('outputs/weekday_hubs.csv', index=False)
print(f"Saved weekday hubs data to outputs/weekday_hubs.csv")

plot_hubs(G_weekday, weekday_hubs, 
          "Top Weekday Citibike Hubs",
          '../plots/weekday_hubs.png')
print(f"Saved weekday hubs plot to plots/weekday_hubs.png")

# Analyze weekend data
weekend_hubs = analyze_hubs(G_weekend)
plot_hubs(G_weekend, weekend_hubs, 
          "Top Weekend Citibike Hubs",
          '../plots/weekend_hubs.png')

# Save weekend hub data
weekend_hubs.to_csv('outputs/weekend_hubs.csv', index=False)
print(f"Saved weekend hubs data to outputs/weekend_hubs.csv")
print(f"Saved weekend hubs plot to plots/weekend_hubs.png")
