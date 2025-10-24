#!/usr/bin/env python3
"""
Visualize Citibike NetworkX graph overlaid on NYC map
Supports multiple visualization methods: static (geopandas), interactive (folium), 3D (pydeck)
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path


def load_graph(graph_path):
    """Load a NetworkX graph from GML file."""
    print(f"Loading graph from {graph_path}...")
    G = nx.read_gml(graph_path)
    print(f"  Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def prepare_geodataframes(G, top_n_edges=None):
    """
    Convert NetworkX graph to GeoDataFrames for visualization.
    
    Args:
        G: NetworkX graph with nodes having 'latitude' and 'longitude' attributes
        top_n_edges: If specified, only keep the top N edges by weight/trip_count
    
    Returns:
        nodes_gdf: GeoDataFrame of nodes
        edges_gdf: GeoDataFrame of edges
    """
    print("Creating GeoDataFrames from graph...")
    
    # Create nodes GeoDataFrame
    node_data = []
    for node_id, attrs in G.nodes(data=True):
        # Handle different attribute names (latitude/lat, longitude/lon/lng)
        lat = attrs.get('latitude') or attrs.get('lat')
        lon = attrs.get('longitude') or attrs.get('lon') or attrs.get('lng')
        
        if lat is None or lon is None:
            print(f"Warning: Node {node_id} missing coordinates, skipping")
            continue
            
        node_data.append({
            'node_id': node_id,
            'name': attrs.get('name', str(node_id)),
            'geometry': Point(lon, lat),
            'degree': G.degree(node_id),
            'in_degree': G.in_degree(node_id) if G.is_directed() else G.degree(node_id),
            'out_degree': G.out_degree(node_id) if G.is_directed() else G.degree(node_id),
        })
    
    nodes_gdf = gpd.GeoDataFrame(node_data, crs='EPSG:4326')
    print(f"  Created {len(nodes_gdf)} nodes")
    
    # Create edges GeoDataFrame
    edge_data = []
    for u, v, attrs in G.edges(data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        # Get coordinates
        u_lat = u_data.get('latitude') or u_data.get('lat')
        u_lon = u_data.get('longitude') or u_data.get('lon') or u_data.get('lng')
        v_lat = v_data.get('latitude') or v_data.get('lat')
        v_lon = v_data.get('longitude') or v_data.get('lon') or v_data.get('lng')
        
        if None in [u_lat, u_lon, v_lat, v_lon]:
            continue
        
        # Get edge weight (try different attribute names)
        weight = attrs.get('trip_count') or attrs.get('weight') or attrs.get('rides') or 1
        
        edge_data.append({
            'from': u,
            'to': v,
            'weight': float(weight),
            'geometry': LineString([(u_lon, u_lat), (v_lon, v_lat)])
        })
    
    edges_gdf = gpd.GeoDataFrame(edge_data, crs='EPSG:4326')
    print(f"  Created {len(edges_gdf)} edges")
    
    # Filter to top edges if requested
    if top_n_edges and len(edges_gdf) > top_n_edges:
        print(f"  Filtering to top {top_n_edges} edges by weight...")
        edges_gdf = edges_gdf.nlargest(top_n_edges, 'weight')
        print(f"  Kept {len(edges_gdf)} edges")
    
    return nodes_gdf, edges_gdf


def create_static_map(nodes_gdf, edges_gdf, output_path, title="Citibike Network", 
                      show_labels=True, figsize=(20, 16)):
    """
    Create a static map visualization using geopandas and contextily.
    
    Args:
        nodes_gdf: GeoDataFrame of nodes
        edges_gdf: GeoDataFrame of edges
        output_path: Path to save the output image
        title: Title for the map
        show_labels: Whether to show labels for high-traffic stations
        figsize: Figure size (width, height)
    """
    print(f"\nCreating static map visualization...")
    
    # Convert to Web Mercator projection for contextily
    nodes_web = nodes_gdf.to_crs(epsg=3857)
    edges_web = edges_gdf.to_crs(epsg=3857)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot edges with width based on weight
    if len(edges_web) > 0:
        print("  Plotting edges...")
        max_weight = edges_web['weight'].max()
        min_weight = edges_web['weight'].min()
        
        # Normalize weights for line width (0.1 to 2.0)
        if max_weight > min_weight:
            edges_web['line_width'] = 0.1 + (edges_web['weight'] - min_weight) / (max_weight - min_weight) * 1.9
        else:
            edges_web['line_width'] = 1.0
        
        edges_web.plot(
            ax=ax,
            linewidth=edges_web['line_width'],
            alpha=0.4,
            color='#ff7f0e',
            zorder=1
        )
    
    # Plot nodes with size based on degree
    if len(nodes_web) > 0:
        print("  Plotting nodes...")
        # Scale node size by degree (20 to 200)
        max_degree = nodes_web['degree'].max()
        min_degree = nodes_web['degree'].min()
        
        if max_degree > min_degree:
            nodes_web['node_size'] = 20 + (nodes_web['degree'] - min_degree) / (max_degree - min_degree) * 180
        else:
            nodes_web['node_size'] = 50
        
        nodes_web.plot(
            ax=ax,
            markersize=nodes_web['node_size'],
            color='#1f77b4',
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5,
            zorder=2
        )
        
        # Add labels for high-traffic stations
        if show_labels:
            print("  Adding station labels...")
            high_degree_threshold = nodes_web['degree'].quantile(0.80)
            high_degree_nodes = nodes_web[nodes_web['degree'] >= high_degree_threshold]
            
            for idx, row in high_degree_nodes.iterrows():
                label = str(row['name']).split('&')[0].strip()
                if len(label) > 25:
                    label = label[:22] + '...'
                
                ax.annotate(
                    label,
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=6,
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.8,
                        linewidth=0.5
                    ),
                    zorder=3
                )
    
    # Add basemap
    print("  Adding basemap...")
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom='auto'
        )
    except Exception as e:
        print(f"  Warning: Could not add basemap: {e}")
        ax.set_facecolor('#f0f0f0')
    
    # Styling
    ax.set_title(
        f'{title}\n{len(nodes_gdf)} Stations, {len(edges_gdf)} Connections',
        fontsize=20,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')
    
    # Save
    plt.tight_layout()
    print(f"  Saving to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved successfully!")


def create_interactive_map(nodes_gdf, edges_gdf, output_path, title="Citibike Network"):
    """
    Create an interactive map using Folium.
    
    Args:
        nodes_gdf: GeoDataFrame of nodes
        edges_gdf: GeoDataFrame of edges
        output_path: Path to save the HTML file
        title: Title for the map
    """
    try:
        import folium
        from folium import plugins
    except ImportError:
        print("Folium not installed. Install with: pip install folium")
        return
    
    print(f"\nCreating interactive Folium map...")
    
    # Calculate center
    center_lat = nodes_gdf.geometry.y.mean()
    center_lon = nodes_gdf.geometry.x.mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # Add edges
    print("  Adding edges...")
    max_weight = edges_gdf['weight'].max()
    for idx, row in edges_gdf.iterrows():
        # Normalize weight for line thickness
        weight_normalized = (row['weight'] / max_weight) * 5 + 1
        
        coords = [(point[1], point[0]) for point in row.geometry.coords]
        
        folium.PolyLine(
            coords,
            weight=weight_normalized,
            color='#ff7f0e',
            opacity=0.4,
            popup=f"{row['from']} → {row['to']}<br>Trips: {int(row['weight'])}"
        ).add_to(m)
    
    # Add nodes
    print("  Adding nodes...")
    for idx, row in nodes_gdf.iterrows():
        # Normalize size
        size = (row['degree'] / nodes_gdf['degree'].max()) * 10 + 3
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=size,
            popup=f"<b>{row['name']}</b><br>"
                  f"Total connections: {row['degree']}<br>"
                  f"Incoming: {row['in_degree']}<br>"
                  f"Outgoing: {row['out_degree']}",
            color='white',
            fillColor='#1f77b4',
            fillOpacity=0.7,
            weight=1
        ).add_to(m)
    
    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: 60px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; padding: 10px">
    <b>{title}</b><br>
    {len(nodes_gdf)} stations, {len(edges_gdf)} connections
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    print(f"  Saving to {output_path}...")
    m.save(output_path)
    print(f"  Saved successfully!")


def create_pydeck_map(nodes_gdf, edges_gdf, output_path, title="Citibike Network"):
    """
    Create a 3D visualization using PyDeck.
    
    Args:
        nodes_gdf: GeoDataFrame of nodes
        edges_gdf: GeoDataFrame of edges
        output_path: Path to save the HTML file
        title: Title for the map
    """
    try:
        import pydeck as pdk
    except ImportError:
        print("PyDeck not installed. Install with: pip install pydeck")
        return
    
    print(f"\nCreating PyDeck 3D visualization...")
    
    # Prepare edge data for arc layer
    edge_data = []
    max_weight = edges_gdf['weight'].max()
    
    for idx, row in edges_gdf.iterrows():
        coords = list(row.geometry.coords)
        edge_data.append({
            'from_lon': coords[0][0],
            'from_lat': coords[0][1],
            'to_lon': coords[1][0],
            'to_lat': coords[1][1],
            'weight': row['weight'],
            'color': [255, 127, 14, int(100 + (row['weight'] / max_weight) * 155)]
        })
    
    # Prepare node data
    node_data = []
    max_degree = nodes_gdf['degree'].max()
    
    for idx, row in nodes_gdf.iterrows():
        node_data.append({
            'lon': row.geometry.x,
            'lat': row.geometry.y,
            'name': row['name'],
            'degree': row['degree'],
            'elevation': row['degree'] * 5,
            'radius': (row['degree'] / max_degree) * 100 + 20
        })
    
    # Create layers
    arc_layer = pdk.Layer(
        'ArcLayer',
        data=edge_data,
        get_source_position=['from_lon', 'from_lat'],
        get_target_position=['to_lon', 'to_lat'],
        get_source_color=[255, 127, 14, 100],
        get_target_color=[255, 127, 14, 100],
        auto_highlight=True,
        width_scale=0.5,
        get_width='weight',
        width_min_pixels=1,
        width_max_pixels=10,
    )
    
    scatterplot_layer = pdk.Layer(
        'ScatterplotLayer',
        data=node_data,
        get_position=['lon', 'lat'],
        get_radius='radius',
        get_fill_color=[31, 119, 180, 200],
        pickable=True,
        auto_highlight=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )
    
    # Set view
    center_lat = nodes_gdf.geometry.y.mean()
    center_lon = nodes_gdf.geometry.x.mean()
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11,
        pitch=45,
        bearing=0
    )
    
    # Create deck
    r = pdk.Deck(
        layers=[arc_layer, scatterplot_layer],
        initial_view_state=view_state,
        tooltip={
            'html': '<b>{name}</b><br/>Connections: {degree}',
            'style': {'color': 'white'}
        },
        map_style='light'
    )
    
    # Save
    print(f"  Saving to {output_path}...")
    r.to_html(output_path)
    print(f"  Saved successfully!")


def main():
    """Main function with example usage."""
    import sys
    
    # Example: Visualize the weekday network
    graph_path = 'citibike_weekend_network.gml'
    
    if not Path(graph_path).exists():
        print(f"Graph file not found: {graph_path}")
        print("\nAvailable graph files:")
        for f in Path('.').rglob('*.gml'):
            print(f"  {f}")
        sys.exit(1)
    
    # Load graph
    G = load_graph(graph_path)
    
    # Prepare GeoDataFrames
    # For large networks, limit edges for better visualization
    nodes_gdf, edges_gdf = prepare_geodataframes(G, top_n_edges=300000)
    
    # Create output directory
    output_dir = Path('map_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Static high-quality map
    create_static_map(
        nodes_gdf, 
        edges_gdf,
        output_path=output_dir / 'citibike_static_map.png',
        title='NYC Citibike Network - Weekday',
        show_labels=True
    )
    
    # 2. Interactive Folium map
    create_interactive_map(
        nodes_gdf,
        edges_gdf,
        output_path=output_dir / 'citibike_interactive_map.html',
        title='NYC Citibike Network - Weekday (Interactive)'
    )
    
    # 3. PyDeck 3D visualization
    create_pydeck_map(
        nodes_gdf,
        edges_gdf,
        output_path=output_dir / 'citibike_3d_map.html',
        title='NYC Citibike Network - Weekday (3D)'
    )
    
    print(f"\n{'='*60}")
    print("All visualizations complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

