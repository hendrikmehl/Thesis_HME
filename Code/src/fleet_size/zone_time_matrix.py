import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd


def calculate_zone_time_matrix(df):    
    # Filter valid trips with reasonable durations
    valid_trips = df[
        (df['PULocationID'].notna()) & 
        (df['DOLocationID'].notna()) &
        (df['trip_time'] > 0) &
        (df['trip_time'] < 7200)  # Less than 2 hours
    ].copy()
    
    print(f"Valid trips: {len(valid_trips)}")
    
    # Calculate average travel times between zones
    avg_df = valid_trips.groupby(['PULocationID', 'DOLocationID'], as_index=False)['trip_time'].agg({
        'duration': 'mean',
        'trip_count': 'count'
    })
    avg_df.columns = ['from_zone', 'to_zone', 'duration', 'trip_count']
    
    print(f"Direct zone connections found: {len(avg_df)} out of {259*259} possible pairs")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with travel times
    for _, row in avg_df.iterrows():
        u, v, d = int(row['from_zone']), int(row['to_zone']), row['duration']
        # Only add if we don't have this edge or if this is a better (shorter) duration
        if not G.has_edge(u, v) or d < G[u][v]['weight']:
            G.add_edge(u, v, weight=d)
        
    # Get all zones that appear in the data
    all_zones = sorted(set(valid_trips['PULocationID'].unique()) | set(valid_trips['DOLocationID'].unique()))
    print(f"Total zones: {len(all_zones)}")
    
    # Check connectivity
    pickup_zones = set(valid_trips['PULocationID'].unique())
    dropoff_zones = set(valid_trips['DOLocationID'].unique())
    isolated_zones = set(all_zones) - pickup_zones
    
    if isolated_zones:
        print(f"Warning: {len(isolated_zones)} zones have no outgoing trips: {sorted(list(isolated_zones))[:10]}...")
    
    # Calculate shortest paths between all pairs
    print("Calculating shortest paths...")
    try:
        # Use Dijkstra for all-pairs shortest paths
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    except Exception as e:
        print(f"Error calculating shortest paths: {e}")
        lengths = {}
    
    # Create matrix
    matrix = pd.DataFrame(index=all_zones, columns=all_zones, dtype=float)
    
    # Fill diagonal with zeros (same zone)
    for zone in all_zones:
        matrix.loc[zone, zone] = 0.0
    
    # Fill with shortest path lengths
    for u in all_zones:
        if u in lengths:
            for v in lengths[u]:
                if v in all_zones:
                    matrix.loc[u, v] = lengths[u][v]    
    
    # Print some statistics
    non_inf_values = matrix.replace([np.inf, -np.inf], np.nan).dropna().values.flatten()
    non_zero_values = non_inf_values[non_inf_values > 0]

    fill_diagonal_values(df, matrix)
    
    if len(non_zero_values) > 0:
        print(f"Travel time statistics:")
        print(f"  Mean: {np.mean(non_zero_values):.1f} seconds")
        print(f"  Median: {np.median(non_zero_values):.1f} seconds")
        print(f"  Max: {np.max(non_zero_values):.1f} seconds")
        print(f"  Connections with data: {len(non_zero_values)} / {len(all_zones)**2}")
    
    # Export to CSV
    matrix.to_csv('zone_time_matrix.csv')

    return matrix

def fill_diagonal_values(df, matrix):
    """
    Fill diagonal values of the matrix with intra-zone travel times.
    For zones with trips starting and ending within the same zone, use the average trip time.
    For zones without such trips, use the overall average of all intra-zone trips.
    """
    # Filter valid trips within the same zone
    intra_zone_trips = df[
        (df['PULocationID'] == df['DOLocationID']) & 
        (df['PULocationID'].notna()) & 
        (df['trip_time'] > 0) &
        (df['trip_time'] < 7200)  # Less than 2 hours
    ].copy()
    
    print(f"Total intra-zone trips found: {len(intra_zone_trips)}")
    
    # Calculate average trip time for each zone
    zone_avg_times = intra_zone_trips.groupby('PULocationID')['trip_time'].mean()
    zones_with_intra_trips = set(zone_avg_times.index)
    
    print(f"Zones with intra-zone trips: {len(zones_with_intra_trips)}")
    
    # Get all zones in the matrix
    all_zones = set(matrix.index)
    zones_without_intra_trips = all_zones - zones_with_intra_trips
    
    print(f"Zones without intra-zone trips: {len(zones_without_intra_trips)}")
    print(f"Example zones without intra-zone trips: {sorted(list(zones_without_intra_trips))[:10]}")
    
    # Calculate overall average for zones without data
    if len(zone_avg_times) > 0:
        overall_avg = zone_avg_times.mean()
        print(f"Overall average intra-zone trip time: {overall_avg:.1f} seconds ({overall_avg/60:.1f} minutes)")
    else:
        overall_avg = 300  # Default 5 minutes if no intra-zone trips
        print("No intra-zone trips found, using default 5 minutes")
    
    # Fill diagonal values
    for zone in matrix.index:
        if zone in zones_with_intra_trips:
            matrix.loc[zone, zone] = zone_avg_times[zone]
        else:
            matrix.loc[zone, zone] = overall_avg
    
    return matrix

def create_interactive_travel_heatmap(matrix, taxi_zones=None):    
    # Convert matrix to minutes if it's in seconds
    matrix_minutes = matrix / 60
    
    # Create interactive heatmap with Plotly
    fig = go.Figure(data=go.Heatmap(
        z=matrix_minutes.values,
        x=matrix_minutes.columns,
        y=matrix_minutes.index,
        colorscale='Viridis',
        colorbar=dict(title="Travel Time (minutes)"),
        hoverongaps=False,
        hovertemplate='Origin Zone: %{y}<br>Destination Zone: %{x}<br>Travel Time: %{z:.1f} min<extra></extra>'
    ))
    
    fig.update_layout(
        title="NYC Taxi Zone Travel Times",
        xaxis_title="Destination Zone",
        yaxis_title="Origin Zone",
        width=800,
        height=800
    )
    
    fig.show()
    return fig

def create_zone_explorer_widget(matrix, taxi_zones=None):
    """Create an interactive widget to explore travel times between zones"""
    
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    zone_options = [(f"Zone {zone}", zone) for zone in sorted(matrix.index)]
    
    origin_dropdown = widgets.Dropdown(
        options=zone_options,
        description='Origin:',
        style={'description_width': 'initial'}
    )
    
    destination_dropdown = widgets.Dropdown(
        options=zone_options,
        description='Destination:',
        style={'description_width': 'initial'}
    )
    
    output = widgets.Output()
    
    def update_travel_time(change=None):
        with output:
            clear_output(wait=True)
            origin = origin_dropdown.value
            dest = destination_dropdown.value
            
            if origin in matrix.index and dest in matrix.columns:
                travel_time = matrix.loc[origin, dest]
                if pd.notna(travel_time):
                    minutes = travel_time / 60 if travel_time > 120 else travel_time
                    print(f"Travel time from Zone {origin} to Zone {dest}: {minutes:.1f} minutes")
                    
                    # Create a simple bar chart showing this route
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(['Travel Time'], [minutes], color='skyblue')
                    ax.set_ylabel('Minutes')
                    ax.set_title(f'Zone {origin} → Zone {dest}')
                    plt.show()
                else:
                    print(f"No travel time data available for Zone {origin} → Zone {dest}")
    
    origin_dropdown.observe(update_travel_time, names='value')
    destination_dropdown.observe(update_travel_time, names='value')
    
    display(widgets.VBox([
        widgets.HBox([origin_dropdown, destination_dropdown]),
        output
    ]))
    
    update_travel_time()  # Initial display

def create_interactive_zone_map(matrix, taxi_zones=None):
    if taxi_zones is None:
        print("Taxi zones data required for interactive map")
        return None
    
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import folium
    from folium import plugins
    
    # Get zone IDs that exist in both matrix and shapefile
    matrix_zones = set(matrix.index.astype(int))
    shapefile_zones = set(taxi_zones['LocationID'].unique())
    available_zones = sorted(matrix_zones & shapefile_zones)
    
    if len(available_zones) == 0:
        print("No matching zones found between matrix and shapefile")
        return None
    
    # Create initial map
    bounds = taxi_zones.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Widgets for zone selection
    origin_widget = widgets.Dropdown(
        options=[(f"Zone {z}", z) for z in available_zones],
        value=available_zones[0],
        description='Origin:'
    )
    
    dest_widget = widgets.Dropdown(
        options=[(f"Zone {z}", z) for z in available_zones],
        value=available_zones[1] if len(available_zones) > 1 else available_zones[0],
        description='Destination:'
    )
    
    output = widgets.Output()
    
    def update_map(change=None):
        with output:
            clear_output(wait=True)
            
            origin_zone = origin_widget.value
            dest_zone = dest_widget.value
            
            # Get travel time
            if origin_zone in matrix.index and dest_zone in matrix.columns:
                travel_time = matrix.loc[origin_zone, dest_zone]
                travel_time_min = travel_time / 60 if travel_time > 120 else travel_time
            else:
                travel_time_min = None
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # Add all zones as base layer
            folium.GeoJson(
                taxi_zones.to_json(),
                style_function=lambda feature: {
                    'fillColor': 'lightgray',
                    'color': 'gray',
                    'weight': 1,
                    'fillOpacity': 0.2,
                }
            ).add_to(m)
            
            # Highlight origin zone
            origin_zone_data = taxi_zones[taxi_zones['LocationID'] == origin_zone]
            if not origin_zone_data.empty:
                folium.GeoJson(
                    origin_zone_data.to_json(),
                    style_function=lambda feature: {
                        'fillColor': 'green',
                        'color': 'darkgreen',
                        'weight': 3,
                        'fillOpacity': 0.7,
                    },
                    popup=f"Origin: Zone {origin_zone}",
                    tooltip=f"Origin: Zone {origin_zone}"
                ).add_to(m)
                
                # Add origin marker
                origin_centroid = origin_zone_data.geometry.centroid.iloc[0]
                folium.Marker(
                    [origin_centroid.y, origin_centroid.x],
                    popup=f"Origin: Zone {origin_zone}",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
            
            # Highlight destination zone
            dest_zone_data = taxi_zones[taxi_zones['LocationID'] == dest_zone]
            if not dest_zone_data.empty:
                folium.GeoJson(
                    dest_zone_data.to_json(),
                    style_function=lambda feature: {
                        'fillColor': 'red',
                        'color': 'darkred',
                        'weight': 3,
                        'fillOpacity': 0.7,
                    },
                    popup=f"Destination: Zone {dest_zone}",
                    tooltip=f"Destination: Zone {dest_zone}"
                ).add_to(m)
                
                # Add destination marker
                dest_centroid = dest_zone_data.geometry.centroid.iloc[0]
                folium.Marker(
                    [dest_centroid.y, dest_centroid.x],
                    popup=f"Destination: Zone {dest_zone}",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
                
                # Draw line between zones if both exist
                if not origin_zone_data.empty:
                    origin_coords = [origin_centroid.y, origin_centroid.x]
                    dest_coords = [dest_centroid.y, dest_centroid.x]
                    
                    # Add travel time info box
                    if travel_time_min is not None:
                        folium.PolyLine(
                            [origin_coords, dest_coords],
                            weight=4,
                            color='blue',
                            opacity=0.8,
                            popup=f"Travel Time: {travel_time_min:.1f} minutes"
                        ).add_to(m)
                        
                        # Add info box in top right
                        travel_info = f"""
                        <div style="position: fixed; 
                                    top: 10px; right: 10px; width: 200px; height: 80px; 
                                    background-color: white; border: 2px solid black;
                                    border-radius: 5px; padding: 10px; z-index:9999; 
                                    font-size:14px;">
                        <b>Travel Information</b><br>
                        Origin: Zone {origin_zone}<br>
                        Destination: Zone {dest_zone}<br>
                        <b>Travel Time: {travel_time_min:.1f} min</b>
                        </div>
                        """
                        m.get_root().html.add_child(folium.Element(travel_info))
            
            # Display the map
            display(m)
            
            # Print travel time info
            if travel_time_min is not None:
                print(f"Travel time from Zone {origin_zone} to Zone {dest_zone}: {travel_time_min:.1f} minutes")
            else:
                print(f"No travel time data available for this route")
    
    # Set up event handlers
    origin_widget.observe(update_map, names='value')
    dest_widget.observe(update_map, names='value')
    
    # Display widgets and initial map
    display(widgets.VBox([
        widgets.HBox([origin_widget, dest_widget]),
        output
    ]))
    
    # Show initial map
    update_map()
    
    return None

def plausibility_check_zone_time_matrix(matrix, taxi_zones=None):
    """
    Comprehensive plausibility check for the zone time matrix.
    
    Args:
        matrix: Zone-to-zone travel time matrix (in seconds)
        taxi_zones: Optional GeoDataFrame with zone geometries for distance checks
    
    Returns:
        dict: Dictionary with check results and statistics
    """
    import warnings
    
    print("🔍 Running Zone Time Matrix Plausibility Checks")
    print("=" * 50)
    
    results = {}
    issues = []
    
    # 1. Basic Structure Checks
    print("\n1. Basic Structure Checks:")
    print(f"   Matrix shape: {matrix.shape}")
    print(f"   Expected square matrix: {matrix.shape[0] == matrix.shape[1]}")
    
    if matrix.shape[0] != matrix.shape[1]:
        issues.append("Matrix is not square")
    
    # 2. Diagonal Check (same zone travel times)
    print("\n2. Diagonal Values Check:")
    diagonal_values = np.diag(matrix.values)
    diagonal_minutes = diagonal_values / 60
    
    print(f"   Diagonal range: {diagonal_minutes.min():.1f} - {diagonal_minutes.max():.1f} minutes")
    print(f"   Diagonal mean: {diagonal_minutes.mean():.1f} minutes")
    print(f"   Diagonal median: {np.median(diagonal_minutes):.1f} minutes")
    
    # Check for unreasonable diagonal values
    if diagonal_minutes.min() < 0:
        issues.append("Negative diagonal values found")
    if diagonal_minutes.max() > 30:
        issues.append(f"Very high diagonal values found (max: {diagonal_minutes.max():.1f} min)")
    if diagonal_minutes.min() == 0 and diagonal_minutes.max() > 0:
        print("   ⚠️  Some zones have 0 travel time, others don't")
    
    results['diagonal_stats'] = {
        'min_minutes': diagonal_minutes.min(),
        'max_minutes': diagonal_minutes.max(),
        'mean_minutes': diagonal_minutes.mean(),
        'median_minutes': np.median(diagonal_minutes)
    }
    
    # 3. Symmetry Check
    print("\n3. Symmetry Check:")
    matrix_T = matrix.T
    symmetric_diff = np.abs(matrix.values - matrix_T.values)
    non_zero_mask = (matrix.values != 0) & (matrix_T.values != 0)
    
    if non_zero_mask.any():
        relative_diff = symmetric_diff[non_zero_mask] / np.maximum(matrix.values[non_zero_mask], matrix_T.values[non_zero_mask])
        asymmetric_pairs = np.sum(relative_diff > 0.5)  # More than 50% difference
        
        print(f"   Highly asymmetric pairs (>50% difference): {asymmetric_pairs}")
        print(f"   Max relative difference: {relative_diff.max():.2f}")
        print(f"   Mean relative difference: {relative_diff.mean():.2f}")
        
        if asymmetric_pairs > len(matrix) * 2:  # Arbitrary threshold
            issues.append(f"Many asymmetric travel times ({asymmetric_pairs} pairs)")
    
    # 4. Missing Data Check
    print("\n4. Missing Data Check:")
    total_pairs = len(matrix) ** 2
    inf_count = np.isinf(matrix.values).sum()
    nan_count = np.isnan(matrix.values).sum()
    zero_count = (matrix.values == 0).sum() - len(matrix)  # Exclude diagonal
    
    print(f"   Total possible pairs: {total_pairs}")
    print(f"   Infinite values: {inf_count} ({inf_count/total_pairs*100:.1f}%)")
    print(f"   NaN values: {nan_count} ({nan_count/total_pairs*100:.1f}%)")
    print(f"   Zero values (non-diagonal): {zero_count} ({zero_count/(total_pairs-len(matrix))*100:.1f}%)")
    
    missing_data_pct = (inf_count + nan_count) / total_pairs * 100
    if missing_data_pct > 50:
        issues.append(f"High percentage of missing data ({missing_data_pct:.1f}%)")
    
    results['missing_data'] = {
        'inf_count': inf_count,
        'nan_count': nan_count,
        'zero_count': zero_count,
        'missing_percentage': missing_data_pct
    }
    
    # 5. Travel Time Distribution Check
    print("\n5. Travel Time Distribution Check:")
    valid_times = matrix.replace([np.inf, -np.inf], np.nan).values.flatten()
    valid_times = valid_times[~np.isnan(valid_times) & (valid_times > 0)]
    
    if len(valid_times) > 0:
        valid_minutes = valid_times / 60
        
        print(f"   Valid travel times: {len(valid_times)} / {total_pairs}")
        print(f"   Range: {valid_minutes.min():.1f} - {valid_minutes.max():.1f} minutes")
        print(f"   Mean: {valid_minutes.mean():.1f} minutes")
        print(f"   Median: {np.median(valid_minutes):.1f} minutes")
        print(f"   95th percentile: {np.percentile(valid_minutes, 95):.1f} minutes")
        
        # Check for unreasonable values
        if valid_minutes.max() > 180:  # More than 3 hours
            issues.append(f"Very long travel times found (max: {valid_minutes.max():.1f} min)")
        if valid_minutes.min() < 0.5:  # Less than 30 seconds
            issues.append(f"Very short travel times found (min: {valid_minutes.min():.1f} min)")
        
        results['time_distribution'] = {
            'min_minutes': valid_minutes.min(),
            'max_minutes': valid_minutes.max(),
            'mean_minutes': valid_minutes.mean(),
            'median_minutes': np.median(valid_minutes),
            'p95_minutes': np.percentile(valid_minutes, 95)
        }
    
    # 6. Geographic Distance Check (if taxi_zones provided)
    if taxi_zones is not None:
        print("\n6. Geographic Distance Check:")
        try:
            # Calculate actual distances between zone centroids
            zone_distances = calculate_zone_distances(taxi_zones, matrix.index)
            
            # Compare travel times with distances
            speed_analysis = analyze_speed_plausibility(matrix, zone_distances)
            results['speed_analysis'] = speed_analysis
            
            if speed_analysis['unrealistic_speeds'] > len(matrix) * 0.1:
                issues.append(f"Many unrealistic speeds detected ({speed_analysis['unrealistic_speeds']} pairs)")
                
        except Exception as e:
            print(f"   Could not perform geographic check: {e}")
    
    # 7. Triangle Inequality Check
    print("\n7. Triangle Inequality Check:")
    triangle_violations = check_triangle_inequality(matrix)
    print(f"   Triangle inequality violations: {triangle_violations}")
    
    if triangle_violations > len(matrix) * 5:  # Arbitrary threshold
        issues.append(f"Many triangle inequality violations ({triangle_violations})")
    
    results['triangle_violations'] = triangle_violations
    
    # 8. Summary
    print("\n" + "=" * 50)
    print("📊 PLAUSIBILITY CHECK SUMMARY")
    print("=" * 50)
    
    if not issues:
        print("✅ All checks passed! Matrix appears plausible.")
    else:
        print("⚠️  Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    results['issues'] = issues
    results['is_plausible'] = len(issues) == 0
    
    return results

def calculate_zone_distances(taxi_zones, zone_ids):
    """Calculate geographic distances between zone centroids"""
    from geopy.distance import geodesic
    
    # Filter zones to only those in the matrix
    available_zones = taxi_zones[taxi_zones['LocationID'].isin(zone_ids)].copy()
    
    # Calculate centroids
    available_zones['centroid'] = available_zones.geometry.centroid
    available_zones['lat'] = available_zones.centroid.y
    available_zones['lon'] = available_zones.centroid.x
    
    distances = {}
    
    for _, zone1 in available_zones.iterrows():
        for _, zone2 in available_zones.iterrows():
            coord1 = (zone1['lat'], zone1['lon'])
            coord2 = (zone2['lat'], zone2['lon'])
            dist_km = geodesic(coord1, coord2).kilometers
            distances[(zone1['LocationID'], zone2['LocationID'])] = dist_km
    
    return distances

def analyze_speed_plausibility(matrix, zone_distances):
    """Analyze if travel times are plausible given geographic distances"""
    
    speeds = []
    unrealistic_count = 0
    
    for (zone1, zone2), distance_km in zone_distances.items():
        if zone1 in matrix.index and zone2 in matrix.columns:
            travel_time_hours = matrix.loc[zone1, zone2] / 3600
            
            if travel_time_hours > 0 and not np.isnan(travel_time_hours) and not np.isinf(travel_time_hours):
                speed_kmh = distance_km / travel_time_hours
                speeds.append(speed_kmh)
                
                # Check for unrealistic speeds (too fast or too slow)
                if speed_kmh > 80 or speed_kmh < 5:  # Outside reasonable urban driving range
                    unrealistic_count += 1
    
    if speeds:
        print(f"   Average speed: {np.mean(speeds):.1f} km/h")
        print(f"   Speed range: {np.min(speeds):.1f} - {np.max(speeds):.1f} km/h")
        print(f"   Unrealistic speeds: {unrealistic_count} / {len(speeds)} pairs")
    
    return {
        'mean_speed_kmh': np.mean(speeds) if speeds else None,
        'min_speed_kmh': np.min(speeds) if speeds else None,
        'max_speed_kmh': np.max(speeds) if speeds else None,
        'unrealistic_speeds': unrealistic_count
    }

def check_triangle_inequality(matrix):
    """Check for triangle inequality violations in the matrix"""
    violations = 0
    zones = matrix.index
    
    # Sample a subset for performance (checking all combinations would be O(n³))
    import random
    sample_size = min(50, len(zones))
    sampled_zones = random.sample(list(zones), sample_size)
    
    for i, zone_a in enumerate(sampled_zones):
        for j, zone_b in enumerate(sampled_zones[i+1:], i+1):
            for k, zone_c in enumerate(sampled_zones[j+1:], j+1):
                
                # Get travel times
                ab = matrix.loc[zone_a, zone_b]
                bc = matrix.loc[zone_b, zone_c]
                ac = matrix.loc[zone_a, zone_c]
                
                # Check if all values are valid
                if all(pd.notna(x) and not np.isinf(x) and x > 0 for x in [ab, bc, ac]):
                    # Triangle inequality: ab + bc >= ac
                    if ab + bc < ac * 0.95:  # Allow 5% tolerance
                        violations += 1
    
    return violations
