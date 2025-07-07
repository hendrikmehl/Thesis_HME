import pandas as pd
import numpy as np
from typing import Dict

def calculate_minimum_drivers(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame) -> int:
    """
    For each trip, find all possible drivers who could serve it,
    then assign the driver with the minimal time difference (who arrives just in time).
    """
    trips = trip_data[['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']].copy()
    trips = trips.sort_values('pickup_datetime').reset_index(drop=True)
    pickup_times = trips['pickup_datetime'].values
    dropoff_times = trips['dropoff_datetime'].values
    pickup_zones = trips['PULocationID'].values
    dropoff_zones = trips['DOLocationID'].values

    drivers = []  # Each driver: (end_time, end_zone)
    for i in range(len(trips)):
        trip_start = pickup_times[i]
        trip_end = dropoff_times[i]
        trip_start_zone = pickup_zones[i]
        trip_end_zone = dropoff_zones[i]

        # Find all possible drivers who can serve this trip
        possible_drivers = []
        for j, (driver_end_time, driver_end_zone) in enumerate(drivers):
            # Calculate time passing between driver end time and trip start
            time_gap = (trip_start - driver_end_time) / np.timedelta64(1, 's')
            travel_time = zone_time_matrix.loc[driver_end_zone, trip_start_zone]
            if time_gap >= travel_time:
                passenger_waiting_time = time_gap - travel_time
                possible_drivers.append((j, passenger_waiting_time, travel_time))

        # Assign the driver who can be there the soonest
        if possible_drivers:
            # First select driver with the least waiting time, then the driver with the least travel time
            best_driver_idx = min(possible_drivers, key=lambda x: (x[1], x[2]))[0]
            drivers[best_driver_idx] = (trip_end, trip_end_zone)
        else:
            drivers.append((trip_end, trip_end_zone))

        if i % 100 == 0:
            print(f"Processed {i+1} trips, current driver count: {len(drivers)}")
    return len(drivers)


def calculate_minimum_drivers_vectorized(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame) -> int:
    """
    Alternative approach using more vectorized operations.
    """
    trips = trip_data[['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']].copy()
    trips = trips.sort_values('pickup_datetime').reset_index(drop=True)
    
    n_trips = len(trips)
    driver_count = 0
    driver_end_times = np.array([], dtype='datetime64[s]')
    driver_end_zones = np.array([], dtype=int)
    
    for i, trip in trips.iterrows():
        trip_start = trip['pickup_datetime'].to_datetime64()
        trip_end = trip['dropoff_datetime'].to_datetime64()
        
        if len(driver_end_times) == 0:
            # First trip - need a new driver
            driver_end_times = np.append(driver_end_times, trip_end)
            driver_end_zones = np.append(driver_end_zones, trip['DOLocationID'])
            driver_count += 1
        else:
            # Vectorized calculation of time gaps
            time_gaps = (trip_start - driver_end_times) / np.timedelta64(1, 's')
            
            # Get travel times for all drivers
            travel_times = zone_time_matrix.loc[driver_end_zones, trip['PULocationID']].values

            # Find available drivers
            available_mask = time_gaps >= travel_times
            
            if np.any(available_mask):
                # Find best driver (minimum waiting time)
                waiting_times = time_gaps - travel_times
                available_waiting_times = np.where(available_mask, waiting_times, np.inf)
                best_driver_idx = np.argmin(available_waiting_times)
                
                # Update the driver
                driver_end_times[best_driver_idx] = trip_end
                driver_end_zones[best_driver_idx] = trip['DOLocationID']
            else:
                # Need a new driver
                driver_end_times = np.append(driver_end_times, trip_end)
                driver_end_zones = np.append(driver_end_zones, trip['DOLocationID'])
                driver_count += 1
        
        if i % 1000 == 0:
            print(f"Processed {i+1} trips, current driver count: {driver_count}, ")
    
    return driver_count


def calculate_drivers_over_time(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame, 
                              time_interval_minutes: int = 15) -> pd.DataFrame:
    """
    Calculate the minimum number of drivers needed at each time interval throughout the day.
    
    Args:
        trip_data: DataFrame with trip information
        zone_time_matrix: Matrix with travel times between zones
        time_interval_minutes: Time interval in minutes for calculating driver requirements
        
    Returns:
        DataFrame with timestamps and corresponding minimum driver counts
    """
    trips = trip_data[['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']].copy()
    trips = trips.sort_values('pickup_datetime').reset_index(drop=True)
    
    # Create time range for the entire day 
    start_time = trips['pickup_datetime'].min().floor('H')  # Round down to nearest hour
    end_time = trips['dropoff_datetime'].max().ceil('H')    # Round up to nearest hour
    
    time_range = pd.date_range(start=start_time, end=end_time, 
                              freq=f'{time_interval_minutes}min')
    
    driver_counts = []
    
    for current_time in time_range:
        # Get trips that are active at this time point
        active_trips = trips[
            (trips['pickup_datetime'] <= current_time) & 
            (trips['dropoff_datetime'] > current_time)
        ].copy()
        
        if len(active_trips) == 0:
            driver_counts.append(0)
            continue
        
        # Calculate minimum drivers needed for these active trips
        # Sort by pickup time to process in chronological order
        active_trips = active_trips.sort_values('pickup_datetime').reset_index(drop=True)
        
        drivers = []  # Each driver: (end_time, end_zone)
        
        for _, trip in active_trips.iterrows():
            trip_start = trip['pickup_datetime']
            trip_end = trip['dropoff_datetime']
            trip_start_zone = trip['PULocationID']
            trip_end_zone = trip['DOLocationID']
            
            # Find all possible drivers who could serve this trip
            possible_drivers = []
            for j, (driver_end_time, driver_end_zone) in enumerate(drivers):
                time_gap = (trip_start - driver_end_time) / np.timedelta64(1, 's')
                travel_time = zone_time_matrix.loc[driver_end_zone, trip_start_zone]
                if time_gap >= travel_time:
                    passenger_waiting_time = time_gap - travel_time
                    possible_drivers.append((j, passenger_waiting_time, travel_time))
            
            # Assign the driver who can be there the soonest
            if possible_drivers:
                best_driver_idx = min(possible_drivers, key=lambda x: (x[1], x[2]))[0]
                drivers[best_driver_idx] = (trip_end, trip_end_zone)
            else:
                drivers.append((trip_end, trip_end_zone))
        
        driver_counts.append(len(drivers))
    
    return pd.DataFrame({
        'timestamp': time_range,
        'minimum_drivers': driver_counts
    })


def calculate_active_drivers_simple(trip_data: pd.DataFrame, time_interval_minutes: int = 15) -> pd.DataFrame:
    """
    Calculate the number of active trips (and thus minimum drivers) at each time interval.
    This is a simpler approach that doesn't consider driver reuse between trips.
    
    Args:
        trip_data: DataFrame with trip information
        time_interval_minutes: Time interval in minutes for calculating driver requirements
        
    Returns:
        DataFrame with timestamps and corresponding active trip counts
    """
    trips = trip_data[['pickup_datetime', 'dropoff_datetime']].copy()
    
    # Create time range for the entire day
    start_time = trips['pickup_datetime'].min().floor('H')
    end_time = trips['dropoff_datetime'].max().ceil('H')
    
    time_range = pd.date_range(start=start_time, end=end_time, 
                              freq=f'{time_interval_minutes}min')
    
    active_counts = []
    
    for current_time in time_range:
        # Count trips that are active at this time point
        active_trips_count = len(trips[
            (trips['pickup_datetime'] <= current_time) & 
            (trips['dropoff_datetime'] > current_time)
        ])
        active_counts.append(active_trips_count)
    
    return pd.DataFrame({
        'timestamp': time_range,
        'active_trips': active_counts,
        'minimum_drivers_simple': active_counts  # Each active trip needs one driver
    })


def calculate_minimum_drivers_timeline(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minimum drivers needed over time by tracking when drivers become available.
    This tracks the actual timeline of driver availability rather than cumulative maximum.
    
    Args:
        trip_data: DataFrame with trip information
        zone_time_matrix: Matrix with travel times between zones
        
    Returns:
        DataFrame with timeline of driver requirements
    """
    trips = trip_data[['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']].copy()
    trips = trips.sort_values('pickup_datetime').reset_index(drop=True)
    
    # Track events: trip starts and driver becomes available
    events = []
    drivers_timeline = []  # Will track (timestamp, driver_count)
    
    # Process trips to build driver timeline
    drivers = []  # Each driver: (end_time, end_zone)
    max_drivers = 0
    
    for i, trip in trips.iterrows():
        trip_start = trip['pickup_datetime']
        trip_end = trip['dropoff_datetime']
        trip_start_zone = trip['PULocationID']
        trip_end_zone = trip['DOLocationID']
        
        # Clean up drivers who could have finished other trips by now
        current_active_drivers = []
        for driver_end_time, driver_end_zone in drivers:
            if driver_end_time > trip_start:  # Driver still busy
                current_active_drivers.append((driver_end_time, driver_end_zone))
        
        # Find available drivers for this trip
        possible_drivers = []
        for j, (driver_end_time, driver_end_zone) in enumerate(current_active_drivers):
            time_gap = (trip_start - driver_end_time) / np.timedelta64(1, 's')
            travel_time = zone_time_matrix.loc[driver_end_zone, trip_start_zone]
            
            if time_gap >= travel_time:  # Driver can reach in time
                passenger_waiting_time = time_gap - travel_time
                possible_drivers.append((j, passenger_waiting_time, travel_time))
        
        if possible_drivers:
            # Assign existing driver
            best_driver_idx = min(possible_drivers, key=lambda x: (x[1], x[2]))[0]
            current_active_drivers[best_driver_idx] = (trip_end, trip_end_zone)
            drivers = current_active_drivers
        else:
            # Need new driver
            current_active_drivers.append((trip_end, trip_end_zone))
            drivers = current_active_drivers
        
        # Record driver count at this time
        current_driver_count = len([d for d in drivers if d[0] > trip_start])
        drivers_timeline.append((trip_start, current_driver_count))
        max_drivers = max(max_drivers, current_driver_count)
        
        if i % 1000 == 0:
            print(f"Processed {i+1} trips, max drivers so far: {max_drivers}")
    
    # Convert to DataFrame
    timeline_df = pd.DataFrame(drivers_timeline, columns=['timestamp', 'active_drivers'])
    timeline_df = timeline_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    return timeline_df


def analyze_driver_requirements_over_time(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame) -> Dict:
    """
    Comprehensive analysis of driver requirements over time using different methods.
    
    Args:
        trip_data: DataFrame with trip information
        zone_time_matrix: Matrix with travel times between zones
        
    Returns:
        Dictionary containing different analyses of driver requirements
    """
    print("Calculating driver requirements over time...")
    
    # Method 1: Simple active trips count (upper bound)
    print("Method 1: Calculating simple active trips...")
    simple_timeline = calculate_active_drivers_simple(trip_data, time_interval_minutes=15)
    
    # Method 2: Minimum drivers at time intervals considering trip sequencing
    print("Method 2: Calculating minimum drivers at intervals...")
    interval_timeline = calculate_drivers_over_time(trip_data, zone_time_matrix, time_interval_minutes=15)
    
    # Method 3: Detailed timeline of driver requirements
    print("Method 3: Calculating detailed driver timeline...")
    detailed_timeline = calculate_minimum_drivers_timeline(trip_data, zone_time_matrix)
    
    # Method 4: Overall minimum (original function for comparison)
    print("Method 4: Calculating overall minimum drivers...")
    overall_minimum = calculate_minimum_drivers(trip_data, zone_time_matrix)
    
    return {
        'simple_timeline': simple_timeline,
        'interval_timeline': interval_timeline,
        'detailed_timeline': detailed_timeline,
        'overall_minimum': overall_minimum,
        'peak_simple': simple_timeline['active_trips'].max(),
        'peak_interval': interval_timeline['minimum_drivers'].max(),
        'peak_detailed': detailed_timeline['active_drivers'].max() if not detailed_timeline.empty else 0
    }