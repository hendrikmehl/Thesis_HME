import pandas as pd
import numpy as np
import heapq
from typing import List, Dict, Tuple


def get_travel_time(matrix: pd.DataFrame, from_zone: int, to_zone: int) -> float:
    """Get travel time between zones from matrix, return inf if not available"""
    try:
        return matrix.loc[from_zone, to_zone]
    except (KeyError, IndexError):
        return np.inf


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
            travel_time = get_travel_time(zone_time_matrix, driver_end_zone, trip_start_zone)
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



def link_sequential_trips(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame) -> Dict:
    """
    Main function to link sequential trips and calculate driver requirements.
    
    Args:
        trip_data: DataFrame with trip information
        zone_time_matrix: Matrix with travel times between zones
        
    Returns:
        Dictionary with results including number of drivers and trip chains
    """
    # Calculate minimum drivers needed
    num_drivers = calculate_minimum_drivers(trip_data, zone_time_matrix)
    
    # Build trip chains for analysis
    trips = trip_data.sort_values('pickup_datetime').reset_index(drop=True)
    trip_chains = []
    driver_assignments = {}
    
    # Track drivers with their current state
    drivers = []
    
    for trip_idx, trip in trips.iterrows():
        trip_start = trip['pickup_datetime']
        trip_end = trip['dropoff_datetime']
        trip_start_zone = trip['PULocationID']
        trip_end_zone = trip['DOLocationID']
        
        # Find available driver
        assigned_driver = None
        
        for driver_id, (driver_end_time, driver_end_zone) in enumerate(drivers):
            time_gap = (trip_start - driver_end_time).total_seconds()
            required_travel_time = get_travel_time(zone_time_matrix, driver_end_zone, trip_start_zone)
            
            if time_gap >= required_travel_time:
                assigned_driver = driver_id
                break
        
        if assigned_driver is not None:
            # Assign to existing driver
            drivers[assigned_driver] = (trip_end, trip_end_zone)
            trip_chains[assigned_driver].append(trip_idx)
        else:
            # Create new driver
            driver_id = len(drivers)
            drivers.append((trip_end, trip_end_zone))
            trip_chains.append([trip_idx])
            assigned_driver = driver_id
        
        driver_assignments[trip_idx] = assigned_driver
    
    return {
        'num_drivers_needed': num_drivers,
        'trip_chains': trip_chains,
        'driver_assignments': driver_assignments,
        'total_trips': len(trip_data)
    }