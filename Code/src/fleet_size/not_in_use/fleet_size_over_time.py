import pandas as pd
import numpy as np

def calculate_fleet_size_over_time(trip_data, zone_time_matrix, target_date='2025-02-01'):
    trips = trip_data[['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']].copy()
    trips = trips.sort_values('pickup_datetime').reset_index(drop=True)

    start_date = pd.to_datetime(target_date) - pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=1) + pd.Timedelta(days=1)
        
    # Filter rides on which the dropoff_datetime is on the target date
    filtered_trips = trips[
        (trips['pickup_datetime'] < end_date) &
        (trips['dropoff_datetime'] > start_date)
    ].copy()

    driver_count = 0
    driver_start_times = np.array([], dtype='datetime64[s]')
    driver_end_times = np.array([], dtype='datetime64[s]')
    driver_end_zones = np.array([], dtype=int)

    for i, trip in filtered_trips.iterrows():
        trip_start = trip['pickup_datetime'].to_datetime64()
        trip_end = trip['dropoff_datetime'].to_datetime64()
        
        if len(driver_end_times) == 0:
            # First trip - need a new driver
            driver_start_times = np.append(driver_start_times, trip_start)
            driver_end_times = np.append(driver_end_times, trip_end)
            driver_end_zones = np.append(driver_end_zones, trip['DOLocationID'])
            driver_count += 1
        else:
            # Vectorized calculation of time gaps
            time_gaps = (trip_start - driver_end_times) / np.timedelta64(1, 's')
            
            # Get travel times for all drivers
            travel_times = zone_time_matrix.loc[driver_end_zones, trip['PULocationID']].values

            # Find available drivers, i.e., those who can start the trip and worked less than 10 hours
            work_durations = (driver_end_times - driver_start_times) / np.timedelta64(1, 's')
            available_mask = (time_gaps >= travel_times) & (work_durations < 10 * 3600)  # 10 hours in seconds

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
                driver_start_times = np.append(driver_start_times, trip_start)
                driver_end_times = np.append(driver_end_times, trip_end)
                driver_end_zones = np.append(driver_end_zones, trip['DOLocationID'])
                driver_count += 1
        
        if i % 1000 == 0:
            print(f"Processed {i+1} trips, current driver count: {driver_count}, ")

    return driver_start_times, driver_end_times, driver_count


def active_drivers_per_minute(driver_start_times, driver_end_times):
    """
    Calculate the number of active drivers for every minute on a specific date.
    A driver is active if start_time <= current_time < end_time
    """
    # Convert numpy datetime64 arrays to pandas datetime
    start_times = pd.to_datetime(driver_start_times)
    end_times = pd.to_datetime(driver_end_times)
    
    # Find the overall time range
    min_time = min(start_times.min(), end_times.min())
    max_time = max(start_times.max(), end_times.max())
    
    # Create time range for every minute
    time_range = pd.date_range(start=min_time.floor('min'), end=max_time.ceil('min'), freq='1min')[:-1]
    
    # Calculate active drivers for each minute
    active_drivers_per_minute = []
    
    for current_time in time_range:
        active_count = sum(
            (start_times <= current_time) & (end_times > current_time)
        )
        active_drivers_per_minute.append(active_count)
    
    # Create result DataFrame
    active_drivers_df = pd.DataFrame({
        'datetime': time_range,
        'active_drivers': active_drivers_per_minute
    })
    
    # Add time-based features for analysis
    active_drivers_df['hour'] = active_drivers_df['datetime'].dt.hour
    active_drivers_df['minute'] = active_drivers_df['datetime'].dt.minute
    active_drivers_df['time_of_day'] = active_drivers_df['datetime'].dt.strftime('%H:%M')
    
    return active_drivers_df


def visualize_active_drivers(active_drivers_df):
    """
    Visualize the number of active drivers over time.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(active_drivers_df['datetime'], active_drivers_df['active_drivers'], 
             label='Active Drivers', color='blue', linewidth=1.5)
    plt.title('Active NYC Taxi Drivers Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Active Drivers')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
