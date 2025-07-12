import pandas as pd
import numpy as np

def calculate_fleet_size_over_time(trip_data, zone_time_matrix, target_date='2025-02-01'):
    trips = trip_data[['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']].copy()
    trips = trips.sort_values('pickup_datetime').reset_index(drop=True)

    start_date = pd.to_datetime(target_date)
    end_date = start_date + pd.Timedelta(days=1)
    
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total trips before filtering: {len(trips)}")
    
    # Filter rides that overlap with the target date
    filtered_trips = trips[
        (trips['pickup_datetime'] < end_date) &
        (trips['dropoff_datetime'] > start_date)
    ].copy()
    
    print(f"Filtered trips: {len(filtered_trips)}")
    if len(filtered_trips) == 0:
        print("No trips found matching the criteria!")
        print(f"Trip date range: {trips['pickup_datetime'].min()} to {trips['pickup_datetime'].max()}")

    driver_count = 0
    # Active drivers
    active_driver_start_times = np.array([], dtype='datetime64[s]')
    active_driver_end_times = np.array([], dtype='datetime64[s]')
    active_driver_end_zones = np.array([], dtype=int)
    active_driver_work_durations = np.array([], dtype='float64')
    
    # Paused drivers (idle for >30 minutes)
    paused_driver_start_times = np.array([], dtype='datetime64[s]')
    paused_driver_end_times = np.array([], dtype='datetime64[s]')
    paused_driver_end_zones = np.array([], dtype=int)
    paused_driver_work_durations = np.array([], dtype='float64')

    # Finished drivers (worked 6.5 hours)
    finished_driver_start_times = np.array([], dtype='datetime64[s]')
    finished_driver_end_times = np.array([], dtype='datetime64[s]')
    finished_driver_work_durations = np.array([], dtype='float64')

    # Track active drivers per minute
    active_drivers_timeline = []

    for i, trip in filtered_trips.iterrows():
        trip_start = trip['pickup_datetime'].to_datetime64()
        trip_end = trip['dropoff_datetime'].to_datetime64()
        trip_duration = (trip_end - trip_start) / np.timedelta64(1, 's')
        
        if len(active_driver_end_times) == 0 and len(paused_driver_end_times) == 0:
            # First trip - need a new driver
            active_driver_start_times = np.append(active_driver_start_times, trip_start)
            active_driver_end_times = np.append(active_driver_end_times, trip_end)
            active_driver_end_zones = np.append(active_driver_end_zones, trip['DOLocationID'])
            active_driver_work_durations = np.append(active_driver_work_durations, trip_duration)
            driver_count += 1
        else:
            best_driver_idx = -1
            
            # Check active drivers first
            if len(active_driver_end_times) > 0:
                time_gaps = (trip_start - active_driver_end_times) / np.timedelta64(1, 's')
                
                # Get travel times for all drivers
                travel_times = zone_time_matrix.loc[active_driver_end_zones, trip['PULocationID']].values

                available_mask = (
                    (time_gaps >= travel_times) &  # Can reach pickup location
                    (time_gaps <= 3600) &  # Not idle for more than 1 hour
                    (active_driver_work_durations < (23400*2))  # Don't work more than 6.5 hours
                )
                
                if np.any(available_mask):
                    waiting_times = time_gaps - travel_times
                    available_waiting_times = np.where(available_mask, waiting_times, np.inf)
                    best_driver_idx = np.argmin(available_waiting_times)
                    # Update active driver
                    active_driver_end_times[best_driver_idx] = trip_end
                    active_driver_end_zones[best_driver_idx] = trip['DOLocationID']
                    active_driver_work_durations[best_driver_idx] += trip_duration

            
            # Only check paused drivers if no active driver is available
            if best_driver_idx == -1 and len(paused_driver_end_times) > 0:
                time_gaps = (trip_start - paused_driver_end_times) / np.timedelta64(1, 's')
                
                # Get travel times for all drivers
                travel_times = zone_time_matrix.loc[paused_driver_end_zones, trip['PULocationID']].values

                available_mask = (
                    (time_gaps >= travel_times) &  # Can reach pickup location
                    (paused_driver_work_durations + trip_duration < (23400*2))  # Won't exceed 6.5 hours
                )
                
                if np.any(available_mask):
                    waiting_times = time_gaps - travel_times
                    available_waiting_times = np.where(available_mask, waiting_times, np.inf)
                    best_driver_idx = np.argmin(available_waiting_times)
                    
                    # Reactivate paused driver
                    active_driver_start_times = np.append(active_driver_start_times, paused_driver_start_times[best_driver_idx])
                    active_driver_end_times = np.append(active_driver_end_times, trip_end)
                    active_driver_end_zones = np.append(active_driver_end_zones, trip['DOLocationID'])
                    active_driver_work_durations = np.append(active_driver_work_durations, paused_driver_work_durations[best_driver_idx] + trip_duration)
                    
                    # Remove from paused arrays
                    paused_driver_start_times = np.delete(paused_driver_start_times, best_driver_idx)
                    paused_driver_end_times = np.delete(paused_driver_end_times, best_driver_idx)
                    paused_driver_end_zones = np.delete(paused_driver_end_zones, best_driver_idx)
                    paused_driver_work_durations = np.delete(paused_driver_work_durations, best_driver_idx)

            if best_driver_idx == -1:
                # Need a new driver
                active_driver_start_times = np.append(active_driver_start_times, trip_start)
                active_driver_end_times = np.append(active_driver_end_times, trip_end)
                active_driver_end_zones = np.append(active_driver_end_zones, trip['DOLocationID'])
                active_driver_work_durations = np.append(active_driver_work_durations, trip_duration)
                driver_count += 1
            
            # Move active drivers to paused if they've been idle for >1 hour
            if len(active_driver_end_times) > 0:
                time_since_last_trip = (trip_start - active_driver_end_times) / np.timedelta64(1, 's')
                paused_mask = time_since_last_trip > 3600  # 1 hour

                if np.any(paused_mask):
                    # Move paused drivers to paused arrays
                    paused_indices = np.where(paused_mask)[0]
                    for idx in sorted(paused_indices, reverse=True):
                        paused_driver_start_times = np.append(paused_driver_start_times, active_driver_start_times[idx])
                        paused_driver_end_times = np.append(paused_driver_end_times, active_driver_end_times[idx])
                        paused_driver_end_zones = np.append(paused_driver_end_zones, active_driver_end_zones[idx])
                        paused_driver_work_durations = np.append(paused_driver_work_durations, active_driver_work_durations[idx])
                        
                        # Remove from active arrays
                        active_driver_start_times = np.delete(active_driver_start_times, idx)
                        active_driver_end_times = np.delete(active_driver_end_times, idx)
                        active_driver_end_zones = np.delete(active_driver_end_zones, idx)
                        active_driver_work_durations = np.delete(active_driver_work_durations, idx)

            # Move drivers to finished if they've been working for >=6.5 hours
            if len(active_driver_end_times) > 0:
                finished_mask = active_driver_work_durations >= 23400  # 6.5 hours in seconds
                
                if np.any(finished_mask):
                    # Move finished drivers to finished arrays
                    finished_indices = np.where(finished_mask)[0]
                    for idx in sorted(finished_indices, reverse=True):
                        finished_driver_start_times = np.append(finished_driver_start_times, active_driver_start_times[idx])
                        finished_driver_end_times = np.append(finished_driver_end_times, active_driver_end_times[idx])
                        finished_driver_work_durations = np.append(finished_driver_work_durations, active_driver_work_durations[idx])
                        
                        # Remove from active arrays
                        active_driver_start_times = np.delete(active_driver_start_times, idx)
                        active_driver_end_times = np.delete(active_driver_end_times, idx)
                        active_driver_end_zones = np.delete(active_driver_end_zones, idx)
                        active_driver_work_durations = np.delete(active_driver_work_durations, idx)
            
            # Also check paused drivers for 6.5 hour limit
            if len(paused_driver_end_times) > 0:
                finished_mask = paused_driver_work_durations >= 23400  # 6.5 hours in seconds
                
                if np.any(finished_mask):
                    # Move finished drivers to finished arrays
                    finished_indices = np.where(finished_mask)[0]
                    for idx in sorted(finished_indices, reverse=True):
                        finished_driver_start_times = np.append(finished_driver_start_times, paused_driver_start_times[idx])
                        finished_driver_end_times = np.append(finished_driver_end_times, paused_driver_end_times[idx])
                        finished_driver_work_durations = np.append(finished_driver_work_durations, paused_driver_work_durations[idx])
                        
                        # Remove from paused arrays
                        paused_driver_start_times = np.delete(paused_driver_start_times, idx)
                        paused_driver_end_times = np.delete(paused_driver_end_times, idx)
                        paused_driver_end_zones = np.delete(paused_driver_end_zones, idx)
                        paused_driver_work_durations = np.delete(paused_driver_work_durations, idx)

        # Record active drivers at this trip time
        active_drivers_timeline.append({
            'datetime': pd.to_datetime(trip_start),
            'active_drivers': len(active_driver_end_times),
            'paused_drivers': len(paused_driver_end_times),
            'finished_drivers': len(finished_driver_end_times),
            'total_drivers' : driver_count
        })
        
        # if i % 1000 == 0:
        #     print(f"Processed {i+1} trips, current driver count: {driver_count}, active: {len(active_driver_end_times)}, paused: {len(paused_driver_end_times)}, finished: {len(finished_driver_end_times)}")

    drivers_df = pd.DataFrame(active_drivers_timeline)
    
    # Add time-based features for analysis
    drivers_df['hour'] = drivers_df['datetime'].dt.hour
    drivers_df['minute'] = drivers_df['datetime'].dt.minute
    drivers_df['time_of_day'] = drivers_df['datetime'].dt.strftime('%H:%M')

    # Combine active, paused, and finished drivers for return (for compatibility)
    if len(active_driver_start_times) > 0 or len(paused_driver_start_times) > 0 or len(finished_driver_start_times) > 0:
        all_start_times = np.concatenate([active_driver_start_times, paused_driver_start_times, finished_driver_start_times])
        all_end_times = np.concatenate([active_driver_end_times, paused_driver_end_times, finished_driver_end_times])
    else:
        all_start_times = np.array([], dtype='datetime64[s]')
        all_end_times = np.array([], dtype='datetime64[s]')
    
    return all_start_times, all_end_times, driver_count, drivers_df


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


def visualize_active_drivers(active_drivers_df, active_trips_lyft=None):
    """
    Visualize the number of active drivers over time.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(active_drivers_df['datetime'], active_drivers_df['active_drivers'], 
             label='Active Drivers', color='blue', linewidth=1.5)
    plt.plot(active_drivers_df['datetime'], active_drivers_df['paused_drivers'], 
             label='Paused Drivers', color='orange', linewidth=1.5)
    plt.plot(active_drivers_df['datetime'], active_drivers_df['finished_drivers'], 
             label='Finished Drivers', color='green', linewidth=1.5)
    plt.plot(active_drivers_df['datetime'], active_drivers_df['total_drivers'], 
             label='Total Drivers', color='red', linewidth=1.5)
    
    if active_trips_lyft is not None:
        active_trips_lyft = pd.DataFrame(active_trips_lyft)
        plt.plot(active_trips_lyft['datetime'], active_trips_lyft['active_rides'], 
                 label='Active Rides (Lyft)', color='purple', linewidth=1.5)

    plt.title('Active NYC Taxi Drivers Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Drivers')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.show()
