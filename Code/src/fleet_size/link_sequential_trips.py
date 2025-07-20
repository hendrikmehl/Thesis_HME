import pandas as pd
import numpy as np

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
        
        # if i % 1000 == 0:
            # print(f"Processed {i+1} trips, current driver count: {driver_count}, ")
    
    return driver_count

def calculate_minimum_drivers_improved(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame) -> int:
    trips = trip_data[['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']].copy()
    trips = trips.sort_values('pickup_datetime').reset_index(drop=True)
    
    driver_count = 0
    driver_end_times = np.array([], dtype='datetime64[s]')
    driver_end_zones = np.array([], dtype=int)
    driver_working_minutes = np.array([], dtype=int)
    
    for i, trip in trips.iterrows():
        trip_start = trip['pickup_datetime'].to_datetime64()
        trip_end = trip['dropoff_datetime'].to_datetime64()
        trip_duration = (trip_end - trip_start) / np.timedelta64(1, 'm')  # in minutes
        
        if len(driver_end_times) == 0:
            # First trip - need a new driver
            driver_end_times = np.append(driver_end_times, trip_end)
            driver_end_zones = np.append(driver_end_zones, trip['DOLocationID'])
            driver_working_minutes = np.append(driver_working_minutes, trip_duration)
            driver_count += 1
        else:
            # Vectorized calculation of time gaps
            time_gaps = (trip_start - driver_end_times) / np.timedelta64(1, 's')
            
            # Get travel times for all drivers
            travel_times = zone_time_matrix.loc[driver_end_zones, trip['PULocationID']].values

            # Find available drivers
            available_mask = ((time_gaps >= travel_times) & 
                              (driver_working_minutes < 6.5 * 60))
            
            if np.any(available_mask):
                # Find best driver (minimum waiting time)
                waiting_times = time_gaps - travel_times
                available_waiting_times = np.where(available_mask, waiting_times, np.inf)
                best_driver_idx = np.argmin(available_waiting_times)
                
                # Update the driver
                driver_end_times[best_driver_idx] = trip_end
                driver_end_zones[best_driver_idx] = trip['DOLocationID']
                driver_working_minutes[best_driver_idx] += trip_duration
            else:
                # Need a new driver
                driver_end_times = np.append(driver_end_times, trip_end)
                driver_end_zones = np.append(driver_end_zones, trip['DOLocationID'])
                driver_working_minutes = np.append(driver_working_minutes, trip_duration)
                driver_count += 1
        
        # if i % 1000 == 0:
        #     print(f"Processed {i+1} trips, current driver count: {driver_count}, ")
    
    return driver_count

def create_fleet_size_summary_table(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime columns are properly formatted
    trip_data = trip_data.copy()
    trip_data['date'] = trip_data['pickup_datetime'].dt.date
    trip_data['time'] = trip_data['pickup_datetime'].dt.time
    
    # Get unique dates
    unique_dates = sorted(trip_data['date'].unique())
    
    results = []
    
    for date in unique_dates:
        print(f"Processing {date}...")
        
        # Filter trips for this date
        daily_trips = trip_data[trip_data['date'] == date].copy()
        
        if len(daily_trips) == 0:
            continue
            
        # Full day fleet size
        fleet_size_full_day = calculate_minimum_drivers_vectorized(daily_trips, zone_time_matrix)
        
        # Peak Morning: 07:00:00 until 10:00:00
        morning_trips = daily_trips[
            (daily_trips['pickup_datetime'].dt.time >= pd.to_datetime('07:00:00').time()) &
            (daily_trips['pickup_datetime'].dt.time < pd.to_datetime('10:00:00').time())
        ]
        fleet_size_peak_morning = calculate_minimum_drivers_vectorized(morning_trips, zone_time_matrix) if len(morning_trips) > 0 else 0
        
        # Peak Evening: 17:00:00 until 20:00:00
        evening_trips = daily_trips[
            (daily_trips['pickup_datetime'].dt.time >= pd.to_datetime('17:00:00').time()) &
            (daily_trips['pickup_datetime'].dt.time < pd.to_datetime('20:00:00').time())
        ]
        fleet_size_peak_evening = calculate_minimum_drivers_vectorized(evening_trips, zone_time_matrix) if len(evening_trips) > 0 else 0
        
        # Off-Peak: 10:00:00 until 17:00:00
        off_peak_trips = daily_trips[
            (daily_trips['pickup_datetime'].dt.time >= pd.to_datetime('10:00:00').time()) &
            (daily_trips['pickup_datetime'].dt.time < pd.to_datetime('17:00:00').time())
        ]
        fleet_size_off_peak = calculate_minimum_drivers_vectorized(off_peak_trips, zone_time_matrix) if len(off_peak_trips) > 0 else 0
        
        # Get weekday
        weekday = pd.to_datetime(date).strftime('%A')
        
        results.append({
            'date': date,
            'Fleet_Size_Full_Day': fleet_size_full_day,
            'Fleet_Size_Peak_Morning': fleet_size_peak_morning,
            'Fleet_Size_Peak_Evening': fleet_size_peak_evening,
            'Fleet_Size_Off_Peak': fleet_size_off_peak,
            'weekday': weekday
        })
        print(f"Finished processing {date}.")

    return pd.DataFrame(results)


def create_fleet_size_summary_table_improved(trip_data: pd.DataFrame, zone_time_matrix: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime columns are properly formatted
    trip_data = trip_data.copy()
    trip_data['date'] = trip_data['pickup_datetime'].dt.date
    trip_data['time'] = trip_data['pickup_datetime'].dt.time
    
    # Get unique dates
    unique_dates = sorted(trip_data['date'].unique())
    
    results = []
    
    for date in unique_dates:
        print(f"Processing {date}...")
        
        # Filter trips for this date
        daily_trips = trip_data[trip_data['date'] == date].copy()
        
        if len(daily_trips) == 0:
            continue
            
        # Full day fleet size
        fleet_size_full_day = calculate_minimum_drivers_improved(daily_trips, zone_time_matrix)
        
        # Peak Morning: 07:00:00 until 10:00:00
        morning_trips = daily_trips[
            (daily_trips['pickup_datetime'].dt.time >= pd.to_datetime('07:00:00').time()) &
            (daily_trips['pickup_datetime'].dt.time < pd.to_datetime('10:00:00').time())
        ]
        fleet_size_peak_morning = calculate_minimum_drivers_improved(morning_trips, zone_time_matrix) if len(morning_trips) > 0 else 0
        
        # Peak Evening: 17:00:00 until 20:00:00
        evening_trips = daily_trips[
            (daily_trips['pickup_datetime'].dt.time >= pd.to_datetime('17:00:00').time()) &
            (daily_trips['pickup_datetime'].dt.time < pd.to_datetime('20:00:00').time())
        ]
        fleet_size_peak_evening = calculate_minimum_drivers_improved(evening_trips, zone_time_matrix) if len(evening_trips) > 0 else 0
        
        # Off-Peak: 10:00:00 until 17:00:00
        off_peak_trips = daily_trips[
            (daily_trips['pickup_datetime'].dt.time >= pd.to_datetime('10:00:00').time()) &
            (daily_trips['pickup_datetime'].dt.time < pd.to_datetime('17:00:00').time())
        ]
        fleet_size_off_peak = calculate_minimum_drivers_improved(off_peak_trips, zone_time_matrix) if len(off_peak_trips) > 0 else 0
        
        # Get weekday
        weekday = pd.to_datetime(date).strftime('%A')
        
        results.append({
            'date': date,
            'Fleet_Size_Full_Day': fleet_size_full_day,
            'Fleet_Size_Peak_Morning': fleet_size_peak_morning,
            'Fleet_Size_Peak_Evening': fleet_size_peak_evening,
            'Fleet_Size_Off_Peak': fleet_size_off_peak,
            'weekday': weekday
        })
        print(f"Finished processing {date}.")

    return pd.DataFrame(results)
