import pandas as pd

def calculate_active_rides_per_minute(df, target_date='2025-02-01'):
    """
    Calculate the number of active rides for every minute on a specific date.
    A ride is active if pickup_datetime <= current_time < dropoff_datetime
    """    
    # Define the target date range (full day)
    start_date = pd.to_datetime(target_date)
    end_date = start_date + pd.Timedelta(days=1)
        
    # Filter rides on which the dropoff_datetime is on the target date
    relevant_rides = df[
        (df['pickup_datetime'] < end_date) &
        (df['dropoff_datetime'] > start_date)
    ].copy()
    
    # Create time series for every minute of the day
    time_range = pd.date_range(start=start_date, end=end_date, freq='1min')[:-1]  # Exclude last point (next day)
    
    # Calculate active rides for each minute
    active_rides_per_minute = []

    # For each minute, count rides that started before or at this minute and end after this minute
    for current_time in time_range:
        active_count = len(relevant_rides[
            (relevant_rides['pickup_datetime'] <= current_time) &
            (relevant_rides['dropoff_datetime'] > current_time)
        ])
        active_rides_per_minute.append(active_count)
    
    # Create result DataFrame
    active_rides_df = pd.DataFrame({
        'datetime': time_range,
        'active_rides': active_rides_per_minute
    })
    
    # Add time-based features for analysis
    active_rides_df['hour'] = active_rides_df['datetime'].dt.hour
    active_rides_df['minute'] = active_rides_df['datetime'].dt.minute
    active_rides_df['time_of_day'] = active_rides_df['datetime'].dt.strftime('%H:%M')
    
    return active_rides_df

def analyze_active_rides_by_company(df, target_date='2025-02-01'):
    """
    Calculate active rides per minute separately for each company
    """
    
    results = {}
    
    for company in ['Uber', 'Lyft']:
        if company in df['company'].values:
            company_data = df[df['company'] == company]
            company_active_rides = calculate_active_rides_per_minute(company_data, target_date)
            results[company] = company_active_rides
        else:
            print(f"{company} not found in data")
    
    return results

def plot_active_trips_over_day(active_rides_by_company, title='Active Trips Over the Day'):
    """
    Plot active trips over the day for each company.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    for company, rides_df in active_rides_by_company.items():
        plt.plot(rides_df['datetime'], rides_df['active_rides'], label=company)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Number of Active Trips')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_active_trips_by_weekday(df, company=None):
    """
    Create 7 line plots, one for each weekday (Monday to Sunday).
    Each plot shows multiple lines for each occurrence of that weekday in the dataset.
    
    Args:
        df: DataFrame with trip data
        company: Optional string to filter by company ('Uber' or 'Lyft')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Filter by company if specified
    if company:
        df = df[df['company'] == company]
    
    # Get unique dates in the dataset
    df['date'] = df['pickup_datetime'].dt.date
    unique_dates = sorted(df['date'].unique())
    
    # Group dates by weekday
    weekday_dates = {i: [] for i in range(7)}  # 0=Monday, 6=Sunday
    for date in unique_dates:
        weekday = pd.to_datetime(date).weekday()
        weekday_dates[weekday].append(date)
    
    # Weekday names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for weekday in range(7):
        ax = axes[weekday]
        dates_for_weekday = weekday_dates[weekday]
        
        if not dates_for_weekday:
            ax.set_title(f'{weekday_names[weekday]} - No data')
            continue
        
        for date in dates_for_weekday:
            # Calculate active rides for this specific date
            date_str = date.strftime('%Y-%m-%d')
            active_rides_df = calculate_active_rides_per_minute(df, target_date=date_str)
            
            # Create time of day (hour:minute) for x-axis
            time_of_day = active_rides_df['datetime'].dt.hour + active_rides_df['datetime'].dt.minute / 60.0
            
            # Plot line for this date
            ax.plot(time_of_day, active_rides_df['active_rides'], 
                   label=date_str, alpha=0.7, linewidth=1)
        
        ax.set_title(f'{weekday_names[weekday]} ({len(dates_for_weekday)} days)')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Active Rides')
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)
        
        # Add legend if there are multiple dates
        if len(dates_for_weekday) > 1:
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide the 8th subplot
    axes[7].set_visible(False)
    
    title = f'Active Rides by Weekday'
    if company:
        title += f' - {company}'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.show()

