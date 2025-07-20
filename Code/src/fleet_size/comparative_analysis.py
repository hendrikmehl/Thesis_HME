import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_fleet_size_share_stacked(uber_df, lyft_df, time_period='Fleet_Size_Full_Day'):
    """
    Visualize fleet size share between Uber and Lyft using stacked bar chart with percentages.
    
    Parameters:
    uber_df (pd.DataFrame): Uber fleet size data
    lyft_df (pd.DataFrame): Lyft fleet size data
    time_period (str): Column name for the time period to analyze (default: 'Fleet_Size_Full_Day')
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Ensure both dataframes have the same date range and are sorted
    uber_df = uber_df.copy()
    lyft_df = lyft_df.copy()
    uber_df['date'] = pd.to_datetime(uber_df['date'])
    lyft_df['date'] = pd.to_datetime(lyft_df['date'])
    
    # Merge dataframes on date
    merged_df = pd.merge(uber_df[['date', time_period, 'weekday']], 
                        lyft_df[['date', time_period]], 
                        on='date', 
                        suffixes=('_uber', '_lyft'))
    
    # Calculate total fleet size and percentages
    merged_df['total_fleet'] = merged_df[f'{time_period}_uber'] + merged_df[f'{time_period}_lyft']
    merged_df['uber_percentage'] = (merged_df[f'{time_period}_uber'] / merged_df['total_fleet']) * 100
    merged_df['lyft_percentage'] = (merged_df[f'{time_period}_lyft'] / merged_df['total_fleet']) * 100
    
    # Create labels with both date and weekday
    merged_df['date_label'] = merged_df['date'].dt.strftime('%m-%d') + '\n' + merged_df['weekday'].str[:3]
    
    # Create stacked bar chart
    x_pos = np.arange(len(merged_df))
    
    # Define colors for weekdays and weekends
    weekend_mask = merged_df['weekday'].isin(['Saturday', 'Sunday'])
    uber_colors = ["#216695" if is_weekend else '#3498db' for is_weekend in weekend_mask]  # Darker blue for weekends
    lyft_colors = ["#912b20" if is_weekend else '#e74c3c' for is_weekend in weekend_mask]  # Darker red for weekends
    
    # Create the stacked bars
    bars1 = ax.bar(x_pos, merged_df['uber_percentage'], 
                   label='Uber', color=uber_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos, merged_df['lyft_percentage'], 
                   bottom=merged_df['uber_percentage'],
                   label='Lyft', color=lyft_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Date (MM-DD) and Weekday', fontsize=12)
    ax.set_ylabel('Fleet Size Share (%)', fontsize=12)
    
    # Format title based on time period
    period_name = time_period.replace('Fleet_Size_', '').replace('_', ' ').title()
    ax.set_title(f'Uber vs Lyft Fleet Size Share - {period_name} - February 2025', 
                fontsize=14, fontweight='bold')
    
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(merged_df['date_label'], rotation=45, ha='right', fontsize=10)
    
    # Add percentage labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Uber percentage (bottom part)
        uber_pct = merged_df.iloc[i]['uber_percentage']
        if uber_pct > 5:  # Only show label if segment is large enough
            ax.text(bar1.get_x() + bar1.get_width()/2., uber_pct/2,
                   f'{uber_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=11, color='white', rotation=90)
        
        # Lyft percentage (top part)
        lyft_pct = merged_df.iloc[i]['lyft_percentage']
        if lyft_pct > 5:  # Only show label if segment is large enough
            ax.text(bar2.get_x() + bar2.get_width()/2., uber_pct + lyft_pct/2,
                   f'{lyft_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=11, color='white', rotation=90)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Add summary statistics
    avg_uber = merged_df['uber_percentage'].mean()
    avg_lyft = merged_df['lyft_percentage'].mean()
    
    textstr = f'Average Fleet Size Share:\nUber: {avg_uber:.1f}%\nLyft: {avg_lyft:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Fleet Size Share Analysis - {period_name} ===")
    print(f"Average Uber fleet size share: {avg_uber:.1f}%")
    print(f"Average Lyft fleet size share: {avg_lyft:.1f}%")
    print(f"Uber fleet size share range: {merged_df['uber_percentage'].min():.1f}% - {merged_df['uber_percentage'].max():.1f}%")
    print(f"Lyft fleet size share range: {merged_df['lyft_percentage'].min():.1f}% - {merged_df['lyft_percentage'].max():.1f}%")


def visualize_fleet_size_share_time_periods(uber_df, lyft_df):
    """
    Visualize fleet size share across different time periods comparing weekdays vs weekends using stacked bar charts.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    uber_df = uber_df.copy()
    lyft_df = lyft_df.copy()
    uber_df['date'] = pd.to_datetime(uber_df['date'])
    lyft_df['date'] = pd.to_datetime(lyft_df['date'])
    
    # Merge dataframes
    merged_df = pd.merge(uber_df[['date', 'Fleet_Size_Peak_Morning', 'Fleet_Size_Off_Peak', 
                                 'Fleet_Size_Peak_Evening', 'weekday']], 
                        lyft_df[['date', 'Fleet_Size_Peak_Morning', 'Fleet_Size_Off_Peak', 
                                'Fleet_Size_Peak_Evening']], 
                        on='date', 
                        suffixes=('_uber', '_lyft'))
    
    # Separate weekdays and weekends
    weekday_mask = ~merged_df['weekday'].isin(['Saturday', 'Sunday'])
    df_weekdays = merged_df[weekday_mask]
    df_weekend = merged_df[~weekday_mask]
    
    # Calculate averages and percentages for weekdays
    weekday_totals = {}
    weekday_uber_pct = {}
    weekday_lyft_pct = {}
    
    for period in ['Peak_Morning', 'Off_Peak', 'Peak_Evening']:
        uber_col = f'Fleet_Size_{period}_uber'
        lyft_col = f'Fleet_Size_{period}_lyft'
        
        total = df_weekdays[uber_col].mean() + df_weekdays[lyft_col].mean()
        weekday_totals[period] = total
        weekday_uber_pct[period] = (df_weekdays[uber_col].mean() / total) * 100
        weekday_lyft_pct[period] = (df_weekdays[lyft_col].mean() / total) * 100
    
    # Calculate averages and percentages for weekends
    weekend_totals = {}
    weekend_uber_pct = {}
    weekend_lyft_pct = {}
    
    for period in ['Peak_Morning', 'Off_Peak', 'Peak_Evening']:
        uber_col = f'Fleet_Size_{period}_uber'
        lyft_col = f'Fleet_Size_{period}_lyft'
        
        total = df_weekend[uber_col].mean() + df_weekend[lyft_col].mean()
        weekend_totals[period] = total
        weekend_uber_pct[period] = (df_weekend[uber_col].mean() / total) * 100
        weekend_lyft_pct[period] = (df_weekend[lyft_col].mean() / total) * 100
    
    # Plot: Side-by-side comparison of weekdays and weekends
    periods = ['Peak Morning', 'Off-Peak', 'Peak Evening']
    x = np.arange(len(periods))
    width = 0.35
    
    uber_values_wd = [weekday_uber_pct['Peak_Morning'], weekday_uber_pct['Off_Peak'], weekday_uber_pct['Peak_Evening']]
    lyft_values_wd = [weekday_lyft_pct['Peak_Morning'], weekday_lyft_pct['Off_Peak'], weekday_lyft_pct['Peak_Evening']]
    
    uber_values_we = [weekend_uber_pct['Peak_Morning'], weekend_uber_pct['Off_Peak'], weekend_uber_pct['Peak_Evening']]
    lyft_values_we = [weekend_lyft_pct['Peak_Morning'], weekend_lyft_pct['Off_Peak'], weekend_lyft_pct['Peak_Evening']]
    
    # Weekday bars (left)
    bars1_wd = ax.bar(x - width/2, uber_values_wd, width, label='Uber (Weekdays)', 
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2_wd = ax.bar(x - width/2, lyft_values_wd, width, bottom=uber_values_wd, 
                      label='Lyft (Weekdays)', color='#e74c3c', alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
    
    # Weekend bars (right) - darker colors
    bars1_we = ax.bar(x + width/2, uber_values_we, width, label='Uber (Weekends)', 
                      color='#216695', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2_we = ax.bar(x + width/2, lyft_values_we, width, bottom=uber_values_we, 
                      label='Lyft (Weekends)', color='#912b20', alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Fleet Size Share (%)', fontsize=12)
    ax.set_title('Fleet Size Share by Time Period - Weekdays vs Weekends', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels for weekdays (larger font)
    for i, (bar1, bar2) in enumerate(zip(bars1_wd, bars2_wd)):
        # Uber percentage
        uber_pct = uber_values_wd[i]
        if uber_pct > 5:
            ax.text(bar1.get_x() + bar1.get_width()/2., uber_pct/2,
                    f'{uber_pct:.1f}%', ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='white')
        
        # Lyft percentage
        lyft_pct = lyft_values_wd[i]
        if lyft_pct > 5:
            ax.text(bar2.get_x() + bar2.get_width()/2., uber_pct + lyft_pct/2,
                    f'{lyft_pct:.1f}%', ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='white')
    
    # Add percentage labels for weekends (larger font)
    for i, (bar1, bar2) in enumerate(zip(bars1_we, bars2_we)):
        # Uber percentage
        uber_pct = uber_values_we[i]
        if uber_pct > 5:
            ax.text(bar1.get_x() + bar1.get_width()/2., uber_pct/2,
                    f'{uber_pct:.1f}%', ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='white')
        
        # Lyft percentage
        lyft_pct = lyft_values_we[i]
        if lyft_pct > 5:
            ax.text(bar2.get_x() + bar2.get_width()/2., uber_pct + lyft_pct/2,
                    f'{lyft_pct:.1f}%', ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='white')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Fleet Size Share Analysis by Time Period ===")
    print("\nWeekdays:")
    for i, period in enumerate(['Peak Morning', 'Off-Peak', 'Peak Evening']):
        print(f"{period}: Uber {uber_values_wd[i]:.1f}%, Lyft {lyft_values_wd[i]:.1f}%")
    
    print("\nWeekends:")
    for i, period in enumerate(['Peak Morning', 'Off-Peak', 'Peak Evening']):
        print(f"{period}: Uber {uber_values_we[i]:.1f}%, Lyft {lyft_values_we[i]:.1f}%")


def visualize_fleet_size_share_by_weekday(uber_df, lyft_df):
    """
    Visualize average fleet size share for each day of the week for full day only.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    uber_df = uber_df.copy()
    lyft_df = lyft_df.copy()
    uber_df['date'] = pd.to_datetime(uber_df['date'])
    lyft_df['date'] = pd.to_datetime(lyft_df['date'])
    
    # Merge dataframes
    merged_df = pd.merge(uber_df[['date', 'Fleet_Size_Full_Day', 'weekday']], 
                        lyft_df[['date', 'Fleet_Size_Full_Day']], 
                        on='date', 
                        suffixes=('_uber', '_lyft'))
    
    # Define day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Calculate daily averages and market share
    daily_data = []
    
    for day in day_order:
        day_data = merged_df[merged_df['weekday'] == day]
        if len(day_data) > 0:
            avg_uber = day_data['Fleet_Size_Full_Day_uber'].mean()
            avg_lyft = day_data['Fleet_Size_Full_Day_lyft'].mean()
            total = avg_uber + avg_lyft
            
            uber_pct = (avg_uber / total) * 100 if total > 0 else 0
            lyft_pct = (avg_lyft / total) * 100 if total > 0 else 0
            
            daily_data.append({
                'day': day,
                'uber_pct': uber_pct,
                'lyft_pct': lyft_pct
            })
    
    # Extract data for plotting
    days = [d['day'] for d in daily_data]
    uber_percentages = [d['uber_pct'] for d in daily_data]
    lyft_percentages = [d['lyft_pct'] for d in daily_data]
    
    # Create stacked bar chart
    x = np.arange(len(days))
    
    # Define colors for weekdays and weekends
    uber_colors = ['#216695' if day in ['Saturday', 'Sunday'] else '#3498db' for day in days]
    lyft_colors = ['#912b20' if day in ['Saturday', 'Sunday'] else '#e74c3c' for day in days]
    
    bars1 = ax.bar(x, uber_percentages, label='Uber', color=uber_colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, lyft_percentages, bottom=uber_percentages, label='Lyft', 
                  color=lyft_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_ylabel('Fleet Size Share (%)', fontsize=12)
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_title('Uber vs Lyft Fleet Size Share by Day of Week - Full Day - February 2025', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(days, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=11)
    
    # Add percentage labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Uber percentage
        uber_pct = uber_percentages[i]
        if uber_pct > 5:
            ax.text(bar1.get_x() + bar1.get_width()/2., uber_pct/2,
                   f'{uber_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=10, color='white')
        
        # Lyft percentage
        lyft_pct = lyft_percentages[i]
        if lyft_pct > 5:
            ax.text(bar2.get_x() + bar2.get_width()/2., uber_pct + lyft_pct/2,
                   f'{lyft_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=10, color='white')
    
    # Add summary statistics box
    avg_uber = np.mean(uber_percentages)
    avg_lyft = np.mean(lyft_percentages)
    
    textstr = f'Weekly Average:\nUber: {avg_uber:.1f}%\nLyft: {avg_lyft:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Fleet Size Share Analysis by Day of Week - Full Day ===")
    for i, day in enumerate(days):
        print(f"{day}: Uber {uber_percentages[i]:.1f}%, Lyft {lyft_percentages[i]:.1f}%")
    print(f"\nWeekly Average: Uber {avg_uber:.1f}%, Lyft {avg_lyft:.1f}%")
