import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_fleet_size_trends(df, company=None):
    """
    Visualize fleet size trends over time for all time periods using bar charts.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Convert date to datetime for proper plotting
    df['date'] = pd.to_datetime(df['date'])
    
    # Create labels with both date and weekday
    df['date_label'] = df['date'].dt.strftime('%m-%d') + '\n' + df['weekday'].str[:3]
    
    # Create bar chart
    x_pos = np.arange(len(df))
    bars = ax.bar(x_pos, df['Fleet_Size_Full_Day'], 
                  color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color weekend bars differently
    weekend_mask = df['weekday'].isin(['Saturday', 'Sunday'])
    for i, (bar, is_weekend) in enumerate(zip(bars, weekend_mask)):
        if is_weekend:
            bar.set_color('#e74c3c')
    
    ax.set_xlabel('Date (MM-DD) and Weekday', fontsize=12)
    ax.set_ylabel('Fleet Size (Number of Drivers)', fontsize=12)
    ax.set_title(f'{company} - Fleet Size Requirements Over Time - February 2025', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['date_label'], rotation=45, ha='right', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Weekday'),
                      Patch(facecolor='#e74c3c', label='Weekend')]
    ax.legend(handles=legend_elements, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def visualize_time_period_comparison(df, company=None):
    """
    Compare average fleet sizes across different time periods using combined bar and box plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Separate weekdays and weekends
    df_weekdays = df[df['weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
    df_weekend = df[df['weekday'].isin(['Saturday', 'Sunday'])]

    # Calculate averages for both weekdays and weekends
    weekday_avg = {
        'Peak Morning': df_weekdays['Fleet_Size_Peak_Morning'].mean(),
        'Off-Peak': df_weekdays['Fleet_Size_Off_Peak'].mean(),
        'Peak Evening': df_weekdays['Fleet_Size_Peak_Evening'].mean(),
    }
    
    weekend_avg = {
        'Peak Morning': df_weekend['Fleet_Size_Peak_Morning'].mean(),
        'Off-Peak': df_weekend['Fleet_Size_Off_Peak'].mean(),
        'Peak Evening': df_weekend['Fleet_Size_Peak_Evening'].mean(),
    }
    
    # Combined bar chart
    periods = list(weekday_avg.keys())
    x = np.arange(len(periods))
    width = 0.35
    
    colors = ['#3498db', '#e74c3c']
    bars1 = ax1.bar(x - width/2, list(weekday_avg.values()), width, 
                   label='Weekdays', color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, list(weekend_avg.values()), width,
                   label='Weekends', color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Average Fleet Size', fontsize=12)
    ax1.set_title(f'{company} - Average Fleet Size by Time Period - Weekdays vs Weekends', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Combined box plot
    weekday_data = [
        df_weekdays['Fleet_Size_Peak_Morning'],
        df_weekdays['Fleet_Size_Off_Peak'],
        df_weekdays['Fleet_Size_Peak_Evening']
    ]
    
    weekend_data = [
        df_weekend['Fleet_Size_Peak_Morning'],
        df_weekend['Fleet_Size_Off_Peak'],
        df_weekend['Fleet_Size_Peak_Evening']
    ]
    
    # Create positions for box plots
    positions_weekdays = [1, 3, 5]
    positions_weekends = [1.5, 3.5, 5.5]
    
    box1 = ax2.boxplot(weekday_data, positions=positions_weekdays, widths=0.4, 
                      patch_artist=True, medianprops={'color': 'black', 'linewidth': 2})
    box2 = ax2.boxplot(weekend_data, positions=positions_weekends, widths=0.4, 
                      patch_artist=True, medianprops={'color': 'black', 'linewidth': 2})
    
    # Color the boxes
    for patch in box1['boxes']:
        patch.set_facecolor(colors[0])
        patch.set_alpha(0.8)
    
    for patch in box2['boxes']:
        patch.set_facecolor(colors[1])
        patch.set_alpha(0.8)
    
    ax2.set_ylabel('Fleet Size Distribution', fontsize=12)
    ax2.set_title(f'{company} - Fleet Size Distribution by Time Period - Weekdays vs Weekends', fontsize=13, fontweight='bold')
    ax2.set_xticks([1.25, 3.25, 5.25])
    ax2.set_xticklabels(periods)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add legend for box plot
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], label='Weekdays'),
                      Patch(facecolor=colors[1], label='Weekends')]
    ax2.legend(handles=legend_elements)
    
    # Set equal y-axes
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

def visualize_weekday_patterns(df, company=None):
    """
    Analyze and visualize weekday vs weekend patterns in fleet size requirements.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Add weekday categories
    df_viz = df.copy()
    df_viz['is_weekend'] = df_viz['weekday'].isin(['Saturday', 'Sunday'])
    df_viz['day_type'] = df_viz['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
    
    # Define day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_viz['weekday'] = pd.Categorical(df_viz['weekday'], categories=day_order, ordered=True)
    df_viz = df_viz.sort_values('weekday')
    
    # 1. Fleet size by weekday (Full Day)
    weekday_avg = df_viz.groupby('weekday')['Fleet_Size_Full_Day'].mean()
    colors = ['#3498db' if day not in ['Saturday', 'Sunday'] else '#e74c3c' for day in weekday_avg.index]
    
    bars = ax1.bar(weekday_avg.index, weekday_avg.values, color=colors, alpha=0.8)
    ax1.set_ylabel('Average Fleet Size', fontsize=11)
    ax1.set_title('Full Day Fleet Size by Weekday', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Weekday vs Weekend comparison
    day_type_comparison = df_viz.groupby('day_type')[
        ['Fleet_Size_Peak_Morning', 'Fleet_Size_Off_Peak', 'Fleet_Size_Peak_Evening']
    ].mean()
    
    x = np.arange(len(day_type_comparison.columns))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, day_type_comparison.loc['Weekday'], width, 
                   label='Weekday', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, day_type_comparison.loc['Weekend'], width,
                   label='Weekend', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Average Fleet Size', fontsize=11)
    ax2.set_title('Weekday vs Weekend Fleet Size by Time Period', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Peak Morning', 'Off-Peak', 'Peak Evening'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Peak hours heatmap
    peak_data = df_viz.pivot_table(
        values=['Fleet_Size_Peak_Morning', 'Fleet_Size_Peak_Evening'], 
        index='weekday', 
        aggfunc='mean'
    )
    peak_data.columns = ['Morning Peak', 'Evening Peak']
    
    sns.heatmap(peak_data.T, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=ax3, cbar_kws={'label': 'Fleet Size'})
    ax3.set_title('Peak Hours Fleet Size Heatmap', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Day of Week', fontsize=11)
    ax3.set_ylabel('Peak Period', fontsize=11)
    
    # 4. Daily variation coefficient
    daily_stats = df_viz.groupby('weekday').agg({
        'Fleet_Size_Full_Day': ['mean', 'std']
    }).round(2)
    daily_stats.columns = ['Mean', 'Std Dev']
    daily_stats['CV'] = (daily_stats['Std Dev'] / daily_stats['Mean'] * 100).round(1)
    
    bars = ax4.bar(daily_stats.index, daily_stats['CV'], 
                  color=['#3498db' if day not in ['Saturday', 'Sunday'] else '#e74c3c' 
                         for day in daily_stats.index], alpha=0.8)
    ax4.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax4.set_title('Fleet Size Variability by Day', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def visualize_fleet_size_summary(df, company=None):
    """
    Main function that calls all visualization functions.
    """
    print("Generating fleet size visualizations...")
    print("1. Fleet Size Trends Over Time")
    visualize_fleet_size_trends(df, company)

    print("2. Weekday Time Period Comparison")
    visualize_time_period_comparison(df, company)

    print("3. Weekday Patterns Analysis")
    visualize_weekday_patterns(df, company)