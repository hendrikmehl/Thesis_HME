import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap_core(
    df,
    x_col,
    y_col,
    x_bin=None,
    y_bin=None,
    x_label=None,
    y_label=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    title=None,
    ax=None,
    x_axis_limits=None,
    y_axis_limits=None
):
    # Remove negative fares
    if 'base_passenger_fare' in df.columns:
        df = df[df['base_passenger_fare'] >= 0]

    df = df.copy()
    # Bin columns if requested
    if x_bin is not None:
        df[x_col + '_binned'] = (df[x_col] * x_bin).round() / x_bin
        x_plot_col = x_col + '_binned'
    else:
        x_plot_col = x_col
    if y_bin is not None:
        df[y_col + '_binned'] = (df[y_col] * y_bin).round() / y_bin
        y_plot_col = y_col + '_binned'
    else:
        y_plot_col = y_col

    # Filter by bounds
    if x_min is not None:
        df = df[df[x_plot_col] >= x_min]
    if x_max is not None:
        df = df[df[x_plot_col] <= x_max]
    if y_min is not None:
        df = df[df[y_plot_col] >= y_min]
    if y_max is not None:
        df = df[df[y_plot_col] <= y_max]

    # --- Ensure all bins are present and axes are aligned ---
    # Calculate bin edges
    if x_bin is not None:
        x_start = x_min if x_min is not None else df[x_plot_col].min()
        x_end = x_max if x_max is not None else df[x_plot_col].max()
        x_bins = np.round(np.arange(x_start, x_end + 1e-8, 1.0 / x_bin), 1)
    else:
        x_bins = np.sort(df[x_plot_col].unique())
    if y_bin is not None:
        y_start = y_min if y_min is not None else df[y_plot_col].min()
        y_end = y_max if y_max is not None else df[y_plot_col].max()
        y_bins = np.round(np.arange(y_start, y_end + 1e-8, 1.0 / y_bin), 1)
    else:
        y_bins = np.sort(df[y_plot_col].unique())

    # Pivot table and reindex to full bin range
    pivot_table = df.pivot_table(index=y_plot_col, columns=x_plot_col, aggfunc='size', fill_value=0)
    pivot_table = pivot_table.reindex(index=y_bins, columns=x_bins, fill_value=0)
    log_pivot_table = np.log1p(pivot_table)

    # Plot
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    sns.heatmap(log_pivot_table, fmt=".1f", cmap="viridis", ax=ax)

    # Set axis labels and title
    ax.set_xlabel(x_label if x_label else x_col)
    ax.set_ylabel(y_label if y_label else y_col)
    ax.invert_yaxis()
    if title:
        ax.set_title(title)

    # Set axis limits if provided
    if x_axis_limits is not None:
        ax.set_xlim(x_axis_limits)
    if y_axis_limits is not None:
        ax.set_ylim(y_axis_limits)

    # --- Set ticks at regular intervals and align with bins ---
    if x_bin is not None:
        # Show every Nth tick (e.g., every 100 units)
        show_every_x = 5  # units
        x_tick_vals = np.arange(x_bins[0], x_bins[-1] + 1e-8, show_every_x)
        x_tick_idx = [np.argmin(np.abs(x_bins - val)) for val in x_tick_vals]
        ax.set_xticks(x_tick_idx)
        ax.set_xticklabels([f"{x_bins[idx]:.1f}" for idx in x_tick_idx], rotation=45)
    if y_bin is not None:
        show_every_y = 5  # units
        y_tick_vals = np.arange(y_bins[0], y_bins[-1] + 1e-8, show_every_y)
        y_tick_idx = [np.argmin(np.abs(y_bins - val)) for val in y_tick_vals]
        ax.set_yticks(y_tick_idx)
        ax.set_yticklabels([f"{y_bins[idx]:.1f}" for idx in y_tick_idx])

    return ax

def create_pricing_distance_heatmap(
    df,
    fare_min=0,
    fare_max=60,
    miles_min=0,
    miles_max=25,
    x_axis_limits=None,
    y_axis_limits=None
):
    df = df.copy()
    # Filter for trip_miles <= 25 and fare <= 60
    df = df[(df['trip_miles'] <= miles_max) & (df['base_passenger_fare'] <= fare_max)]

    plot_heatmap_core(
        df,
        x_col='base_passenger_fare',
        y_col='trip_miles',
        x_bin=10,
        y_bin=10,
        x_label='Fare',
        y_label='Trip Distance',
        x_min=fare_min,
        x_max=fare_max,
        y_min=miles_min,
        y_max=miles_max,
        title='Logarithmic Heatmap of Trip Distance and Fare',
        x_axis_limits=x_axis_limits,
        y_axis_limits=y_axis_limits
    )
    plt.show()

def create_pricing_speed_heatmap(
    df,
    fare_min=0,
    fare_max=60,
    speed_min=0,
    speed_max=35,
    x_axis_limits=None,
    y_axis_limits=None
):
    df = df.copy()
    # Calculate trip speed
    df['trip_speed'] = df['trip_miles'] / (df['trip_time'] / 3600)  # Convert trip_time to hours

    # Filter for trip_speed <= speed_max and fare <= 60
    df = df[(df['trip_speed'] <= speed_max) & (df['base_passenger_fare'] <= fare_max)]

    plot_heatmap_core(
        df,
        x_col='base_passenger_fare',
        y_col='trip_speed',
        x_bin=10,
        y_bin=10,
        x_label='Fare',
        y_label='Trip Speed',
        x_min=fare_min,
        x_max=fare_max,
        y_min=speed_min,
        y_max=speed_max,
        title='Logarithmic Heatmap of Trip Speed and Fare',
        x_axis_limits=x_axis_limits,
        y_axis_limits=y_axis_limits
    )
    plt.show()


def create_pricing_duration_heatmap(
    df,
    fare_min=0,
    fare_max=60,
    time_min=0,
    time_max=1800,
    x_axis_limits=None,
    y_axis_limits=None
):
    df = df.copy()
    # Filter for trip_time <= 1800 and fare <= 34
    df = df[(df['trip_time'] <= time_max) & (df['base_passenger_fare'] <= fare_max)]

    plot_heatmap_core(
        df,
        x_col='base_passenger_fare',
        y_col='trip_time',
        x_bin=10,
        y_bin=None,
        x_label='Fare',
        y_label='Trip Time',
        x_min=fare_min,
        x_max=fare_max,
        y_min=time_min,
        y_max=time_max,
        title='Logarithmic Heatmap of Trip Time and Fare',
        x_axis_limits=x_axis_limits,
        y_axis_limits=y_axis_limits
    )
    plt.show()

def create_pricing_distance_heatmap_by_weekday_and_company(
    df,
    fare_min=0,
    fare_max=60,
    miles_min=0,
    miles_max=25,
    x_axis_limits=None,
    y_axis_limits=None
):
    df = df.copy()
    df = df[(df['trip_miles'] <= miles_max) & (df['base_passenger_fare'] <= fare_max)]

    companies = sorted(df['company'].unique())
    weekdays = sorted(df['pickup_weekday'].unique())
    n_cols = len(companies)
    n_rows = len(weekdays)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for i, weekday in enumerate(weekdays):
        for j, company in enumerate(companies):
            subset = df[(df['company'] == company) & (df['pickup_weekday'] == weekday)]
            title = f'Company: {company}, Weekday: {weekday}'
            plot_heatmap_core(
                subset,
                x_col='base_passenger_fare',
                y_col='trip_miles',
                x_bin=10,
                y_bin=None,
                x_label='Fare',
                y_label='Trip Distance',
                x_min=fare_min,
                x_max=fare_max,
                y_min=miles_min,
                y_max=miles_max,
                title=title,
                ax=axes[i, j],
                x_axis_limits=x_axis_limits,
                y_axis_limits=y_axis_limits
            )
    plt.tight_layout()
    plt.show()

def create_pricing_duration_heatmap_by_weekday_and_company(
    df,
    fare_min=0,
    fare_max=34,
    time_min=0,
    time_max=1800,
    x_axis_limits=34,
    y_axis_limits=1800
):
    df = df.copy()
    df = df[(df['trip_time'] <= time_max) & (df['base_passenger_fare'] <= fare_max)]

    companies = sorted(df['company'].unique())
    weekdays = sorted(df['pickup_weekday'].unique())
    n_cols = len(companies)
    n_rows = len(weekdays)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for i, weekday in enumerate(weekdays):
        for j, company in enumerate(companies):
            subset = df[(df['company'] == company) & (df['pickup_weekday'] == weekday)]
            title = f'Company: {company}, Weekday: {weekday}'
            plot_heatmap_core(
                subset,
                x_col='base_passenger_fare',
                y_col='trip_time',
                x_bin=10,
                y_bin=None,
                x_label='Fare',
                y_label='Trip Time',
                x_min=fare_min,
                x_max=fare_max,
                y_min=time_min,
                y_max=time_max,
                title=title,
                ax=axes[i, j],
                x_axis_limits=x_axis_limits,
                y_axis_limits=y_axis_limits
            )
    plt.tight_layout()
    plt.show()

def create_pricing_distance_heatmap_by_hour_and_company(
    df,
    fare_min=0,
    fare_max=60,
    miles_min=0,
    miles_max=25,
    x_bin=1,
    y_bin=1,
    x_axis_limits=None,
    y_axis_limits=None
):
    """
    Plot distance-fare heatmaps for each hour of day and company.
    Assumes 'pickup_datetime' is a datetime column.
    """
    df = df.copy()
    df = df[(df['trip_miles'] <= miles_max) & (df['base_passenger_fare'] <= fare_max)]
    if 'pickup_datetime' in df.columns:
        df['pickup_hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour
    else:
        raise ValueError("pickup_datetime column required for hourly analysis.")

    companies = sorted(df['company'].unique())
    hours = range(24)
    n_cols = len(companies)
    n_rows = 24
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2 * n_rows), squeeze=False)
    for i, hour in enumerate(hours):
        for j, company in enumerate(companies):
            subset = df[(df['company'] == company) & (df['pickup_hour'] == hour)]
            title = f'Company: {company}, Hour: {hour:02d}:00'
            plot_heatmap_core(
                subset,
                x_col='base_passenger_fare',
                y_col='trip_miles',
                x_bin=x_bin,
                y_bin=y_bin,
                x_label='Fare',
                y_label='Trip Distance',
                x_min=fare_min,
                x_max=fare_max,
                y_min=miles_min,
                y_max=miles_max,
                title=title,
                ax=axes[i, j],
                x_axis_limits=x_axis_limits,
                y_axis_limits=y_axis_limits
            )
    plt.tight_layout()
    plt.show()

def create_pricing_duration_heatmap_by_pickup_zone_and_company(
    df,
    fare_min=0,
    fare_max=34,
    time_min=0,
    time_max=1800,
    x_bin=1,
    y_bin=None,
    x_axis_limits=None,
    y_axis_limits=None,
    max_zones=10
):
    """
    Plot duration-fare heatmaps for each pickup zone and company.
    Only shows up to max_zones most frequent zones.
    Assumes 'pickup_zone' column exists.
    """
    df = df.copy()
    df = df[(df['trip_time'] <= time_max) & (df['base_passenger_fare'] <= fare_max)]
    if 'pickup_zone' not in df.columns:
        raise ValueError("pickup_zone column required for zone analysis.")

    # Use only the most frequent pickup zones
    top_zones = df['pickup_zone'].value_counts().nlargest(max_zones).index.tolist()
    companies = sorted(df['company'].unique())
    n_cols = len(companies)
    n_rows = len(top_zones)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    for i, zone in enumerate(top_zones):
        for j, company in enumerate(companies):
            subset = df[(df['company'] == company) & (df['pickup_zone'] == zone)]
            title = f'Company: {company}, Pickup Zone: {zone}'
            plot_heatmap_core(
                subset,
                x_col='base_passenger_fare',
                y_col='trip_time',
                x_bin=x_bin,
                y_bin=y_bin,
                x_label='Fare',
                y_label='Trip Time',
                x_min=fare_min,
                x_max=fare_max,
                y_min=time_min,
                y_max=time_max,
                title=title,
                ax=axes[i, j],
                x_axis_limits=x_axis_limits,
                y_axis_limits=y_axis_limits
            )
    plt.tight_layout()
    plt.show()