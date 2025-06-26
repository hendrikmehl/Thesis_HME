import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd

def load_trip_data(file_path='../../data/raw/fhvhv_tripdata_2025-02.parquet'):
    """
    Load FHVHV trip data from parquet file.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        pandas DataFrame with trip data
    """
    # Read the Parquet file
    table = pq.read_table(file_path)
    # Convert to Pandas DataFrame
    df = table.to_pandas()
        
    return df

def load_taxi_zones(file_path='../data/raw/taxi_zones/taxi_zones.shp'):
    taxi_zones = gpd.read_file(file_path)
    if taxi_zones.crs != 'EPSG:4326':
        taxi_zones = taxi_zones.to_crs('EPSG:4326')
    return taxi_zones

def load_data_reports_monthly(file_path='../../data/raw/fhvhv_tripdata_2025-02.parquet'):
    df = pd.read_csv(file_path)
    return df