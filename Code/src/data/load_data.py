import pyarrow.parquet as pq
import pandas as pd

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
