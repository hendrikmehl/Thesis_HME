import pandas as pd

def preprocess_trip_data(df):
    df = add_company_column(df)
    df = remove_unnecessary_columns(df)
    df = convert_datetime_columns(df)

def add_company_column(df):
    # Add company column
    def get_company(license_num):
        if license_num == 'HV0003':
            return 'Uber'
        elif license_num == 'HV0005':
            return 'Lyft'
        else:
            return None
    
    df['company'] = df['hvfhs_license_num'].apply(get_company)

    return df

def remove_unnecessary_columns(df):
    # Remove unnecessary columns
    columns_to_remove = ['hvfhs_license_num', 'dispatching_base_num']
    df = df.drop(columns=columns_to_remove, errors='ignore')
    
    return df

def convert_datetime_columns(df):
    # Convert datetime columns to datetime type
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
    df['request_datetime'] = pd.to_datetime(df['request_datetime'], errors='coerce')
    df['on_scene_datetime'] = pd.to_datetime(df['on_scene_datetime'], errors='coerce')

    # Drop rows with NaT in datetime columns
    df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])
    
    return df
