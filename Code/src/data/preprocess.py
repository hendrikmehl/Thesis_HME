import pandas as pd

def preprocess_trip_data(df):
    df = add_company_column(df)
    df = remove_unnecessary_columns(df)
    df = convert_datetime_columns(df)
    null_value_summary(df)
    return df

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
    columns_to_remove = ['hvfhs_license_num', 'dispatching_base_num', 'orignating_base_num']
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

def null_value_summary(df):
    # Check which companies have null values in on_scene_datetime
    print("Null values in 'on_scene_datetime' by company:")
    on_scene_nulls = df[df['on_scene_datetime'].isnull()]['company'].value_counts()
    print(on_scene_nulls)
    print(f"Total null values in on_scene_datetime: {df['on_scene_datetime'].isnull().sum()}")
    company_data = df[df['company'] == 'Lyft']
    total_company_rows = len(company_data)
    on_scene_nulls = company_data['on_scene_datetime'].isnull().sum()
    on_scene_pct = (on_scene_nulls / total_company_rows * 100).round(2)
    print(f"Lyft:")
    print(f"  on_scene_datetime: {on_scene_nulls:,} ({on_scene_pct}%)")
    print(f"  Total rows: {total_company_rows:,}")
    print()