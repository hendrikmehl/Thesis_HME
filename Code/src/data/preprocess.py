import pandas as pd

def preprocess_trip_data(df):
    df = add_company_column(df)
    df = remove_unnecessary_columns(df)
    df = convert_datetime_columns(df)
    # null_value_summary(df)
    df = add_weekday_column(df)
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
    print("=" * 60)
    print("NULL VALUE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Overall null value statistics
    total_rows = len(df)
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / total_rows * 100).round(2)
    
    print(f"Total rows in dataset: {total_rows:,}")
    print("\nNull values by column:")
    print("-" * 40)
    
    for col in df.columns:
        if null_counts[col] > 0:
            print(f"{col}: {null_counts[col]:,} ({null_percentages[col]}%)")
    
    # Identify rows with any null values
    rows_with_nulls = df.isnull().any(axis=1)
    total_rows_with_nulls = rows_with_nulls.sum()
    
    print(f"\nRows with at least one null value: {total_rows_with_nulls:,} ({(total_rows_with_nulls/total_rows*100):.2f}%)")
    
    # Analyze null patterns by company
    if 'company' in df.columns:
        print("\nNull value patterns by company:")
        print("-" * 40)
        for company in df['company'].dropna().unique():
            company_data = df[df['company'] == company]
            company_rows = len(company_data)
            
            print(f"\n{company}:")
            print(f"  Total rows: {company_rows:,}")
            
            for col in df.columns:
                if col != 'company':
                    null_count = company_data[col].isnull().sum()
                    if null_count > 0:
                        null_pct = (null_count / company_rows * 100)
                        print(f"  {col}: {null_count:,} ({null_pct:.2f}%)")
    
    # Analyze temporal patterns in null values
    if 'pickup_datetime' in df.columns:
        print("\nTemporal patterns of null values:")
        print("-" * 40)
        
        # Check if nulls cluster in specific time periods
        df_with_nulls = df[df.isnull().any(axis=1)]
        if len(df_with_nulls) > 0 and not df_with_nulls['pickup_datetime'].isnull().all():
            print(f"Date range of entries with nulls:")
            print(f"  Earliest: {df_with_nulls['pickup_datetime'].min()}")
            print(f"  Latest: {df_with_nulls['pickup_datetime'].max()}")
            
            # Check if nulls are more common on certain weekdays
            if 'pickup_weekday' in df.columns:
                null_by_weekday = df_with_nulls['pickup_weekday'].value_counts()
                print(f"\nNull entries by weekday:")
                for day, count in null_by_weekday.items():
                    total_day_entries = df[df['pickup_weekday'] == day].shape[0]
                    pct = (count / total_day_entries * 100) if total_day_entries > 0 else 0
                    print(f"  {day}: {count:,} ({pct:.2f}% of all {day} entries)")
    
    # Most common null value combinations
    print("\nMost common null value patterns:")
    print("-" * 40)
    null_patterns = df.isnull().value_counts().head(5)
    for pattern, count in null_patterns.items():
        if any(pattern):  # Only show patterns with at least one null
            null_cols = [df.columns[i] for i, is_null in enumerate(pattern) if is_null]
            print(f"  Null in {null_cols}: {count:,} rows ({(count/total_rows*100):.2f}%)")
    
    print("\n" + "=" * 60)

def add_weekday_column(df):
    # Add weekday column
    df['pickup_weekday'] = df['pickup_datetime'].dt.day_name()
    df['dropoff_weekday'] = df['dropoff_datetime'].dt.day_name()
    
    return df

def remove_outliers(df):
    # Remove outliers based on pickup and dropoff datetime
    df = df[(df['pickup_datetime'] < df['dropoff_datetime']) & (df['pickup_datetime'] > '2020-01-01')]
    
    return df