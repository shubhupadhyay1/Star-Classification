import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    """Loads dataset and performs initial cleaning."""
    raw = pd.read_csv(file_path)
    
    # Example cleaning steps (customize as needed)
    raw_cleaned = raw.dropna().reset_index(drop=True)
    raw_cleaned.drop_duplicates(inplace=True)

    raw_cleaned['Plx'] = pd.to_numeric(raw_cleaned['Plx'], errors='coerce')
    raw_cleaned['Vmag'] = pd.to_numeric(raw_cleaned['Vmag'], errors='coerce')
    raw_cleaned['B-V'] = pd.to_numeric(raw_cleaned['B-V'], errors='coerce')

    return raw_cleaned

def feature_engineering(df):
    """Performs feature engineering on cleaned data."""
    # Convert parallax to distance, calculate temperature, etc.
    df['Plx_arcsec'] = df['Plx'] * 0.001
    df['Distance_pc'] = 1 / df['Plx_arcsec']
    df['Distance_ly'] = df['Distance_pc'] * 3.26
    df['Amag'] = df['Vmag'] + 5 * (np.log10(df['Distance_pc']) - 1)
    df['Temperature_K'] = 7090 / (df['B-V'] + 0.72)
    
    return df

if __name__ == '__main__':
    df = load_and_clean_data('data/raw/Star99999_raw.csv')
    df = feature_engineering(df)
    print(df.head())  # Display first few rows to check the output

