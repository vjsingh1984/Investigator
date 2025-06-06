#!/usr/bin/env python3
"""
Script to read and analyze AAPL.parquet file for proper price data
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_parquet_data(file_path):
    """Read parquet file and check for proper price data"""
    
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        print(f"Successfully loaded parquet file: {file_path}")
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nData types:")
        print(df.dtypes)
        
        # Display first few rows
        print("\n" + "="*80)
        print("First 10 rows:")
        print("="*80)
        print(df.head(10))
        
        # Display last few rows
        print("\n" + "="*80)
        print("Last 10 rows:")
        print("="*80)
        print(df.tail(10))
        
        # Check for price columns
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower() or 'open' in col.lower() or 'high' in col.lower() or 'low' in col.lower()]
        
        if price_columns:
            print("\n" + "="*80)
            print("Price column analysis:")
            print("="*80)
            
            for col in price_columns:
                print(f"\nColumn: {col}")
                print(f"  - Non-null count: {df[col].notna().sum()}")
                print(f"  - Null count: {df[col].isna().sum()}")
                print(f"  - Zero values: {(df[col] == 0).sum()}")
                print(f"  - Min value: {df[col].min()}")
                print(f"  - Max value: {df[col].max()}")
                print(f"  - Mean value: {df[col].mean():.2f}")
                print(f"  - Latest value: {df[col].iloc[-1] if len(df) > 0 else 'N/A'}")
        
        # Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            print("\n" + "="*80)
            print("Date range:")
            print("="*80)
            print(f"Start date: {df.index.min()}")
            print(f"End date: {df.index.max()}")
            print(f"Total days: {(df.index.max() - df.index.min()).days}")
        
        # Check for any column with numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\n" + "="*80)
            print("Summary statistics for numeric columns:")
            print("="*80)
            print(df[numeric_cols].describe())
            
            # Check for zero or negative values
            print("\n" + "="*80)
            print("Data quality check:")
            print("="*80)
            for col in numeric_cols:
                zero_count = (df[col] == 0).sum()
                negative_count = (df[col] < 0).sum()
                if zero_count > 0 or negative_count > 0:
                    print(f"\nWarning for column '{col}':")
                    if zero_count > 0:
                        print(f"  - Contains {zero_count} zero values")
                    if negative_count > 0:
                        print(f"  - Contains {negative_count} negative values")
        
        return df
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None

if __name__ == "__main__":
    # Path to the parquet file
    parquet_file = Path("/Users/vijaysingh/code/InvestiGator/data/price_cache/AAPL.parquet")
    
    # Check the file
    df = check_parquet_data(parquet_file)
    
    if df is not None:
        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)