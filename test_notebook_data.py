#!/usr/bin/env python3
import pandas as pd
import numpy as np

try:
    # Load data
    processed_data_dir = "data/processed/etap2/"
    print("Loading datasets...")
    listings_df = pd.read_pickle(processed_data_dir + "listings_e2_df.pkl")
    calendar_df = pd.read_pickle(processed_data_dir + "calendar_e2_df.pkl")
    reviews_df = pd.read_pickle(processed_data_dir + "reviews_e2_df.pkl")

    print(f"Listings: {listings_df.shape}")
    print(f"Calendar: {calendar_df.shape}")
    print(f"Reviews: {reviews_df.shape}")

    print("\nListings columns (first 20):")
    print(listings_df.columns[:20].tolist())

    # Test key columns
    key_cols = [
        "price",
        "host_is_superhost",
        "bathrooms_text",
        "amenities",
        "name",
        "description",
    ]
    print("\nKey columns check:")
    for col in key_cols:
        if col in listings_df.columns:
            print(
                f"✓ {col}: {listings_df[col].dtype} ({listings_df[col].notna().sum()} non-null)"
            )
        else:
            print(f"✗ {col}: NOT FOUND")

    # Test price cleaning
    def clean_price_column(df, price_col="price"):
        df_clean = df.copy()
        if price_col in df.columns:
            df_clean["price_numeric"] = df_clean[price_col].replace(
                {r"\$|,": ""}, regex=True
            )
            df_clean["price_numeric"] = pd.to_numeric(
                df_clean["price_numeric"], errors="coerce"
            )
            return df_clean
        else:
            print(f"Warning: {price_col} column not found")
            return df_clean

    listings_processed = clean_price_column(listings_df)
    if "price_numeric" in listings_processed.columns:
        print(
            f"\nPrice processing successful: {listings_processed['price_numeric'].notna().sum()} valid prices"
        )
        print(
            f"Price range: ${listings_processed['price_numeric'].min():.2f} - ${listings_processed['price_numeric'].max():.2f}"
        )

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
