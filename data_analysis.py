'''Basic Exploratory Data Analysis (EDA) for the Nocarz dataset.

This script loads the primary datasets and performs initial inspection.
'''

import pandas as pd
import os
import numpy as np
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt

DATA_DIR = '.'

def load_listings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Loads the listings dataset.

    Args:
        data_dir: The directory containing the dataset files.

    Returns:
        A pandas DataFrame containing the listings data.

    Raises:
        FileNotFoundError: If listings.csv is not found in data_dir.
    """
    file_path = os.path.join(data_dir, 'listings.csv')
    print(f"Loading listings from: {file_path}")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: listings.csv not found at {file_path}")
        raise

def load_calendar(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Loads the calendar dataset.

    Args:
        data_dir: The directory containing the dataset files.

    Returns:
        A pandas DataFrame containing the calendar data.

    Raises:
        FileNotFoundError: If calendar.csv is not found in data_dir.
    """
    file_path = os.path.join(data_dir, 'calendar.csv')
    print(f"Loading calendar from: {file_path}")
    try:
        return pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: calendar.csv not found at {file_path}")
        raise

def load_reviews(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Loads the reviews dataset.

    Args:
        data_dir: The directory containing the dataset files.

    Returns:
        A pandas DataFrame containing the reviews data.

    Raises:
        FileNotFoundError: If reviews.csv is not found in data_dir.
    """
    file_path = os.path.join(data_dir, 'reviews.csv')
    print(f"Loading reviews from: {file_path}")
    try:
        return pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: reviews.csv not found at {file_path}")
        raise

def clean_price_column(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Cleans a price column by removing currency symbols and converting to float.

    Args:
        df: DataFrame containing the price column.
        price_col: Name of the price column to clean.

    Returns:
        DataFrame with cleaned price column.
    """
    if price_col not in df.columns:
        return df

    df_copy = df.copy()

    if df_copy[price_col].dtype == 'object':
        df_copy[price_col] = df_copy[price_col].astype(str)
        df_copy[price_col] = df_copy[price_col].str.replace('$', '')
        df_copy[price_col] = df_copy[price_col].str.replace(',', '')
        df_copy[price_col] = pd.to_numeric(df_copy[price_col], errors='coerce')

    return df_copy


def process_calendar_data(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans price and converts availability in the calendar DataFrame.

    Args:
        calendar_df: The raw calendar DataFrame.

    Returns:
        Processed calendar DataFrame with cleaned price and numeric availability.
    """
    processed_df = clean_price_column(calendar_df, 'price')

    processed_df['available_numeric'] = processed_df['available'].map({'t': 1, 'f': 0})

    return processed_df

def calculate_occupancy(processed_calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the occupancy rate per listing.

    Assumes 'available_numeric' column exists where 0 means booked.

    Args:
        processed_calendar_df: Calendar DataFrame after processing.

    Returns:
        DataFrame with listing_id and calculated occupancy_rate.
    """
    if 'available_numeric' not in processed_calendar_df.columns:
        print("Error: 'available_numeric' column required for occupancy calculation.")
        return None
    if 'date' not in processed_calendar_df.columns or not pd.api.types.is_datetime64_any_dtype(processed_calendar_df['date']):
         print("Error: 'date' column (datetime type) required for occupancy calculation.")
         return None

    occupancy = processed_calendar_df.groupby('listing_id').agg(
        total_days=('available_numeric', 'count'),
        booked_days=('available_numeric', lambda x: (x == 0).sum())
    ).reset_index()

    occupancy['occupancy_rate'] = occupancy.apply(
        lambda row: row['booked_days'] / row['total_days'] if row['total_days'] > 0 else 0,
        axis=1
    )
    return occupancy[['listing_id', 'occupancy_rate']]

def calculate_adr(processed_calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Average Daily Rate (ADR) per listing for available days.

    Assumes 'available_numeric' and 'price' columns exist.

    Args:
        processed_calendar_df: Calendar DataFrame after processing.

    Returns:
        DataFrame with listing_id and calculated adr. Returns None if columns missing.
    """
    if 'available_numeric' not in processed_calendar_df.columns:
        print("Error: 'available_numeric' column required for ADR calculation.")
        return None
    if 'price' not in processed_calendar_df.columns:
        print("Error: 'price' column required for ADR calculation.")
        return None

    # Filter for available days and calculate average price
    available_days = processed_calendar_df[processed_calendar_df['available_numeric'] == 1].copy()

    # Check if there are any available days to avoid errors on empty groups
    if available_days.empty:
        print("Warning: No available days found in calendar data to calculate ADR.")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['listing_id', 'adr'])

    adr = available_days.groupby('listing_id')['price'].mean().reset_index()
    adr.rename(columns={'price': 'adr'}, inplace=True)

    # Round ADR to 2 decimal places
    adr['adr'] = adr['adr'].round(2)

    return adr[['listing_id', 'adr']]

def visualize_neighbourhood_prices(listings_df: pd.DataFrame) -> folium.Map:
    """Creates an interactive map showing average prices by neighborhood.

    Args:
        listings_df: DataFrame containing listings data with price and neighborhood information.

    Returns:
        Folium Map object with neighborhood price visualization.
    """
    cleaned_df = clean_price_column(listings_df, 'price')

    if 'neighbourhood_cleansed' not in cleaned_df.columns:
        print("Error: 'neighbourhood_cleansed' column not found in listings data")
        return None

    neighbourhood_prices = cleaned_df.groupby('neighbourhood_cleansed')['price'].agg(['mean', 'count']).reset_index()
    neighbourhood_prices.columns = ['neighbourhood', 'avg_price', 'listing_count']
    neighbourhood_prices = neighbourhood_prices[neighbourhood_prices['listing_count'] > 5]
    neighbourhood_prices['avg_price'] = neighbourhood_prices['avg_price'].round(2)

    print("\n--- Neighborhood Price Analysis ---")
    print(neighbourhood_prices.sort_values('avg_price', ascending=False).head(10))

    if 'latitude' in cleaned_df.columns and 'longitude' in cleaned_df.columns:
        center_lat = cleaned_df['latitude'].mean()
        center_lon = cleaned_df['longitude'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        marker_cluster = MarkerCluster().add_to(m)

        for idx, row in neighbourhood_prices.iterrows():
            neighborhood = row['neighbourhood']
            avg_price = row['avg_price']
            listing_count = row['listing_count']

            neighborhood_listings = cleaned_df[cleaned_df['neighbourhood_cleansed'] == neighborhood]
            if len(neighborhood_listings) > 0:
                neighborhood_center = [
                    neighborhood_listings['latitude'].mean(),
                    neighborhood_listings['longitude'].mean()
                ]

                popup_text = f"<b>{neighborhood}</b><br>Average Price: £{avg_price:.2f}<br>Listings: {listing_count}"

                folium.Marker(
                    location=neighborhood_center,
                    popup=popup_text,
                    icon=folium.Icon(color='green' if avg_price > neighbourhood_prices['avg_price'].median() else 'blue')
                ).add_to(marker_cluster)

        for idx, row in cleaned_df.sample(min(500, len(cleaned_df))).iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(row['price']):
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"Price: £{row['price']:.2f}",
                    fill=True,
                    fill_opacity=0.7,
                    color='red',
                    fill_color='red'
                ).add_to(m)

        map_file = 'london_neighborhood_prices.html'
        m.save(map_file)
        print(f"\nInteractive map saved to {map_file}")
        return m
    else:
        print("Error: Latitude or longitude columns not found")
        return None

def main() -> None:
    """Main function to load data and perform initial analysis."""
    print("--- Loading Data ---")
    try:
        listings_df = load_listings()
        calendar_df = load_calendar()
        reviews_df = load_reviews()
    except FileNotFoundError:
        print("Halting execution due to missing file(s). Please check DATA_DIR.")
        return

    print("\n--- Initial Listings Analysis ---")
    print("Listings Head:")
    print(listings_df.head())
    print("\nListings Info:")
    listings_df.info()
    print("\nListings Description (Numerical):")
    print(listings_df.describe())
    print("\nListings Description (Object):")
    print(listings_df.describe(include='object'))

    print("\n--- Initial Calendar Analysis ---")
    print("Calendar Head:")
    print(calendar_df.head())
    print("\nCalendar Info:")
    calendar_df.info()
    print("\nCalendar Description:")
    print(calendar_df.describe(include='all'))

    print("\n--- Processing Calendar Data ---")
    processed_calendar_df = process_calendar_data(calendar_df)
    if processed_calendar_df is not None:
        print("Calendar Data Processed (cleaned price, added available_numeric).")
        print("Processed Calendar Head:")
        print(processed_calendar_df.head())
        # Basic Stats from Processed Calendar
        min_date = processed_calendar_df['date'].min().strftime('%Y-%m-%d')
        max_date = processed_calendar_df['date'].max().strftime('%Y-%m-%d')
        avg_price_calendar = processed_calendar_df['price'].mean()
        availability_rate = processed_calendar_df['available_numeric'].mean()
        print(f"\nCalendar Date Range: {min_date} to {max_date}")
        print(f"Average Price (when listed): £{avg_price_calendar:.2f}")
        print(f"Overall Availability Rate: {availability_rate:.2%}") # Rate of 't'

        print("\n--- Calculating Occupancy ---")
        occupancy_df = calculate_occupancy(processed_calendar_df)
        if occupancy_df is not None:
            print("Occupancy Calculated.")
            print("Occupancy Stats:")
            print(occupancy_df['occupancy_rate'].describe())
            # Merge occupancy back into listings_df
            # Ensure listing_id types match before merging if necessary
            # listings_df['id'] might need cleaning/conversion if not float/int
            try:
                 # Attempt conversion if 'id' exists and isn't numeric
                if 'id' in listings_df.columns and not pd.api.types.is_numeric_dtype(listings_df['id']):
                     listings_df['id'] = pd.to_numeric(listings_df['id'], errors='coerce')

                # Ensure both are numeric and handle potential NaNs from coercion
                listings_df = listings_df.dropna(subset=['id'])
                occupancy_df = occupancy_df.dropna(subset=['listing_id'])

                listings_df['id'] = listings_df['id'].astype(np.int64) # Or appropriate int type
                occupancy_df['listing_id'] = occupancy_df['listing_id'].astype(np.int64)

                listings_df = pd.merge(listings_df, occupancy_df, left_on='id', right_on='listing_id', how='left')
                print("\nListings Info (after merging occupancy):")
                listings_df.info(verbose=False) # Use verbose=False for brevity
            except Exception as e:
                print(f"Error merging occupancy: {e}")
                print("Skipping occupancy merge.")
        else:
            print("Skipping occupancy-related analysis and merge.")

        print("\n--- Calculating ADR ---")
        adr_df = calculate_adr(processed_calendar_df)
        if adr_df is not None:
            print("ADR Calculated.")
            print("ADR Stats:")
            print(adr_df['adr'].describe())
            # Merge ADR back into listings_df
            try:
                # Ensure listing_id types match (already handled above if occupancy merge succeeded)
                 if 'id' in listings_df.columns and pd.api.types.is_numeric_dtype(listings_df['id']): # Check if ID is usable
                    adr_df = adr_df.dropna(subset=['listing_id'])
                    adr_df['listing_id'] = adr_df['listing_id'].astype(listings_df['id'].dtype) # Match type
                    listings_df = pd.merge(listings_df, adr_df, left_on='id', right_on='listing_id', how='left', suffixes=('', '_adr'))
                    # Drop the redundant listing_id column from the merge
                    if 'listing_id_adr' in listings_df.columns:
                         listings_df = listings_df.drop(columns=['listing_id_adr'])
                    if 'listing_id' in listings_df.columns and 'id' in listings_df.columns and 'listing_id' != 'id':
                        # If occupancy merge failed but ADR merge succeeded, listing_id might be present
                        listings_df = listings_df.drop(columns=['listing_id'])

                    print("\nListings Info (after merging ADR):")
                    listings_df.info(verbose=False)
                 else:
                     print("Skipping ADR merge as listings 'id' column is problematic.")

            except Exception as e:
                print(f"Error merging ADR: {e}")
                print("Skipping ADR merge.")

        else:
             print("Skipping ADR-related analysis and merge.")

    else:
        print("Skipping calendar processing, occupancy, and ADR analysis.")

    print("\n--- Initial Reviews Analysis ---")
    print("Reviews Head:")
    print(reviews_df.head())
    print("\nReviews Info:")
    reviews_df.info()
    print("\nReviews Description:")
    print(reviews_df.describe(include='all'))

    print("\n--- Creating Neighborhood Price Visualization ---")
    visualize_neighbourhood_prices(listings_df)

if __name__ == "__main__":
    main()