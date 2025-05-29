import pandas as pd
import numpy as np

# --- Configuration ---
LISTINGS_PATH = "./listings.csv"
CALENDAR_PATH = "./calendar.csv"
REVIEWS_PATH = "./reviews.csv"
LISTINGS_ID_COL = "id"  # Correct column name for listings
CALENDAR_ID_COL = "listing_id"
REVIEWS_ID_COL = "listing_id"

# --- Load Data ---
print(f"Loading listings from: {LISTINGS_PATH}")
df_listings = pd.read_csv(LISTINGS_PATH, low_memory=False)

print(f"Loading calendar from: {CALENDAR_PATH}")
# Load first, then parse dates with error handling
df_calendar = pd.read_csv(CALENDAR_PATH)
if "date" in df_calendar.columns:
    df_calendar["date"] = pd.to_datetime(df_calendar["date"], errors="coerce")

print(f"Loading reviews from: {REVIEWS_PATH}")
# Load first, then parse dates with error handling
df_reviews = pd.read_csv(REVIEWS_PATH)
if "date" in df_reviews.columns:
    df_reviews["date"] = pd.to_datetime(df_reviews["date"], errors="coerce")

print("\n--- Initial Data Shapes ---")
print(f"Listings: {df_listings.shape}")
print(f"Calendar: {df_calendar.shape}")
print(f"Reviews: {df_reviews.shape}")

# --- Analyze listing_id Completeness ---
print("\n--- Listing ID Analysis ---")

id_stats = {}


def calculate_id_stats(df, name, id_col):
    # Ensure id_col column exists
    if id_col not in df.columns:
        print(f"Warning: '{id_col}' column not found in {name}.")
        return {
            "Total Rows": len(df),
            f"Rows with non-null {id_col}": 0,
            f"Unique {id_col}s": 0,
            f"% Non-Null {id_col}": 0,
        }

    total_rows = len(df)
    non_null_ids = df[id_col].notna().sum()
    unique_ids = df[id_col].nunique()
    stats = {
        "Total Rows": total_rows,
        f"Rows with non-null {id_col}": non_null_ids,
        f"Unique {id_col}s": unique_ids,
        f"% Non-Null {id_col}": (
            round((non_null_ids / total_rows) * 100, 2) if total_rows > 0 else 0
        ),
    }
    return stats


# Use correct ID column names
id_stats["listings"] = calculate_id_stats(df_listings, "listings", LISTINGS_ID_COL)
id_stats["calendar"] = calculate_id_stats(df_calendar, "calendar", CALENDAR_ID_COL)
id_stats["reviews"] = calculate_id_stats(df_reviews, "reviews", REVIEWS_ID_COL)

stats_df = pd.DataFrame(id_stats).T
print(stats_df)

# --- Simulate Core Merge (Listings + Calendar) ---
print("\n--- Merge Simulation (Listings INNER JOIN Calendar) ---")

# Proceed only if ID columns exist in both dataframes
if LISTINGS_ID_COL in df_listings.columns and CALENDAR_ID_COL in df_calendar.columns:
    # Convert ID columns to numeric, coercing errors to NaN, then drop NaNs for the merge simulation
    listings_ids = (
        pd.to_numeric(df_listings[LISTINGS_ID_COL], errors="coerce").dropna().unique()
    )
    calendar_ids = (
        pd.to_numeric(df_calendar[CALENDAR_ID_COL], errors="coerce").dropna().unique()
    )

    # Create temporary DFs with a common column name for merging
    df_listings_ids = pd.DataFrame(
        {CALENDAR_ID_COL: listings_ids}
    )  # Rename listings ID col to match calendar
    df_calendar_ids = pd.DataFrame({CALENDAR_ID_COL: calendar_ids})

    # Perform the inner merge on unique non-null IDs using the common column name
    merged_df = pd.merge(
        df_listings_ids,
        df_calendar_ids,
        on=CALENDAR_ID_COL,  # Merge on the common name
        how="inner",
    )

    merged_unique_ids = merged_df[CALENDAR_ID_COL].nunique()
    listings_unique_ids = len(listings_ids)
    calendar_unique_ids = len(calendar_ids)

    print(
        f"Unique valid listing_ids in listings (col '{LISTINGS_ID_COL}'): {listings_unique_ids}"
    )
    print(
        f"Unique valid listing_ids in calendar (col '{CALENDAR_ID_COL}'): {calendar_unique_ids}"
    )
    print(
        f"Unique listing_ids present in BOTH listings and calendar (after cleaning): {merged_unique_ids}"
    )

    if listings_unique_ids > 0:
        percentage_retained = round((merged_unique_ids / listings_unique_ids) * 100, 2)
        print(
            f"Percentage of unique listings retained after merge: {percentage_retained}%"
        )
    else:
        print(
            "Cannot calculate retention percentage, no valid unique listings found in listings df."
        )
else:
    print(
        f"Skipping merge simulation: ID column missing ('{LISTINGS_ID_COL}' in listings or '{CALENDAR_ID_COL}' in calendar)."
    )
