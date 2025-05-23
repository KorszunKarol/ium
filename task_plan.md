# Task Plan: Data Sufficiency Assessment (Task ii)

This plan outlines the steps to assess whether the provided data (`listings.csv`, `calendar.csv`, `reviews.csv`) is sufficient and of adequate quality for the Nocarz London marketing campaign analysis.

- [x] **1. Assess Impact of Missing `listing_id` on Data Linkage:**
    - [x] Count total rows, non-null `listing_id` rows, and unique `listing_id`s in `listings`, `calendar`, and `reviews`.
    - [x] Perform inner merge of `listings` and `calendar` on `listing_id`.
    - [x] Report row count and unique `listing_id` count after merge.
    - [x] Summarize the percentage of listings lost due to missing IDs for the core analysis.

- [x] **2. Assess Impact of Missing `neighbourhood_cleansed`:**
    - [x] Load necessary data (likely merged from step 1 or listings directly).
    - [x] Count listings with missing `neighbourhood_cleansed`.
    - [x] Plot distribution of listings per `neighbourhood_cleansed`, including a 'Missing' category.
    - [x] (Optional) Explore imputation using lat/lon if available and complete enough.
    - [x] Summarize the potential impact on neighborhood ranking reliability.

- [x] **3. Evaluate `calendar.csv` Data Quality for Occupancy & ADR Calculation:**
    - [x] Load `calendar.csv`.
    - [x] Clean and convert `price` column; plot distribution (histogram/boxplot) to find outliers/errors.
    - [x] Validate `available` column values ('t'/'f'/NaN).
    - [x] Calculate and plot distribution of observation period length (max date - min date) per `listing_id`.
    - [x] Calculate and summarize data quality metrics (missing price/availability %, avg observation period) per neighborhood (requires merging).
    - [x] Plot calculated `occupancy_rate` vs. observation period length to check stability.
    - [x] Summarize reliability of derived metrics ($Occ_{hist}$, $ADR_{hist}$).

- [x] **4. Basic Quality Check of `listings.csv` Features:**
    - [x] Load `listings.csv`.
    - [x] Generate and visualize missing value patterns (heatmap/barchart) focusing on key predictive features (e.g., `property_type`, `room_type`, `accommodates`, `bedrooms`, `review_scores_rating`).
    - [x] Plot distributions of key numerical (`accommodates`, `review_scores_rating`) and categorical (`property_type`, `room_type`) features.
    - [x] Summarize overall quality impression of listing features.

- [x] **5. Cross-File Consistency Checks:**
    - [x] Merge necessary data (`listings`, `reviews`, potentially aggregated calendar).
    - [x] Compare `review_scores_rating` (listings) vs. calculated average `numerical_review` (reviews) using a scatter plot for matching listings.
    - [x] Compare `availability_365` (listings) vs. `1 - occupancy_rate` (calculated from calendar) if applicable/reliable.
    - [x] Summarize observed consistencies or inconsistencies.

- [x] **6. Final Summary & Recommendations:**
    - [x] Synthesize findings from steps 1-5.
    - [x] Provide a conclusion on data sufficiency for the primary goal (neighborhood ranking based on historical revenue).
    - [x] Recommend necessary cleaning steps, imputations, or data filtering.
    - [x] State whether additional data seems necessary.
