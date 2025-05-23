# Project Context: Nocarz Marketing Campaign Analysis

## Business Problem

The Nocarz board, prompted by the London branch, needs to identify locations within London that have the potential for the highest profit. This information is required to target a marketing campaign aimed at encouraging property owners in those specific high-potential areas to list their properties on the Nocarz portal.

## Goal

Analyze the provided data (originally from a SharePoint link, now available as CSV files: `listings.csv`, `calendar.csv`, `reviews.csv`, `sessions.csv`, `users.csv`) to determine which London locations are likely to generate the most revenue or profit. The primary focus should be on data related to property listings (`listings.csv`).

## Data

The data was provided via a SharePoint link and subsequently extracted from a `KK-KK.tar` archive into several CSV files. The most relevant files for identifying profitable locations are:

*   **`listings.csv`**: Contains core information about each listing.
    *   Key Columns: `id` (listing_id), `price` (nightly price, needs cleaning), `neighbourhood_cleansed` (target grouping column), `latitude`, `longitude`, `property_type`, `room_type`, `accommodates`, `availability_365`, `number_of_reviews`, `review_scores_rating`.
*   **`calendar.csv`**: Contains daily availability and pricing data for listings. Crucial for calculating occupancy and revenue.
    *   Key Columns: `listing_id` (links to `listings.csv`), `date`, `available` (t/f for booked status), `price`/`adjusted_price` (daily price, needs cleaning).
*   **`reviews.csv`**: Contains details for individual reviews. Useful for calculating booking frequency.
    *   Key Columns: `listing_id` (links to `listings.csv`), `date`, `reviewer_id`, `numerical_review`.
*   `sessions.csv`: Likely contains user website interaction data. Deprioritized for this analysis.
*   `users.csv`: Likely contains user profile data. Deprioritized for this analysis.

## Next Steps (Refined Analysis Plan)

The analysis requires loading and processing the key CSV files, likely using a data analysis environment (e.g., Python with Pandas).

1.  **Load Data**: Load `listings.csv`, `calendar.csv`, and potentially `reviews.csv`.
2.  **Clean Data**:
    *   Convert price columns (in `listings.csv` and `calendar.csv`) to numeric types.
    *   Parse date columns.
    *   Handle missing values appropriately.
3.  **Process `calendar.csv`**:
    *   Aggregate by `listing_id`.
    *   Calculate the occupancy rate (count of days where `available`='t' / total days listed in the calendar).
    *   Calculate the average daily price when available.
4.  **Process `reviews.csv` (Optional but Recommended)**:
    *   Aggregate by `listing_id`.
    *   Calculate review velocity (e.g., number of reviews per month/year) as a proxy for booking frequency.
5.  **Merge Data**:
    *   Merge the aggregated occupancy/price data from `calendar.csv` into `listings.csv` using `listing_id`.
    *   Merge the aggregated review velocity data (if calculated) from `reviews.csv` into `listings.csv` using `listing_id`.
6.  **Aggregate by Neighbourhood**: Group the merged dataframe by `neighbourhood_cleansed`.
7.  **Calculate Neighbourhood Metrics**: For each neighbourhood, calculate metrics such as:
    *   Average/Median nightly price.
    *   Average/Median occupancy rate.
    *   Average/Median estimated annual revenue per listing (e.g., Avg Daily Price * Avg Occupancy Rate * 365).
    *   Average/Median review velocity (if calculated).
    *   Number of listings.
    *   Average review score (`review_scores_rating`).
8.  **Identify Top Locations**: Rank neighbourhoods based on a combination of these metrics (especially estimated revenue, occupancy rate, and potentially listing volume) to identify the most promising locations for the marketing campaign.

--- Loading Data ---
Loading listings from: ./listings.csv
Loading calendar from: ./calendar.csv
Loading reviews from: ./reviews.csv

--- Initial Listings Analysis ---
Listings Head:
             id                                      listing_url     scrape_id  ... calculated_host_listings_count_private_rooms calculated_host_listings_count_shared_rooms reviews_per_month
0  1.897406e+07             https://www.nocarz.pl/rooms/18974065  2.024121e+13  ...                                          0.0                                         NaN               NaN
1  1.074536e+18  https://www.nocarz.pl/rooms/1074535877676155362           NaN  ...                                          2.0                                         NaN              0.25
2  7.963047e+17   https://www.nocarz.pl/rooms/796304692597746009  2.024121e+13  ...                                          NaN                                         0.0               NaN
3  3.694235e+07             https://www.nocarz.pl/rooms/36942349           NaN  ...                                          0.0                                         0.0               NaN
4           NaN   https://www.nocarz.pl/rooms/629991867534346504  2.024121e+13  ...                                          0.0                                         0.0              0.40

[5 rows x 75 columns]

Listings Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 28543 entries, 0 to 28542
Data columns (total 75 columns):
 #   Column                                        Non-Null Count  Dtype
---  ------                                        --------------  -----
 0   id                                            19994 non-null  float64
 1   listing_url                                   19965 non-null  object
 2   scrape_id                                     20092 non-null  float64
 3   last_scraped                                  19976 non-null  object
 4   source                                        19876 non-null  object
 5   name                                          20019 non-null  object
 6   description                                   19180 non-null  object
 7   neighborhood_overview                         9194 non-null   object
 8   picture_url                                   19912 non-null  object
 9   host_id                                       19971 non-null  float64
 10  host_url                                      19981 non-null  object
 11  host_name                                     19944 non-null  object
 12  host_since                                    19911 non-null  object
 13  host_location                                 15314 non-null  object
 14  host_about                                    10157 non-null  object
 15  host_response_time                            13107 non-null  object
 16  host_response_rate                            13057 non-null  object
 17  host_acceptance_rate                          14236 non-null  object
 18  host_is_superhost                             19724 non-null  object
 19  host_thumbnail_url                            20002 non-null  object
 20  host_picture_url                              20026 non-null  object
 21  host_neighbourhood                            9768 non-null   object
 22  host_listings_count                           20019 non-null  float64
 23  host_total_listings_count                     19967 non-null  float64
 24  host_verifications                            19929 non-null  object
 25  host_has_profile_pic                          19952 non-null  object
 26  host_identity_verified                        19984 non-null  object
 27  neighbourhood                                 9153 non-null   object
 28  neighbourhood_cleansed                        19899 non-null  object
 29  neighbourhood_group_cleansed                  0 non-null      float64
 30  latitude                                      20008 non-null  float64
 31  longitude                                     20003 non-null  float64
 32  property_type                                 19929 non-null  object
 33  room_type                                     19954 non-null  object
 34  accommodates                                  19953 non-null  float64
 35  bathrooms                                     13186 non-null  float64
 36  bathrooms_text                                19957 non-null  object
 37  bedrooms                                      17240 non-null  float64
 38  beds                                          13092 non-null  float64
 39  amenities                                     20078 non-null  object
 40  price                                         13127 non-null  object
 41  minimum_nights                                20018 non-null  float64
 42  maximum_nights                                20077 non-null  float64
 43  minimum_minimum_nights                        20107 non-null  float64
 44  maximum_minimum_nights                        19816 non-null  float64
 45  minimum_maximum_nights                        19967 non-null  float64
 46  maximum_maximum_nights                        19847 non-null  float64
 47  minimum_nights_avg_ntm                        20142 non-null  float64
 48  maximum_nights_avg_ntm                        20101 non-null  float64
 49  calendar_updated                              0 non-null      float64
 50  has_availability                              18853 non-null  object
 51  availability_30                               20000 non-null  float64
 52  availability_60                               19951 non-null  float64
 53  availability_90                               19972 non-null  float64
 54  availability_365                              19925 non-null  float64
 55  calendar_last_scraped                         19980 non-null  object
 56  number_of_reviews                             19972 non-null  float64
 57  number_of_reviews_ltm                         19997 non-null  float64
 58  number_of_reviews_l30d                        19979 non-null  float64
 59  first_review                                  14782 non-null  object
 60  last_review                                   14907 non-null  object
 61  review_scores_rating                          14792 non-null  float64
 62  review_scores_accuracy                        14725 non-null  float64
 63  review_scores_cleanliness                     14725 non-null  float64
 64  review_scores_checkin                         14719 non-null  float64
 65  review_scores_communication                   14815 non-null  float64
 66  review_scores_location                        14840 non-null  float64
 67  review_scores_value                           14667 non-null  float64
 68  license                                       0 non-null      float64
 69  instant_bookable                              19867 non-null  object
 70  calculated_host_listings_count                20026 non-null  float64
 71  calculated_host_listings_count_entire_homes   20005 non-null  float64
 72  calculated_host_listings_count_private_rooms  20124 non-null  float64
 73  calculated_host_listings_count_shared_rooms   20031 non-null  float64
 74  reviews_per_month                             14811 non-null  float64
dtypes: float64(41), object(34)
memory usage: 16.3+ MB

Listings Description (Numerical):
                 id     scrape_id       host_id  ...  calculated_host_listings_count_private_rooms  calculated_host_listings_count_shared_rooms  reviews_per_month
count  1.999400e+04  2.009200e+04  1.997100e+04  ...                                  20124.000000                                 20031.000000       14811.000000
mean   5.750160e+17  2.024121e+13  1.925223e+08  ...                                      1.941513                                     0.022715           1.019395
std    5.314921e+17  3.906347e-03  1.997370e+08  ...                                      7.658135                                     0.382118           1.339458
min    2.432800e+04  2.024121e+13  2.594000e+03  ...                                      0.000000                                     0.000000           0.010000
25%    2.709120e+07  2.024121e+13  2.541621e+07  ...                                      0.000000                                     0.000000           0.160000
50%    7.137340e+17  2.024121e+13  1.018764e+08  ...                                      0.000000                                     0.000000           0.530000
75%    1.097723e+18  2.024121e+13  3.475728e+08  ...                                      1.000000                                     0.000000           1.310000
max    1.308834e+18  2.024121e+13  6.662365e+08  ...                                    125.000000                                    13.000000          23.010000

[8 rows x 41 columns]

Listings Description (Object):
                                 listing_url last_scraped       source         name  ... calendar_last_scraped first_review last_review instant_bookable
count                                  19965        19976        19876        20019  ...                 19980        14782       14907            19867
unique                                 19965            3            2        19735  ...                     4         3368        2260                2
top     https://www.nocarz.pl/rooms/18974065   2024-12-12  city scrape  Double Room  ...            2024-12-12   2024-12-01  2024-12-08                f
freq                                       1        10839        13107            9  ...                 10849           47         482            13785

[4 rows x 34 columns]

--- Initial Calendar Analysis ---
Calendar Head:
     listing_id       date available    price adjusted_price  minimum_nights  maximum_nights
0  1.990216e+07 2025-05-02       NaN   $37.00            NaN             9.0          1125.0
1  3.707577e+07 2025-11-25         f   $72.00            NaN             NaN          1125.0
2  1.091268e+18        NaT       NaN  $311.00            NaN             4.0             NaN
3  8.296219e+17 2025-12-03         t   $36.00            NaN             5.0           365.0
4           NaN 2025-09-16         f   $86.00            NaN             1.0            30.0

Calendar Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10417861 entries, 0 to 10417860
Data columns (total 7 columns):
 #   Column          Dtype
---  ------          -----
 0   listing_id      float64
 1   date            datetime64[ns]
 2   available       object
 3   price           object
 4   adjusted_price  object
 5   minimum_nights  float64
 6   maximum_nights  float64
dtypes: datetime64[ns](1), float64(3), object(3)
memory usage: 556.4+ MB

Calendar Description:
          listing_id                           date available    price adjusted_price  minimum_nights  maximum_nights
count   7.291415e+06                        7294110   7293170  7292814           1826    7.293310e+06    7.290930e+06
unique           NaN                            NaN         2     1510            376             NaN             NaN
top              NaN                            NaN         f  $100.00     $10,000.00             NaN             NaN
freq             NaN                            NaN   4575951   249993            162             NaN             NaN
mean    5.773759e+17  2025-06-11 18:37:17.961970944       NaN      NaN            NaN    6.489314e+00    5.352821e+05
min     1.391300e+04            2024-12-11 00:00:00       NaN      NaN            NaN    1.000000e+00    1.000000e+00
25%     2.752456e+07            2025-03-13 00:00:00       NaN      NaN            NaN    1.000000e+00    9.000000e+01
50%     7.163810e+17            2025-06-12 00:00:00       NaN      NaN            NaN    2.000000e+00    3.650000e+02
75%     1.099444e+18            2025-09-11 00:00:00       NaN      NaN            NaN    4.000000e+00    1.125000e+03
max     1.308834e+18            2025-12-21 00:00:00       NaN      NaN            NaN    1.125000e+03    2.147484e+09
std     5.304222e+17                            NaN       NaN      NaN            NaN    2.505976e+01    3.353414e+07

--- Processing Calendar Data ---
Calendar Data Processed (cleaned price, added available_numeric).
Processed Calendar Head:
     listing_id       date available  price adjusted_price  minimum_nights  maximum_nights  available_numeric
0  1.990216e+07 2025-05-02       NaN   37.0            NaN             9.0          1125.0                NaN
1  3.707577e+07 2025-11-25         f   72.0            NaN             NaN          1125.0                0.0
2  1.091268e+18        NaT       NaN  311.0            NaN             4.0             NaN                NaN
3  8.296219e+17 2025-12-03         t   36.0            NaN             5.0           365.0                1.0
4           NaN 2025-09-16         f   86.0            NaN             1.0            30.0                0.0

Calendar Date Range: 2024-12-11 to 2025-12-21
Average Price (when listed): £239.61
Overall Availability Rate: 37.26%

--- Calculating Occupancy ---
Occupancy Calculated.
Occupancy Stats:
count    95142.000000
mean         0.627459
std          0.378994
min          0.000000
25%          0.258065
50%          0.760870
75%          1.000000
max          1.000000
Name: occupancy_rate, dtype: float64

--- Initial Reviews Analysis ---
Reviews Head:
     listing_id            id       date  reviewer_id reviewer_name                                           comments  numerical_review
0  7.276367e+06  9.056255e+17 2023-06-03   22414746.0           NaN  L'appartamento di Molly si trova a due minuti ...               4.0
1  1.733173e+07  4.830042e+08        NaT          NaN           NaN  Awesome stylish basement apartment. Close to b...               5.0
2  1.717541e+07           NaN        NaT  124877747.0           NaN                                                NaN               NaN
3  6.403758e+17  7.955223e+17        NaT  242127230.0       Maureen  Oscar's flat was wonderful in an exceptional l...               5.0
4           NaN  6.469419e+17 2022-06-11          NaN           NaN  amazing location (very easy to get places) and...               5.0

Reviews Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 572190 entries, 0 to 572189
Data columns (total 7 columns):
 #   Column            Non-Null Count   Dtype
---  ------            --------------   -----
 0   listing_id        400384 non-null  float64
 1   id                400847 non-null  float64
 2   date              400491 non-null  datetime64[ns]
 3   reviewer_id       400655 non-null  float64
 4   reviewer_name     400029 non-null  object
 5   comments          400151 non-null  object
 6   numerical_review  400197 non-null  float64
dtypes: datetime64[ns](1), float64(4), object(2)
memory usage: 30.6+ MB

Reviews Description:
          listing_id            id                           date   reviewer_id reviewer_name comments  numerical_review
count   4.003840e+05  4.008470e+05                         400491  4.006550e+05        400029   400151     400197.000000
unique           NaN           NaN                            NaN           NaN         64390   389769               NaN
top              NaN           NaN                            NaN           NaN         David        .               NaN
freq             NaN           NaN                            NaN           NaN          3101      562               NaN
mean    2.677126e+17  6.355871e+17  2021-10-21 04:22:41.396885504  1.795743e+08           NaN      NaN          4.262978
min     1.391300e+04  7.497300e+04            2009-12-21 00:00:00  8.200000e+01           NaN      NaN          0.000000
25%     1.438755e+07  4.774833e+08            2019-06-30 00:00:00  3.726751e+07           NaN      NaN          4.000000
50%     3.222546e+07  7.679472e+17            2022-11-26 00:00:00  1.169931e+08           NaN      NaN          5.000000
75%     6.710436e+17  1.094134e+18            2024-02-18 00:00:00  2.804340e+08           NaN      NaN          5.000000
max     1.307375e+18  1.310182e+18            2024-12-12 00:00:00  6.663612e+08           NaN      NaN          5.000000
std     4.184376e+17  4.973098e+17                            NaN  1.717695e+08           NaN      NaN          1.195427

--- Creating Neighborhood Price Visualization ---

--- Neighborhood Price Analysis ---
             neighbourhood  avg_price  listing_count
21                 Lambeth     344.69            474
19  Kensington and Chelsea     323.63            665
32             Westminster     321.07           1170
5                   Camden     270.01            586
6           City of London     246.49             67
18               Islington     235.17            421
31              Wandsworth     215.64            498
12  Hammersmith and Fulham     194.76            389
29           Tower Hamlets     189.87            635
26    Richmond upon Thames     186.37            102

Interactive map saved to london_neighborhood_prices.html
karolito@Karolito:~/IUM$