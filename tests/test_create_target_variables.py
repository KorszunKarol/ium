import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add the 'scripts' directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.create_target_variables import TargetVariableGenerator


class TestTargetVariableGenerator(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.default_config = {
            "minimum_observation_days": 30,
            "minimum_bookings": 3,
            "confidence_thresholds": {"high": 180, "medium": 90, "low": 30},
            "outlier_treatment": {
                "revenue_percentile_cap": 99,
                "adr_percentile_cap": 99,
                "min_reasonable_adr": 10,
                "max_reasonable_adr": 2000,
            },
            "seasonality": {
                "use_monthly_adjustment": True,
                "use_weekday_adjustment": True,
                "market_location": "london",
            },
        }
        self.generator = TargetVariableGenerator(config_path=None)
        self.generator.config = self.default_config # Ensure tests use this specific config

        # Create a simple listings_df for use in multiple tests
        self.listings_data = {'id': [1, 2, 3], 'minimum_nights': [1, 1, 35]}
        self.listings_df = pd.DataFrame(self.listings_data)

    def _create_sample_calendar_data(self, listing_id, start_date_str, num_days, price, bookings_pattern):
        """Helper to create calendar data for a listing."""
        dates = [datetime.strptime(start_date_str, '%Y-%m-%d') + timedelta(days=i) for i in range(num_days)]
        available = [(i % bookings_pattern != 0) for i in range(num_days)] # True if not booked
        prices = [price if not avail else price + 10 for i, avail in enumerate(available)] # Slightly different price if available vs booked
        
        return pd.DataFrame({
            'listing_id': listing_id,
            'date': dates,
            'available': available,
            'price_cleaned': prices
        })

    def test_basic_target_generation(self):
        """Test a basic run of the generate_targets method."""
        # Listing 1: Meets criteria
        calendar_df1 = self._create_sample_calendar_data(listing_id=1, start_date_str='2023-01-01', num_days=60, price=100, bookings_pattern=3)
        # Listing 2: Too few observation days
        calendar_df2 = self._create_sample_calendar_data(listing_id=2, start_date_str='2023-01-01', num_days=10, price=50, bookings_pattern=2)
        # Listing 3: Extreme minimum nights (filtered by _filter_with_listings_context)
        calendar_df3 = self._create_sample_calendar_data(listing_id=3, start_date_str='2023-01-01', num_days=60, price=200, bookings_pattern=4)
        
        calendar_df = pd.concat([calendar_df1, calendar_df2, calendar_df3], ignore_index=True)
        
        target_df = self.generator.generate_targets(calendar_df, self.listings_df)

        self.assertIsInstance(target_df, pd.DataFrame)
        self.assertIn('annual_revenue_adj', target_df.columns)
        self.assertIn('occupancy_rate_adj', target_df.columns)
        self.assertIn('adr_adj', target_df.columns)
        self.assertIn('confidence_score', target_df.columns)

        # Listing 1 should be present and have a reasonable confidence
        self.assertTrue(1 in target_df['listing_id'].values)
        listing1_confidence = target_df.loc[target_df['listing_id'] == 1, 'confidence_score'].iloc[0]
        self.assertTrue(listing1_confidence > 30) # Expect at least low confidence

        # Listing 2 should be filtered out due to min_observation_days or have very low confidence / be removed by final validation
        # _final_validation removes listings with obs_days < min_obs_days (30)
        self.assertFalse(2 in target_df['listing_id'].values) 
        
        # Listing 3 should be filtered out by _filter_with_listings_context due to minimum_nights > 30
        self.assertFalse(3 in target_df['listing_id'].values)


    def test_seasonality_adjustment_logic(self):
        """Test the _apply_seasonality_adjustments method's effect."""
        # Test with a listing observed only in peak summer (July)
        # Default London seasonality for July is 1.30 (monthly)
        # Weekdays will average out. Let's assume an average weekday factor around 1.0 for simplicity of manual calc here.
        # Combined factor expected to be > 1.0 ( (1.30 + ~1.0)/2 )
        
        listing_id = 1
        num_days = 31 # Full month of July
        july_start_date = '2023-07-01'
        
        # Create calendar data for July, all booked
        dates = [datetime.strptime(july_start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(num_days)]
        calendar_entries = []
        for i, date_obj in enumerate(dates):
            calendar_entries.append({
                'listing_id': listing_id,
                'date': date_obj,
                'available': False, # Booked
                'price_cleaned': 100,
                'is_booked': True,
                'month': date_obj.month,
                'weekday': date_obj.weekday(),
                'revenue': 100
            })
        calendar_df_listing = pd.DataFrame(calendar_entries)

        # Create performance_df input for _apply_seasonality_adjustments
        performance_data = {
            'listing_id': [listing_id],
            'raw_adr': [100.0],
            'raw_occupancy_rate': [1.0], # Fully booked
            'raw_daily_revenue': [100.0]
        }
        performance_df = pd.DataFrame(performance_data)

        adjusted_df = self.generator._apply_seasonality_adjustments(performance_df, calendar_df_listing)

        self.assertIn('seasonally_adj_adr', adjusted_df.columns)
        self.assertIn('seasonally_adj_occupancy', adjusted_df.columns)

        raw_adr = adjusted_df['raw_adr'].iloc[0]
        adj_adr = adjusted_df['seasonally_adj_adr'].iloc[0]
        raw_occ = adjusted_df['raw_occupancy_rate'].iloc[0]
        adj_occ = adjusted_df['seasonally_adj_occupancy'].iloc[0]
        
        # July is high season (monthly factor 1.30). Weekday factors vary.
        # Let's get the exact factors from the profile
        july_factor = self.generator.seasonality_profile['monthly'][7] # July is 7
        
        # Calculate expected average weekday factor for July 2023
        weekday_factors_july = [self.generator.seasonality_profile['weekday'][d.weekday()] for d in dates]
        avg_weekday_factor_july = np.mean(weekday_factors_july) # Simple mean as all days are booked with same weight for this test setup

        expected_monthly_factor = july_factor
        expected_weekday_factor = avg_weekday_factor_july
        expected_combined_factor = (expected_monthly_factor + expected_weekday_factor) / 2
        
        # Check the factors stored by the method
        self.assertAlmostEqual(adjusted_df['monthly_seasonality_factor'].iloc[0], expected_monthly_factor, places=2)
        self.assertAlmostEqual(adjusted_df['weekday_seasonality_factor'].iloc[0], expected_weekday_factor, places=2)
        self.assertAlmostEqual(adjusted_df['combined_seasonality_factor'].iloc[0], expected_combined_factor, places=2)

        # ADR is adjusted by combined factor, Occupancy by monthly factor
        self.assertAlmostEqual(adj_adr, raw_adr / expected_combined_factor, places=2)
        self.assertAlmostEqual(adj_occ, raw_occ / expected_monthly_factor, places=2)

        # Since July is high season (factor > 1), adjusted values should be lower than raw
        self.assertLess(adj_adr, raw_adr)
        self.assertLess(adj_occ, raw_occ)


    def test_confidence_score_calculation(self):
        """Test the _calculate_confidence_scores method."""
        target_data = {
            'listing_id': [1, 2, 3, 4],
            'observation_days': [200, 100, 45, 20],
            'total_bookings': [30, 15, 5, 1],
            'adr_adj': [100, 150, 80, 50], # All reasonable
            'occupancy_rate_adj': [0.8, 0.6, 0.5, 0.3], # All reasonable
            'annualization_factor': [365.25/200, 365.25/100, 365.25/45, 365.25/20] # ~1.8, ~3.6, ~8.1, ~18.2
        }
        target_df = pd.DataFrame(target_data)

        # Manually calculate expected scores for listing 1 (high confidence)
        # obs_days=200 >= high_thresh(180) -> obs_score = 40
        # total_bookings=30 >= 20 -> booking_score = 30
        # consistency: adr, occ are reasonable. annualization_factor ~1.8 (<6) -> consistency_score = 20
        # seasonality_coverage: obs_days=200 -> 200/30 = 6.66 -> seasonality_score = min(10, 6.66) = 6.66 (approx 7 if points are int)
        # The code has: min(10, obs_days / 30) which will be float.
        # Total = 40 + 30 + 20 + min(10, 200/30) = 90 + 6.66... = 96.66... -> high
        expected_score_l1 = 40 + 30 + 20 + min(10, 200/30)
        
        # Manually calculate for listing 4 (very low confidence)
        # obs_days=20 < low_thresh(30) -> obs_score = 20 * 20/30 = 13.33
        # total_bookings=1 < min_bookings(3) -> booking_score = 15 * 1/3 = 5
        # consistency: adr, occ reasonable. annualization_factor ~18.2 (>6) -> consistency_score = 20 - 5 = 15
        # seasonality_coverage: obs_days=20 -> 20/30 = 0.66 -> seasonality_score = min(10, 0.66) = 0.66
        # Total = 13.33 + 5 + 15 + 0.66 = 33.99 -> very_low
        expected_score_l4 = (20 * 20/self.default_config['confidence_thresholds']['low']) + \
                              (15 * 1/self.default_config['minimum_bookings']) + \
                              (20 - 5) + \
                              min(10, 20/30)

        result_df = self.generator._calculate_confidence_scores(target_df)

        self.assertAlmostEqual(result_df.loc[result_df['listing_id'] == 1, 'confidence_score'].iloc[0], expected_score_l1, places=1)
        self.assertEqual(result_df.loc[result_df['listing_id'] == 1, 'confidence_level'].iloc[0], 'high')
        
        self.assertAlmostEqual(result_df.loc[result_df['listing_id'] == 4, 'confidence_score'].iloc[0], expected_score_l4, places=1)
        self.assertEqual(result_df.loc[result_df['listing_id'] == 4, 'confidence_level'].iloc[0], 'very_low')


    def test_outlier_treatment(self):
        """Test the _treat_outliers method."""
        target_data = {
            'listing_id': [1, 2, 3, 100, 101], # Added more rows for percentile stability
            'annual_revenue_adj': [1000, 2000, 1000000, 3000, 4000], # 1M is an outlier
            'adr_adj': [50, 60, 800, 70, 80], # 800 is an outlier
            'occupancy_rate_adj': [0.5, 0.6, 1.5, -0.2, 0.8] # 1.5 and -0.2 are outliers
        }
        # Create a larger df to make percentiles more stable for testing
        np.random.seed(0)
        revenue_normal = np.random.normal(loc=50000, scale=10000, size=95)
        adr_normal = np.random.normal(loc=100, scale=20, size=95)
        occupancy_normal = np.random.uniform(low=0.1, high=0.9, size=95)

        target_df_data = {
            'listing_id': list(range(1, 96)) + [96, 97, 98, 99, 100],
            'annual_revenue_adj': np.concatenate([revenue_normal, [1000, 2000, 10000000, 3000, 4000]]), # Extreme outlier
            'adr_adj': np.concatenate([adr_normal, [50, 60, 2500, 70, 80]]), # Extreme outlier
            'occupancy_rate_adj': np.concatenate([occupancy_normal, [0.5, 0.6, 1.5, -0.2, 0.8]]) # Outliers
        }
        target_df = pd.DataFrame(target_df_data)
        
        # Calculate 99th percentile for revenue and ADR before capping
        revenue_cap_value = target_df['annual_revenue_adj'].quantile(self.default_config['outlier_treatment']['revenue_percentile_cap'] / 100)
        adr_cap_value = target_df['adr_adj'].quantile(self.default_config['outlier_treatment']['adr_percentile_cap'] / 100)

        treated_df = self.generator._treat_outliers(target_df.copy()) # Use copy

        # Assert that extreme values are capped
        self.assertLessEqual(treated_df['annual_revenue_adj'].max(), revenue_cap_value)
        self.assertFalse(treated_df['annual_revenue_adj'].max() > revenue_cap_value) # Ensure it's not greater than cap
        self.assertTrue(10000000 not in treated_df['annual_revenue_adj'].values)


        self.assertLessEqual(treated_df['adr_adj'].max(), adr_cap_value)
        self.assertFalse(treated_df['adr_adj'].max() > adr_cap_value)
        self.assertTrue(2500 not in treated_df['adr_adj'].values)
        
        # Assert occupancy rate is clipped between 0 and 1
        self.assertTrue(treated_df['occupancy_rate_adj'].min() >= 0)
        self.assertTrue(treated_df['occupancy_rate_adj'].max() <= 1)
        self.assertTrue(1.5 not in treated_df['occupancy_rate_adj'].values)
        self.assertTrue(-0.2 not in treated_df['occupancy_rate_adj'].values)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
