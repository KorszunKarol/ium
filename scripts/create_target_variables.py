import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


class TargetVariableGenerator:
    """
    Creates reliable, seasonally-adjusted target variables for Airbnb listings.

    This class generates annual revenue, occupancy rate, and ADR (Average Daily Rate)
    targets that are adjusted for seasonality and market fluctuations, providing
    stable target variables for machine learning models.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the target variable generator.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.seasonality_profile = self._get_seasonality_profile()
        self.stats = {}
        self.log_messages = []

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration or use defaults."""
        default_config = {
            "minimum_observation_days": 30,
            "minimum_bookings": 3,
            "confidence_thresholds": {
                "high": 180,
                "medium": 90,
                "low": 30,
            },
            "outlier_treatment": {
                "revenue_percentile_cap": 100,
                "adr_percentile_cap": 100,
                "min_reasonable_adr": 10,
                "max_reasonable_adr": 2000,
            },
            "seasonality": {
                "use_monthly_adjustment": True,
                "use_weekday_adjustment": True,
                "market_location": "london",
            },
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def _get_seasonality_profile(self) -> Dict:
        """
        Get seasonality adjustment factors for the market.

        These factors represent how demand/pricing typically varies by month
        relative to the annual average. For example, 1.2 means 20% above average.

        For London Airbnb market (based on typical patterns):
        - Summer months (Jun-Aug) are peak season
        - December is also high due to holidays
        - January-February are typically lowest
        """

        london_seasonality = {
            1: 0.85,
            2: 0.88,
            3: 0.95,
            4: 1.05,
            5: 1.10,
            6: 1.25,
            7: 1.30,
            8: 1.25,
            9: 1.15,
            10: 1.05,
            11: 0.95,
            12: 1.15,
        }


        weekday_seasonality = {
            0: 0.90,
            1: 0.92,
            2: 0.95,
            3: 0.98,
            4: 1.10,
            5: 1.20,
            6: 1.15,
        }

        return {
            "monthly": london_seasonality,
            "weekday": weekday_seasonality,
        }

    def log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        print(log_entry)

    def generate_targets(
        self,
        calendar_df: pd.DataFrame,
        listings_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Main method to generate seasonally-adjusted target variables.

        Args:
            calendar_df: Calendar dataframe with booking data
            listings_df: Listings dataframe (optional, for filtering/context)

        Returns:
            DataFrame with target variables for each listing
        """
        self.log("🎯 Starting Target Variable Generation")
        self.log(f"Calendar data shape: {calendar_df.shape}")


        calendar_clean = self._prepare_calendar_data(calendar_df)


        if listings_df is not None:
            calendar_clean = self._filter_with_listings_context(calendar_clean, listings_df)


        observed_performance = self._calculate_observed_performance(calendar_clean)


        adjusted_performance = self._apply_seasonality_adjustments(
            observed_performance, calendar_clean
        )


        target_variables = self._annualize_metrics(adjusted_performance)


        target_variables = self._calculate_confidence_scores(target_variables)


        target_variables = self._treat_outliers(target_variables)


        target_variables = self._final_validation(target_variables)

        self.log(f"✅ Target generation complete. Generated targets for {len(target_variables)} listings")
        return target_variables

    def _prepare_calendar_data(self, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean calendar data."""
        self.log("📅 Preparing calendar data")

        df = calendar_df.copy()


        df['date'] = pd.to_datetime(df['date'])
        df['available'] = df['available'].astype(bool)
        df['price_cleaned'] = pd.to_numeric(df['price_cleaned'], errors='coerce')


        initial_rows = len(df)
        df = df[df['price_cleaned'] > 0]
        df = df[df['price_cleaned'].notna()]

        if len(df) < initial_rows:
            self.log(f"Removed {initial_rows - len(df)} rows with invalid prices")


        df['is_booked'] = ~df['available']


        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday
        df['year'] = df['date'].dt.year
        df['week_of_year'] = df['date'].dt.isocalendar().week


        df['revenue'] = df['price_cleaned'] * df['is_booked']

        self.stats['calendar_date_range'] = {
            'start': df['date'].min(),
            'end': df['date'].max(),
            'total_days': (df['date'].max() - df['date'].min()).days,
            'unique_listings': df['listing_id'].nunique(),
        }

        self.log(f"Calendar prepared: {df['date'].min()} to {df['date'].max()}")
        self.log(f"Covering {self.stats['calendar_date_range']['total_days']} days for {self.stats['calendar_date_range']['unique_listings']} listings")

        return df

    def _filter_with_listings_context(
        self, calendar_df: pd.DataFrame, listings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter calendar data using listings context."""
        self.log("🏠 Applying listings context filters")


        if 'minimum_nights' in listings_df.columns:
            reasonable_min_nights = listings_df['minimum_nights'] <= 30
            valid_listings = listings_df[reasonable_min_nights]['id']

            initial_count = calendar_df['listing_id'].nunique()
            calendar_df = calendar_df[calendar_df['listing_id'].isin(valid_listings)]
            final_count = calendar_df['listing_id'].nunique()

            self.log(f"Filtered out {initial_count - final_count} listings with extreme minimum nights")

        return calendar_df

    def _calculate_observed_performance(self, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate raw observed performance metrics for each listing."""
        self.log("📊 Calculating observed performance metrics")


        listing_stats = calendar_df.groupby('listing_id').agg({
            'is_booked': ['sum', 'count'],
            'revenue': 'sum',
            'price_cleaned': 'mean',
            'date': ['min', 'max'],
        }).reset_index()


        listing_stats.columns = [
            'listing_id', 'total_bookings', 'total_observations',
            'total_revenue', 'avg_listed_price', 'obs_start_date', 'obs_end_date'
        ]


        listing_stats['observation_days'] = (
            listing_stats['obs_end_date'] - listing_stats['obs_start_date']
        ).dt.days + 1


        listing_stats['raw_occupancy_rate'] = (
            listing_stats['total_bookings'] / listing_stats['total_observations']
        )


        adr_data = calendar_df[calendar_df['is_booked']].groupby('listing_id').agg({
            'price_cleaned': 'mean'
        }).reset_index()
        adr_data.columns = ['listing_id', 'raw_adr']


        listing_stats = listing_stats.merge(adr_data, on='listing_id', how='left')


        listing_stats['raw_adr'] = listing_stats['raw_adr'].fillna(
            listing_stats['avg_listed_price']
        )


        listing_stats['raw_daily_revenue'] = (
            listing_stats['raw_occupancy_rate'] * listing_stats['raw_adr']
        )

        self.log(f"Calculated performance for {len(listing_stats)} listings")
        self.log(f"Average observation period: {listing_stats['observation_days'].mean():.1f} days")

        return listing_stats

    def _apply_seasonality_adjustments(
        self, performance_df: pd.DataFrame, calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply seasonality adjustments to normalize performance metrics."""
        self.log("🌊 Applying seasonality adjustments")


        listing_seasonality = []

        for listing_id in performance_df['listing_id']:
            listing_calendar = calendar_df[calendar_df['listing_id'] == listing_id]

            if len(listing_calendar) == 0:
                continue


            monthly_factors = []
            monthly_weights = []

            for _, row in listing_calendar.iterrows():
                monthly_factor = self.seasonality_profile['monthly'][row['month']]
                monthly_factors.append(monthly_factor)

                weight = 2.0 if row['is_booked'] else 1.0
                monthly_weights.append(weight)


            avg_monthly_factor = np.average(monthly_factors, weights=monthly_weights)


            weekday_factors = []
            weekday_weights = []

            for _, row in listing_calendar.iterrows():
                weekday_factor = self.seasonality_profile['weekday'][row['weekday']]
                weekday_factors.append(weekday_factor)
                weight = 2.0 if row['is_booked'] else 1.0
                weekday_weights.append(weight)

            avg_weekday_factor = np.average(weekday_factors, weights=weekday_weights)


            combined_factor = (avg_monthly_factor + avg_weekday_factor) / 2

            listing_seasonality.append({
                'listing_id': listing_id,
                'monthly_seasonality_factor': avg_monthly_factor,
                'weekday_seasonality_factor': avg_weekday_factor,
                'combined_seasonality_factor': combined_factor,
            })

        seasonality_df = pd.DataFrame(listing_seasonality)


        adjusted_df = performance_df.merge(seasonality_df, on='listing_id', how='left')


        adjusted_df['seasonally_adj_adr'] = (
            adjusted_df['raw_adr'] / adjusted_df['combined_seasonality_factor']
        )

        adjusted_df['seasonally_adj_occupancy'] = (
            adjusted_df['raw_occupancy_rate'] / adjusted_df['monthly_seasonality_factor']
        )

        adjusted_df['seasonally_adj_daily_revenue'] = (
            adjusted_df['seasonally_adj_occupancy'] * adjusted_df['seasonally_adj_adr']
        )

        self.log(f"Applied seasonality adjustments to {len(adjusted_df)} listings")

        return adjusted_df

    def _annualize_metrics(self, adjusted_df: pd.DataFrame) -> pd.DataFrame:
        """Convert observed metrics to annualized projections."""
        self.log("📈 Annualizing metrics to yearly projections")


        adjusted_df['annualization_factor'] = 365.25 / adjusted_df['observation_days']


        adjusted_df['annualization_factor'] = adjusted_df['annualization_factor'].clip(
            upper=12.0
        )


        adjusted_df['annual_revenue_adj'] = (
            adjusted_df['seasonally_adj_daily_revenue'] * 365.25
        )


        adjusted_df['occupancy_rate_adj'] = adjusted_df['seasonally_adj_occupancy']


        adjusted_df['adr_adj'] = adjusted_df['seasonally_adj_adr']


        adjusted_df['annual_revenue_raw'] = (
            adjusted_df['total_revenue'] * adjusted_df['annualization_factor']
        )

        self.log(f"Annualized metrics for {len(adjusted_df)} listings")

        return adjusted_df

    def _calculate_confidence_scores(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence scores based on data quality and quantity."""
        self.log("🎯 Calculating confidence scores")

        confidence_scores = []

        for _, row in target_df.iterrows():
            score_components = []


            obs_days = row['observation_days']
            if obs_days >= self.config['confidence_thresholds']['high']:
                obs_score = 40
            elif obs_days >= self.config['confidence_thresholds']['medium']:
                obs_score = 30
            elif obs_days >= self.config['confidence_thresholds']['low']:
                obs_score = 20
            else:
                obs_score = max(0, 20 * obs_days / self.config['confidence_thresholds']['low'])

            score_components.append(obs_score)


            total_bookings = row['total_bookings']
            if total_bookings >= 20:
                booking_score = 30
            elif total_bookings >= 10:
                booking_score = 25
            elif total_bookings >= self.config['minimum_bookings']:
                booking_score = 15
            else:
                booking_score = max(0, 15 * total_bookings / self.config['minimum_bookings'])

            score_components.append(booking_score)



            adr = row['adr_adj']
            occupancy = row['occupancy_rate_adj']

            consistency_score = 20


            if adr < self.config['outlier_treatment']['min_reasonable_adr'] or \
               adr > self.config['outlier_treatment']['max_reasonable_adr']:
                consistency_score -= 10

            if occupancy < 0 or occupancy > 1:
                consistency_score -= 10

            if row['annualization_factor'] > 6:
                consistency_score -= 5

            score_components.append(max(0, consistency_score))



            seasonality_score = min(10, obs_days / 30)
            score_components.append(seasonality_score)


            total_score = sum(score_components)


            if total_score >= 80:
                confidence_level = 'high'
            elif total_score >= 60:
                confidence_level = 'medium'
            elif total_score >= 40:
                confidence_level = 'low'
            else:
                confidence_level = 'very_low'

            confidence_scores.append({
                'listing_id': row['listing_id'],
                'confidence_score': total_score,
                'confidence_level': confidence_level,
                'obs_score': obs_score,
                'booking_score': booking_score,
                'consistency_score': consistency_score,
                'seasonality_score': seasonality_score,
            })

        confidence_df = pd.DataFrame(confidence_scores)


        result_df = target_df.merge(confidence_df, on='listing_id', how='left')

        confidence_dist = result_df['confidence_level'].value_counts()
        self.log(f"Confidence distribution: {confidence_dist.to_dict()}")

        return result_df

    def _treat_outliers(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier treatment to target variables."""
        self.log("🔧 Treating outliers in target variables")


        outlier_treatments = {
            'annual_revenue_adj': self.config['outlier_treatment']['revenue_percentile_cap'],
            'adr_adj': self.config['outlier_treatment']['adr_percentile_cap'],
            'occupancy_rate_adj': 100,
        }

        for col, percentile_cap in outlier_treatments.items():
            if col in target_df.columns:

                lower_bound = target_df[col].quantile(0.01)







                if col == 'annual_revenue_adj':
                    outliers_count_lower = (target_df[col] < lower_bound).sum()
                    target_df[col] = target_df[col].clip(lower=lower_bound)
                    if outliers_count_lower > 0:
                        self.log(f"Capped {outliers_count_lower} lower outliers in {col}")
                else:
                    upper_bound = target_df[col].quantile(percentile_cap / 100)
                    outliers_count = ((target_df[col] < lower_bound) |
                                     (target_df[col] > upper_bound)).sum()
                    target_df[col] = target_df[col].clip(lower=lower_bound, upper=upper_bound)

                    if outliers_count > 0:
                        self.log(f"Capped {outliers_count} outliers in {col}")


        target_df['occupancy_rate_adj'] = target_df['occupancy_rate_adj'].clip(0, 1)

        return target_df

    def _final_validation(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup of target variables."""
        self.log("✅ Final validation of target variables")

        initial_count = len(target_df)


        min_obs_days = self.config['minimum_observation_days']
        min_bookings = self.config['minimum_bookings']

        valid_listings = (
            (target_df['observation_days'] >= min_obs_days) &
            (target_df['total_bookings'] >= min_bookings) &
            (target_df['annual_revenue_adj'] > 0) &
            (target_df['adr_adj'] > 0) &
            (target_df['occupancy_rate_adj'] >= 0)
        )

        target_df = target_df[valid_listings]

        removed_count = initial_count - len(target_df)
        if removed_count > 0:
            self.log(f"Removed {removed_count} listings with insufficient/invalid data")


        final_columns = [
            'listing_id',
            'annual_revenue_adj',
            'occupancy_rate_adj',
            'adr_adj',
            'confidence_score',
            'confidence_level',
            'observation_days',
            'total_bookings',
            'total_observations',
            'annualization_factor',
            'combined_seasonality_factor',

            'annual_revenue_raw',
            'raw_occupancy_rate',
            'raw_adr',
        ]


        target_df['data_quality_flags'] = target_df.apply(self._generate_quality_flags, axis=1)

        result_df = target_df[final_columns + ['data_quality_flags']].copy()


        self.stats['final_targets'] = {
            'total_listings': len(result_df),
            'confidence_distribution': result_df['confidence_level'].value_counts().to_dict(),
            'target_statistics': {
                'annual_revenue_adj': {
                    'mean': result_df['annual_revenue_adj'].mean(),
                    'median': result_df['annual_revenue_adj'].median(),
                    'std': result_df['annual_revenue_adj'].std(),
                },
                'occupancy_rate_adj': {
                    'mean': result_df['occupancy_rate_adj'].mean(),
                    'median': result_df['occupancy_rate_adj'].median(),
                    'std': result_df['occupancy_rate_adj'].std(),
                },
                'adr_adj': {
                    'mean': result_df['adr_adj'].mean(),
                    'median': result_df['adr_adj'].median(),
                    'std': result_df['adr_adj'].std(),
                },
            }
        }

        self.log(f"Final validation complete. {len(result_df)} listings with valid targets")

        return result_df

    def _generate_quality_flags(self, row: pd.Series) -> List[str]:
        """Generate data quality flags for each listing."""
        flags = []

        if row['observation_days'] < 60:
            flags.append('short_observation')

        if row['total_bookings'] < 5:
            flags.append('few_bookings')

        if row['annualization_factor'] > 4:
            flags.append('high_extrapolation')

        if row['confidence_score'] < 50:
            flags.append('low_confidence')

        return flags

    def save_artifacts(self, target_df: pd.DataFrame, output_dir: str):
        """Save target variables and related artifacts."""
        os.makedirs(output_dir, exist_ok=True)


        target_df.to_pickle(os.path.join(output_dir, 'reliable_targets.pkl'))
        target_df.to_csv(os.path.join(output_dir, 'reliable_targets.csv'), index=False)


        with open(os.path.join(output_dir, 'target_generation_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)


        with open(os.path.join(output_dir, 'target_generation_stats.json'), 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)


        with open(os.path.join(output_dir, 'seasonality_profile.json'), 'w') as f:
            json.dump(self.seasonality_profile, f, indent=2)


        with open(os.path.join(output_dir, 'target_generation_log.txt'), 'w') as f:
            f.write('\n'.join(self.log_messages))

        self.log(f"All artifacts saved to {output_dir}")

    def generate_report(self, target_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive report on target variable generation."""
        report = {
            'summary': {
                'total_listings_processed': len(target_df),
                'calendar_date_range': self.stats.get('calendar_date_range', {}),
                'target_statistics': self.stats.get('final_targets', {}).get('target_statistics', {}),
            },
            'data_quality': {
                'confidence_distribution': self.stats.get('final_targets', {}).get('confidence_distribution', {}),
                'average_observation_days': target_df['observation_days'].mean(),
                'average_bookings_per_listing': target_df['total_bookings'].mean(),
                'annualization_factor_stats': {
                    'mean': target_df['annualization_factor'].mean(),
                    'median': target_df['annualization_factor'].median(),
                    'max': target_df['annualization_factor'].max(),
                }
            },
            'seasonality_adjustments': {
                'monthly_factors_used': self.seasonality_profile['monthly'],
                'weekday_factors_used': self.seasonality_profile['weekday'],
                'average_adjustment_factor': target_df['combined_seasonality_factor'].mean(),
            },
            'target_variable_ranges': {
                'annual_revenue_adj': {
                    'min': target_df['annual_revenue_adj'].min(),
                    'max': target_df['annual_revenue_adj'].max(),
                    'q25': target_df['annual_revenue_adj'].quantile(0.25),
                    'q75': target_df['annual_revenue_adj'].quantile(0.75),
                },
                'occupancy_rate_adj': {
                    'min': target_df['occupancy_rate_adj'].min(),
                    'max': target_df['occupancy_rate_adj'].max(),
                    'q25': target_df['occupancy_rate_adj'].quantile(0.25),
                    'q75': target_df['occupancy_rate_adj'].quantile(0.75),
                },
                'adr_adj': {
                    'min': target_df['adr_adj'].min(),
                    'max': target_df['adr_adj'].max(),
                    'q25': target_df['adr_adj'].quantile(0.25),
                    'q75': target_df['adr_adj'].quantile(0.75),
                },
            },
            'recommendations': [
                'Focus modeling on listings with medium-high confidence scores',
                'Consider separate models for different confidence levels',
                'Monitor seasonality adjustments for accuracy',
                'Validate target variables against external benchmarks if available',
                'Use confidence scores as sample weights in model training',
            ]
        }

        return report


def main():
    """Example usage of the target variable generator."""


    generator = TargetVariableGenerator()

    print("Loading data...")
    data_dir = "data/processed/etap2/"

    try:

        calendar_df = pd.read_pickle(os.path.join(data_dir, "calendar_e2_df.pkl"))
        listings_df = pd.read_pickle(os.path.join(data_dir, "listings_e2_df.pkl"))

        print(f"Loaded calendar: {calendar_df.shape}")
        print(f"Loaded listings: {listings_df.shape}")


        print("\nGenerating target variables...")
        target_variables = generator.generate_targets(calendar_df, listings_df)


        output_dir = "data/processed/etap2/target_variables/"
        os.makedirs(output_dir, exist_ok=True)


        generator.save_artifacts(target_variables, output_dir)


        report = generator.generate_report(target_variables)
        with open(os.path.join(output_dir, 'target_generation_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✅ Target variable generation complete!")
        print(f"Generated targets for {len(target_variables)} listings")
        print(f"Output saved to: {output_dir}")

        print(f"\nTarget variable summary:")
        print(f"Annual Revenue (adj): ${target_variables['annual_revenue_adj'].median():.0f} median")
        print(f"Occupancy Rate (adj): {target_variables['occupancy_rate_adj'].median():.1%} median")
        print(f"ADR (adj): ${target_variables['adr_adj'].median():.0f} median")

        print(f"\nConfidence distribution:")
        conf_dist = target_variables['confidence_level'].value_counts()
        for level, count in conf_dist.items():
            print(f"  {level}: {count} listings ({count/len(target_variables)*100:.1f}%)")

        print(f"\nSample of target variables:")
        sample_cols = ['listing_id', 'annual_revenue_adj', 'occupancy_rate_adj',
                      'adr_adj', 'confidence_level', 'observation_days']
        print(target_variables[sample_cols].head(10))

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure data files exist in the specified directory.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
