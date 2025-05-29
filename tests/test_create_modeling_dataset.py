import sys
import unittest
from pathlib import Path
import os
import shutil
import pandas as pd
import numpy as np
import json # <--- Added import json

# Add the 'scripts' directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Import the main function/module to be tested
from scripts import create_modeling_dataset # So we can access create_modeling_dataset.main

class TestCreateModelingDatasetIntegration(unittest.TestCase):

    def setUp(self):
        """Set up for the integration test."""
        self.test_base_dir = Path("tests/tmp_test_data")
        
        self.test_input_base_dir = self.test_base_dir / "processed/etap2" # Mimics script's structure
        self.test_input_features_dir = self.test_input_base_dir / "feature_engineered"
        self.test_input_targets_dir = self.test_input_base_dir / "target_variables"
        self.test_output_modeling_dir = self.test_input_base_dir / "modeling"

        # Create temporary directories
        os.makedirs(self.test_input_features_dir, exist_ok=True)
        os.makedirs(self.test_input_targets_dir, exist_ok=True)
        os.makedirs(self.test_output_modeling_dir, exist_ok=True)

        # Create sample DataFrames
        self.sample_features_data = {
            'id': [1, 2, 3],
            'feature_a': ['a1', 'a2', 'a3'],
            'feature_b': [1.1, 2.2, 3.3],
            # Add other columns that analyze_dataset_quality might check to avoid NaNs if possible
            'price': [100, 150, 120],
            'room_type_Entire home/apt': [1,0,1],
            'accommodates': [4,2,3],
            'bedrooms': [2,1,1],
            'beds': [2,1,2],
            'latitude': [40.1, 40.2, 40.3],
            'longitude': [-74.1, -74.2, -74.3],
            'review_scores_rating': [4.5, 4.0, 5.0],
            'number_of_reviews': [10,5,20],
            'amenities_count': [5,3,7],
            'distance_to_center': [1.0, 2.5, 0.5],
            'host_is_superhost': [True, False, True],
            'neighbourhood_cleansed': ['A', 'B', 'A'] # For geo coverage check
        }
        self.sample_features_df = pd.DataFrame(self.sample_features_data)

        self.sample_targets_data = {
            'listing_id': [1, 2, 4], # ID 1 and 2 match, ID 3 from features is missing, ID 4 from targets is missing in features
            'annual_revenue_adj': [50000.0, 75000.0, 90000.0],
            'occupancy_rate_adj': [0.7, 0.8, 0.85],
            'adr_adj': [200.0, 250.0, 300.0],
            'confidence_score': [85.0, 90.0, 80.0]
        }
        self.sample_targets_df = pd.DataFrame(self.sample_targets_data)

        # Save sample DataFrames to temporary input directories
        self.features_pkl_path = self.test_input_features_dir / "listings_feature_engineered.pkl"
        self.targets_pkl_path = self.test_input_targets_dir / "reliable_targets.pkl"
        
        self.sample_features_df.to_pickle(self.features_pkl_path)
        self.sample_targets_df.to_pickle(self.targets_pkl_path)

        # Store original paths from the script to restore them later
        # These are module-level variables in the script
        self.original_processed_data_base_dir = getattr(create_modeling_dataset, 'processed_data_base_dir', None)
        # output_dir is defined within main(), so we don't store/restore it at module level
        self.original_key_features_for_check = getattr(create_modeling_dataset, 'key_features_for_check', None)


    def test_create_modeling_dataset_integration(self):
        """Test the full execution of the create_modeling_dataset.py script."""
        
        original_load_data = create_modeling_dataset.load_data
        original_save_dataset = create_modeling_dataset.save_modeling_dataset
        # Store original key_features_for_check if it exists, otherwise it might be None
        original_kf_check_in_module = getattr(create_modeling_dataset, 'key_features_for_check', None)


        def mock_load_data(processed_data_base_dir_arg):
            # Ignore the argument, use test paths
            self.assertTrue(str(self.test_input_base_dir) in processed_data_base_dir_arg or "data/processed/etap2" in processed_data_base_dir_arg)
            return pd.read_pickle(self.features_pkl_path), pd.read_pickle(self.targets_pkl_path)

        def mock_save_modeling_dataset(modeling_dataset, quality_report, output_dir_arg):
            # Check that output_dir_arg is what main() would construct based on its local processed_data_base_dir
            # For the actual test run, we don't strictly need to assert output_dir_arg if mock saves to self.test_output_modeling_dir
            os.makedirs(self.test_output_modeling_dir, exist_ok=True)
            dataset_path_pkl = self.test_output_modeling_dir / "modeling_dataset.pkl"
            dataset_path_csv = self.test_output_modeling_dir / "modeling_dataset.csv" 
            quality_path = self.test_output_modeling_dir / "dataset_quality_report.json"
            
            modeling_dataset.to_pickle(dataset_path_pkl)
            modeling_dataset.to_csv(dataset_path_csv, index=False) 
            with open(quality_path, "w") as f:
                json.dump(quality_report, f, indent=2, default=str)
            return {"dataset_pkl": dataset_path_pkl, "dataset_csv": dataset_path_csv, "quality_report": quality_path}

        create_modeling_dataset.load_data = mock_load_data
        create_modeling_dataset.save_modeling_dataset = mock_save_modeling_dataset
        
        # Monkeypatch the global key_features_for_check at the end of the script create_modeling_dataset.py
        # This list is used by main() when it calls analyze_dataset_quality and then prints the report.
        if hasattr(create_modeling_dataset, 'key_features_for_check'):
            create_modeling_dataset.key_features_for_check = [ 
                'price', 'room_type_Entire home/apt', 'accommodates', 'bedrooms', 'beds',
                'latitude', 'longitude', 'review_scores_rating', 'number_of_reviews',
                'amenities_count', 'distance_to_center', 'host_is_superhost'
            ]
        
        try:
            # Call main. main() itself will use its hardcoded "data/processed/etap2/" for its local
            # processed_data_base_dir, which is passed to our mock_load_data.
            # Our mock_save_dataset will save to the test directory.
            create_modeling_dataset.main()
        finally:
            # Restore original functions
            create_modeling_dataset.load_data = original_load_data
            create_modeling_dataset.save_modeling_dataset = original_save_dataset
            if self.original_key_features_for_check is not None and original_kf_check_in_module is not None:
                 create_modeling_dataset.key_features_for_check = self.original_key_features_for_check


        expected_dataset_pkl = self.test_output_modeling_dir / "modeling_dataset.pkl"
        expected_dataset_csv = self.test_output_modeling_dir / "modeling_dataset.csv"
        expected_quality_report = self.test_output_modeling_dir / "dataset_quality_report.json"

        self.assertTrue(os.path.exists(expected_dataset_pkl))
        self.assertTrue(os.path.exists(expected_dataset_csv))
        self.assertTrue(os.path.exists(expected_quality_report))

        loaded_modeling_df = pd.read_pickle(expected_dataset_pkl)

        self.assertEqual(len(loaded_modeling_df), 2) 
        self.assertListEqual(sorted(list(loaded_modeling_df['id'].unique())), [1, 2])

        for col in ['feature_a', 'feature_b', 'price', 'accommodates']: 
            self.assertIn(col, loaded_modeling_df.columns)
        
        self.assertNotIn('listing_id', loaded_modeling_df.columns) 
        self.assertIn('annual_revenue_adj', loaded_modeling_df.columns)
        self.assertIn('confidence_score', loaded_modeling_df.columns)

        row_id_1 = loaded_modeling_df[loaded_modeling_df['id'] == 1].iloc[0]
        self.assertEqual(row_id_1['feature_a'], 'a1')
        self.assertEqual(row_id_1['annual_revenue_adj'], 50000.0)

        with open(expected_quality_report, 'r') as f:
            quality_data = json.load(f)
        self.assertEqual(quality_data['dataset_size'], 2)
        self.assertEqual(quality_data['target_statistics']['annual_revenue_adj']['count'], 2)
        # Check one feature from the patched key_features_for_check list
        self.assertTrue(quality_data['feature_completeness']['accommodates']['completeness_percentage'] == 100.0)


    def tearDown(self):
        """Clean up temporary directories and files."""
        if os.path.exists(self.test_base_dir):
            shutil.rmtree(self.test_base_dir)
        
        # Restore original key_features_for_check if it was patched
        if self.original_key_features_for_check is not None and hasattr(create_modeling_dataset, 'key_features_for_check'):
            create_modeling_dataset.key_features_for_check = self.original_key_features_for_check


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
