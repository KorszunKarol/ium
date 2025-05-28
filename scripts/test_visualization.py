#!/usr/bin/env python3
"""
Quick test script to generate a single visualization to verify everything works
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from pathlib import Path

def quick_test():
    """Generate a quick test visualization"""

    # Setup
    data_dir = 'data/processed/etap2/'
    output_dir = 'reports/figures/etap2/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading test data...")
    listings_df = pd.read_pickle(data_dir + 'listings_e2_df.pkl')
    print(f"Loaded {len(listings_df):,} listings")

    # Generate a simple test plot
    if 'room_type' in listings_df.columns:
        plt.figure(figsize=(10, 6))
        room_counts = listings_df['room_type'].value_counts()

        plt.bar(range(len(room_counts)), room_counts.values, color='steelblue', alpha=0.8)
        plt.xticks(range(len(room_counts)), room_counts.index, rotation=45)
        plt.xlabel('Room Type')
        plt.ylabel('Number of Listings')
        plt.title('Test: Room Type Distribution', fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, count in enumerate(room_counts.values):
            plt.text(i, count + 100, f'{count:,}', ha='center', fontsize=10)

        plt.tight_layout()

        # Save
        filepath = os.path.join(output_dir, 'test_room_type_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Test plot saved: {filepath}")
        print("✅ Visualization generation is working correctly!")
        return True
    else:
        print("❌ room_type column not found")
        return False

if __name__ == "__main__":
    print("=== Quick Visualization Test ===")
    success = quick_test()
    if success:
        print("\n🎉 Ready to generate all visualizations!")
        print("Run: ./run_visualizations.sh")
    else:
        print("\n❌ Test failed - check your data")
