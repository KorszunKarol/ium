#!/usr/bin/env python3
"""
Summary script to show what visualizations have been generated
"""

import os
from pathlib import Path

def show_summary():
    """Display a summary of generated visualizations"""

    output_dir = 'reports/figures/etap2/'

    print("🎨 PROFESSIONAL AIRBNB DATA VISUALIZATIONS")
    print("=" * 50)
    print(f"📁 Location: {output_dir}")
    print()

    if not os.path.exists(output_dir):
        print("❌ Output directory not found!")
        return

    files = sorted([f for f in os.listdir(output_dir) if f.endswith(('.png', '.html'))])

    if not files:
        print("❌ No visualization files found!")
        return

    print(f"✅ Successfully generated {len(files)} visualizations:")
    print()

    categories = {
        "Missing Data Analysis": ["01_", "02_"],
        "Neighborhood Analysis": ["03_", "04_"],
        "Geographic Distribution": ["05_"],
        "Property Features": ["06_", "07_", "08_"],
        "Review Scores": ["09_", "10_"],
        "Calendar & Pricing": ["11_", "12_"],
        "Revenue Analysis": ["13_", "14_"]
    }

    for category, prefixes in categories.items():
        print(f"📊 {category}:")
        category_files = [f for f in files if any(f.startswith(p) for p in prefixes)]
        for file in category_files:
            file_path = os.path.join(output_dir, file)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            file_type = "🌐 Interactive" if file.endswith('.html') else "📈 Static"
            print(f"   {file_type} {file} ({size_mb:.1f} MB)")
        print()

    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files) / (1024*1024)
    print(f"📊 Total size: {total_size:.1f} MB")
    print()
    print("🚀 READY FOR USE:")
    print("   • High-resolution PNG files for reports and presentations")
    print("   • Interactive HTML map for data exploration")
    print("   • Professional styling suitable for academic/business use")
    print("   • All files saved at 300 DPI for publication quality")
    print()
    print("📖 Documentation: reports/figures/etap2/README.md")

if __name__ == "__main__":
    show_summary()
