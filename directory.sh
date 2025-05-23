#!/bin/bash

echo "Integrating Etap 2 structure into the current IUM project..."

# --- Create essential shared/new directories if they don't exist ---
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/etap1_archive  # Strongly recommended for old Etap 1 data files

mkdir -p api_service         # For the new microservice
mkdir -p models              # For all project models
mkdir -p reports/figures     # General reports folder with a figures subfolder

echo "Ensured directories: data/raw, data/processed, data/etap1_archive, api_service/, models/, reports/figures."

# --- .gitignore: Check and suggest additions ---
if [ -f .gitignore ]; then
  echo ".gitignore exists. Please ensure it covers Python artifacts, venv, OS files, and potentially large data/model files (see suggestions in script comments)."
else
  echo "Creating a basic .gitignore..."
cat << EOF > .gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
.env
venv/
ENV/
pip-selfcheck.json

# Distribution / packaging
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.manifest
*.egg

# Jupyter Notebook checkpoints
.ipynb_checkpoints

# OS specific
*.DS_Store
Thumbs.db

# IDE specific
.vscode/
.idea/

# Data & Models (Consider Git LFS for these or keep them untracked if very large)
# data/raw/
# data/processed/
# models/

# Logs
*.log
logs/
EOF
fi

# --- README.md: Suggest manual update ---
echo ""
echo "IMPORTANT: Please update your main IUM/README.md significantly."
echo "It should now serve as the guide for the entire project (Etap 1 and Etap 2)."
echo "Include:"
echo "  - Overview of both etaps."
echo "  - Setup instructions for the combined project (virtual env, requirements.txt)."
echo "  - Instructions on how to run Etap 2 components (API, new models, A/B eval scripts)."
echo "  - Clear explanation of the data in 'data/raw/' (new Etap 2 data) vs. 'data/etap1_archive/'."
echo "  - Naming conventions used (e.g., 'e1_' vs 'e2_' prefixes for files)."

# --- requirements.txt: Suggest manual update ---
echo ""
echo "IMPORTANT: Review and update your IUM/requirements.txt."
echo "Ensure it includes ALL packages needed for both Etap 1 (if still relevant) and all Etap 2 tasks (e.g., fastapi, uvicorn, modeling libraries like xgboost/lightgbm)."

# --- Placeholder API files (basic structure) ---
# You can create these manually or use this as a starting point
if [ ! -d "api_service" ]; then mkdir -p api_service; fi
touch api_service/main.py
touch api_service/schemas.py
touch api_service/utils.py
echo "Created placeholder files in api_service/. Please develop your API here."

# --- File Organization: Moving and Renaming Files ---
echo ""
echo "Organizing existing files according to the integrated structure..."

# Function to safely move a file
move_file() {
  local src="$1"
  local dest="$2"
  if [ -f "$src" ]; then
    # Ensure destination directory exists if dest is a path like dir/file.ext
    mkdir -p "$(dirname "$dest")"
    if mv "$src" "$dest"; then
      echo "Moved '$src' to '$dest'"
    else
      echo "ERROR: Failed to move '$src' to '$dest'."
    fi
  else
    echo "File '$src' not found. Skipping move."
  fi
}

# Function to safely rename (move) a file
rename_file() {
  local old_name="$1"
  local new_name="$2"
  if [ -f "$old_name" ]; then
    if mv "$old_name" "$new_name"; then
      echo "Renamed '$old_name' to '$new_name'"
    else
      echo "ERROR: Failed to rename '$old_name' to '$new_name'."
    fi
  else
    echo "File '$old_name' not found. Skipping rename."
  fi
}

echo ""
echo "Moving root CSV files (assumed NEW Etap 2 data, except users.csv) to data/raw/..."
# IMPORTANT: The 'users.csv' in root is assumed to be the OLD Etap 1 version.
# The NEW Etap 2 'users.csv' must be manually placed in data/raw/ by the user later.
move_file "listings.csv" "data/raw/listings.csv"
move_file "calendar.csv" "data/raw/calendar.csv"
move_file "reviews.csv" "data/raw/reviews.csv"
move_file "sessions.csv" "data/raw/sessions.csv"

echo ""
echo "Archiving OLD Etap 1 files..."
move_file "users.csv" "data/etap1_archive/users_etap1.csv" # Old users.csv
move_file "dokument.tex" "reports/dokument_etap1.tex"
move_file "london_neighborhood_prices.html" "reports/figures/e1_london_prices.html"

echo ""
echo "Renaming Etap 1 notebooks and scripts with 'e1_' prefix..."
rename_file "data_assessment.ipynb" "e1_data_assessment.ipynb"
rename_file "eda_notebook.ipynb" "e1_eda_notebook.ipynb"
rename_file "data_linkage_visualization.ipynb" "e1_data_linkage_visualization.ipynb"
rename_file "notebook_info.ipynb" "e1_notebook_info.ipynb"

rename_file "data_analysis.py" "e1_data_analysis.py"
rename_file "run_analysis.py" "e1_run_analysis.py"
rename_file "get_csv_shapes.py" "e1_get_csv_shapes.py"

echo ""
echo "File organization actions complete. Please review the output above for any issues."
# --- End File Organization ---

echo ""
echo "Etap 2 integration setup guidance complete."
echo "--------------------------------------------------------------------"
echo "ACTION ITEMS FOR YOU AND YOUR PARTNER:"
echo "1.  If not already done, initialize Git in IUM/: \`git init\`"
echo "2.  Carefully review and commit the .gitignore."
echo "3.  **The script has attempted to move \`listings.csv\`, \`calendar.csv\`, \`reviews.csv\`, \`sessions.csv\` from the root to \`IUM/data/raw/\`. PLEASE VERIFY. You still need to MANUALLY PLACE THE NEW ETAP 2 \`users.csv\` (if different from the old one) and any other new raw CSVs from your source (e.g., WinRAR) into \`IUM/data/raw/\`.**"
echo "4.  **The script has moved the \`users.csv\` (assumed OLD Etap 1 version from root) to \`IUM/data/etap1_archive/users_etap1.csv\`. Manually move any OTHER Etap 1-specific data files that were in the \`IUM/\` root into \`IUM/data/etap1_archive/\`.** This is key to avoid data confusion."
echo "5.  **The script has moved \`dokument.tex\` to \`IUM/reports/dokument_etap1.tex\` and \`london_neighborhood_prices.html\` to \`IUM/reports/figures/e1_london_prices.html\`. Review these changes.**"
echo "6.  **Update your main \`IUM/README.md\` comprehensively.**"
echo "7.  **Update/Consolidate your \`IUM/requirements.txt\` for all project needs.** Install into your virtual environment."
echo "8.  **The script has attempted to rename common Etap 1 notebooks and scripts with an \`e1_\` prefix (e.g., \`data_assessment.ipynb\` -> \`e1_data_assessment.ipynb\`). Please verify and rename any others manually if needed.**"
echo "9.  **Ensure all new Etap 2 notebooks, scripts, processed data, and models use a clear \`e2_\` prefix in their filenames.**"
echo "10. Start developing your Etap 2 components within this integrated structure!"
echo "--------------------------------------------------------------------"