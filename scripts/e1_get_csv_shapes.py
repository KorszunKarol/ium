import pandas as pd

sessions_file = "sessions.csv"
users_file = "users.csv"

try:
    # Reading sessions.csv - might be large, consider chunking if memory is an issue
    # For now, just try reading it directly to get the shape
    print(f"Reading {sessions_file}...")
    # It's often helpful to specify dtype='unicode' or low_memory=False for mixed-type columns
    sessions_df = pd.read_csv(sessions_file, low_memory=False)
    print(f"Shape of {sessions_file}: {sessions_df.shape}")
except Exception as e:
    print(f"Error reading {sessions_file}: {e}")

try:
    # Reading users.csv
    print(f"Reading {users_file}...")
    users_df = pd.read_csv(users_file, low_memory=False)
    print(f"Shape of {users_file}: {users_df.shape}")
except Exception as e:
    print(f"Error reading {users_file}: {e}")

print("Script finished.")
