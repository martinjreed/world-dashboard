import pandas as pd

# Load your main long dataset
df = pd.read_csv("world_data_long.csv")

# Load the new CSV (country names, not ISO)
students = pd.read_csv("new-UG-students-countries-2425.csv")

# Make sure the column names are consistent
students.columns = ["country", "value"]

# Add metadata columns to match long format
students["indicator"] = "students_2024"
students["year"] = 2024

# Try to match ISO3 codes from your main dataset (using country name)
iso_lookup = df[["iso3", "country"]].drop_duplicates()
students = students.merge(iso_lookup, on="country", how="left")

# Some countries might fail to match due to naming differences
missing = students[students["iso3"].isna()]
if not missing.empty:
    print("⚠️ Unmatched countries:\n", missing["country"].tolist())

# Append to the dataset and save as a new file
out = pd.concat([df, students[["iso3","country","year","indicator","value"]]], ignore_index=True)
out.to_csv("world_data_long_plus.csv", index=False)

print(f"✅ Added {len(students)} new rows as 'students_2024'. Saved to world_data_long_plus.csv")
