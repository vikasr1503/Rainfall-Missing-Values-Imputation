import pandas as pd

INPUT_FILE = "merged_delhi_air_quality_data.csv"
OUTPUT_FILE = "merged_delhi_air_quality_with_features.csv"

# Load existing merged file
df = pd.read_csv(INPUT_FILE)

# Timestamp conversion
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# Sort for time-series features
df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

# Lag features (15-min data)
# ==================================================
# PM2.5
# ==================================================

df["lag_1_day_pm25"] = (
    df.groupby("station_id")["pm2.5"].shift(96)
)

df["lag_7_day_pm25"] = (
    df.groupby("station_id")["pm2.5"].shift(672)
)

# ==================================================
# PM10
# ==================================================

df["lag_1_day_pm10"] = (
    df.groupby("station_id")["pm10"].shift(96)
)

df["lag_7_day_pm10"] = (
    df.groupby("station_id")["pm10"].shift(672)
)

# ==================================================
# NO2
# ==================================================

df["lag_1_day_no2"] = (
    df.groupby("station_id")["no2"].shift(96)
)

df["lag_7_day_no2"] = (
    df.groupby("station_id")["no2"].shift(672)
)

# ==================================================
# OZONE
# ==================================================

df["lag_1_day_ozone"] = (
    df.groupby("station_id")["ozone"].shift(96)
)

df["lag_7_day_ozone"] = (
    df.groupby("station_id")["ozone"].shift(672)
)

# Example extra time columns
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Rows:", len(df))