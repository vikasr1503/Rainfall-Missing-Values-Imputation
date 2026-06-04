import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ==========================================================
# FILES
# ==========================================================

INPUT_FILE = "merged_delhi_air_quality_with_features.csv"
OUTPUT_FILE = "merged_delhi_air_quality_with_features.csv"

# ==========================================================
# HAVERSINE DISTANCE
# ==========================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Rows: {len(df):,}")

# ==========================================================
# STATION MASTER TABLE
# ==========================================================

stations = (
    df[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"Stations found: {len(stations)}")

# ==========================================================
# COMPUTE 5 NEAREST STATIONS
# ==========================================================

print("Computing nearest stations...")

nearest_map = {}

for _, row in stations.iterrows():

    sid = row["station_id"]

    distances = []

    for _, other in stations.iterrows():

        other_sid = other["station_id"]

        if sid == other_sid:
            continue

        d = haversine(
            row["latitude"],
            row["longitude"],
            other["latitude"],
            other["longitude"]
        )

        distances.append((other_sid, d))

    distances.sort(key=lambda x: x[1])

    nearest_map[sid] = [
        station_id
        for station_id, _ in distances[:5]
    ]

print("Nearest-neighbour map ready.")

# ==========================================================
# CREATE LOOKUP TABLE
# ==========================================================

print("Creating timestamp-station lookup...")

lookup = df.set_index(
    ["timestamp", "station_id"]
)[
    ["pm2.5", "pm10", "no2", "ozone"]
]

# ==========================================================
# COMPUTE SPATIAL AVERAGES
# ==========================================================

print("Computing spatial averages...")

spatial_pm25 = []
spatial_pm10 = []
spatial_no2 = []
spatial_ozone = []

total_rows = len(df)

for idx, row in enumerate(df.itertuples(index=False), start=1):

    ts = row.timestamp
    sid = row.station_id

    neighbors = nearest_map[sid]

    pm25_vals = []
    pm10_vals = []
    no2_vals = []
    ozone_vals = []

    for n_sid in neighbors:

        try:

            vals = lookup.loc[(ts, n_sid)]

            pm25_vals.append(vals["pm2.5"])
            pm10_vals.append(vals["pm10"])
            no2_vals.append(vals["no2"])
            ozone_vals.append(vals["ozone"])

        except KeyError:
            continue

    spatial_pm25.append(
        np.nanmean(pm25_vals) if len(pm25_vals) else np.nan
    )

    spatial_pm10.append(
        np.nanmean(pm10_vals) if len(pm10_vals) else np.nan
    )

    spatial_no2.append(
        np.nanmean(no2_vals) if len(no2_vals) else np.nan
    )

    spatial_ozone.append(
        np.nanmean(ozone_vals) if len(ozone_vals) else np.nan
    )

    if idx % 100000 == 0:
        print(
            f"Processed {idx:,}/{total_rows:,}"
        )

# ==========================================================
# ADD FEATURES
# ==========================================================

df["spatial_avg_pm25"] = spatial_pm25
df["spatial_avg_pm10"] = spatial_pm10
df["spatial_avg_no2"] = spatial_no2
df["spatial_avg_ozone"] = spatial_ozone

# ==========================================================
# SAVE
# ==========================================================

print("Saving final dataset...")

df.to_csv(OUTPUT_FILE, index=False)

print("\n========================================")
print("Done!")
print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(df):,}")
print("Added:")
print("  spatial_avg_pm25")
print("  spatial_avg_pm10")
print("  spatial_avg_no2")
print("  spatial_avg_ozone")
print("========================================")