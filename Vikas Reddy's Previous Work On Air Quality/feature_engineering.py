import os
import re
import glob
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================
BASE_FOLDER = "."
RAW_FOLDER = os.path.join(BASE_FOLDER, "Raw_data_15Min_2023_2024_2025_DelhiStations")
COORD_FILE = os.path.join(BASE_FOLDER, "delhi_monitoring_stations_coordinates.csv")
OUTPUT_FILE = os.path.join(BASE_FOLDER, "merged_delhi_air_quality_data.csv")

# ==========================================================
# HELPER: CLEAN TEXT FOR MATCHING
# ==========================================================
def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text

# ==========================================================
# HELPER: EXTRACT STATION NAME FROM FILE NAME
# ==========================================================
def extract_station_name(filename):
    """
    Example:
    Raw_data_15Min_2023_site_103_CRRI_Mathura_Road_Delhi_IMD_15Min.csv
    -> CRRI Mathura Road
    """
    name = os.path.basename(filename).replace(".csv", "")
    parts = name.split("_")

    try:
        site_index = parts.index("site")
        station_parts = parts[site_index + 2:]   # skip site + number
    except ValueError:
        station_parts = parts

    remove_words = {
        "delhi", "imd", "cpcb", "dpcc", "iitm",
        "15min", "raw", "data", "2023", "2024", "2025"
    }

    cleaned = []
    for p in station_parts:
        if p.lower() not in remove_words and p.strip() != "":
            cleaned.append(p)

    return " ".join(cleaned).strip()

# ==========================================================
# LOAD COORDINATES FILE
# Correct structure:
# station_id,Station,Latitude,Longitude
# ==========================================================
coords = pd.read_csv(COORD_FILE)

# Clean headers
coords.columns = [c.strip() for c in coords.columns]

# Rename columns consistently
coords = coords.rename(columns={
    "Station": "station_name",
    "Latitude": "latitude",
    "Longitude": "longitude"
})

# Keep required columns
coords = coords[["station_id", "station_name", "latitude", "longitude"]].copy()

# Matching key
coords["match_key"] = coords["station_name"].apply(normalize_text)

station_col = "station_name"
lat_col = "latitude"
lon_col = "longitude"
id_col = "station_id"

# ==========================================================
# PROCESS ALL RAW FILES
# ==========================================================
all_data = []

csv_files = glob.glob(os.path.join(RAW_FOLDER, "*.csv"))

print(f"Found {len(csv_files)} files...\n")

for file in csv_files:
    try:
        # ------------------------------------------
        # Extract station name
        # ------------------------------------------
        guessed_station = extract_station_name(file)

        # Manual fixes for special names
        special_map = {
            "iit delhi": "IIT Delhi",
            "ihbas dilshad garden": "IHBAS, Dilshad Garden",
            "north campus du": "North Campus, DU",
        }

        guessed_key = normalize_text(guessed_station)

        if guessed_key in [normalize_text(k) for k in special_map.keys()]:
            for k, v in special_map.items():
                if guessed_key == normalize_text(k):
                    guessed_station = v
                    guessed_key = normalize_text(v)
                    break

        # ------------------------------------------
        # Match coordinates
        # ------------------------------------------
        match = coords[coords["match_key"] == guessed_key]

        # Partial fallback
        if match.empty:
            for _, row in coords.iterrows():
                key = row["match_key"]
                if guessed_key in key or key in guessed_key:
                    match = pd.DataFrame([row])
                    break

        if match.empty:
            print(f"⚠ No coordinate match found for: {os.path.basename(file)}")
            continue

        row = match.iloc[0]

        station_id = row[id_col]
        station_name = row[station_col]
        latitude = row[lat_col]
        longitude = row[lon_col]

        # ------------------------------------------
        # Read raw data
        # ------------------------------------------
        df = pd.read_csv(file)

        # Detect required columns dynamically
        timestamp_col = [c for c in df.columns if "timestamp" in c.lower()][0]
        pm25_col = [c for c in df.columns if "pm2.5" in c.lower()][0]

        pm10_candidates = [
            c for c in df.columns
            if "pm10" in c.lower()
        ]

        pm10_col = pm10_candidates[0] if pm10_candidates else None

        no2_col = [c for c in df.columns if "no2" in c.lower()][0]
        ozone_col = [c for c in df.columns if "ozone" in c.lower()][0]

        temp = pd.DataFrame({
            "station_id": station_id,
            "station_name": station_name,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": df[timestamp_col],
            "pm2.5": df[pm25_col],
            "pm10": df[pm10_col] if pm10_col else pd.NA,
            "no2": df[no2_col],
            "ozone": df[ozone_col]
        })

        all_data.append(temp)
        print(f"✓ Processed: {os.path.basename(file)}")

    except Exception as e:
        print(f"✗ Error in {os.path.basename(file)} --> {e}")

# ==========================================================
# MERGE ALL FILES
# ==========================================================
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce")

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n================================================")
    print(f"Done! Final merged file saved at:\n{OUTPUT_FILE}")
    print(f"Total rows: {len(final_df):,}")
    print("Columns:", list(final_df.columns))
    print("================================================")
else:
    print("No data merged.")