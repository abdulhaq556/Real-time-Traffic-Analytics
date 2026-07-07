#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import pandas as pd
from datetime import datetime
import time
import os
import random

# ==========================================
# Real-Time Traffic Data Pipeline (ETL)
# ==========================================

### 1. Configuration ###

API_KEY = "OEZCfjkL5k4imwQiQdjLVEWLI981FV1s"
LOCATIONS = [
    {"name": "Ring Road - Maadi", "lat": 29.9544, "lon": 31.2858},
    {"name": "October Bridge", "lat": 30.0526, "lon": 31.2372},
    {"name": "26th July Corridor", "lat": 30.0381, "lon": 31.0264},
    {"name": "Abbas El Akkad", "lat": 30.0631, "lon": 31.3341},
    {"name": "Galaa Square", "lat": 30.0398, "lon": 31.2188}
]


### 2. Extracts data from TomTom API and applies Feature Engineering (Noise & Vehicle Count) ###

def fetch_and_transform_data(locations, api_key):
    new_records = []
    timestamp = datetime.now()
    
    for loc in locations:
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={loc['lat']},{loc['lon']}&key={api_key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()['flowSegmentData']
                real_speed = data['currentSpeed']
                free_flow_speed = data['freeFlowSpeed']
                
                # Add realistic noise to the speed
                current_speed = real_speed + random.randint(-2, 2) if real_speed > 5 else real_speed
                
                # Calculate vehicle count
                if current_speed > 0:
                    vehicle_count = int((free_flow_speed / current_speed) * 20) + random.randint(-3, 3)
                else:
                    vehicle_count = 100
                    
                new_records.append({
                    'timestamp': timestamp,
                    'location_name': loc['name'],
                    'coordinates': f"{loc['lat']}, {loc['lon']}",
                    'speed': current_speed,
                    'vehicle_count': abs(vehicle_count)
                })
        except Exception as e:
            print(f"Error fetching data for {loc['name']}: {e}")
            
    return new_records


### 3. Loads the transformed records into a destination (CSV file) ###

def load_data(records, filename="realtime_egypt_stream.csv"):
    if not records:
        return
        
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Save data to a local CSV file
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully loaded {len(df)} records into {filename}.")


### 4. Main execution loop ###

def main():
    print("Fetching live traffic data ...")
    print("-" * 50)
    
    while True:
        # Step 1 & 2: Extract and Transform
        records = fetch_and_transform_data(LOCATIONS, API_KEY)
        
        # Step 3: Load
        load_data(records)
        
        # Wait before the next API call
        time.sleep(60)

# Entry point of the script
if __name__ == "__main__":
    main()

