from azure.eventhub import EventHubProducerClient, EventData
import json
import time
import requests
from datetime import datetime
import random

# ==========================================
# Real-Time Traffic Data Pipeline (ETL)
# Send Data to Azure Event Hub
# ==========================================

### 1. Configuration ###
from dotenv import load_dotenv, find_dotenv
from os import getenv

load_dotenv(find_dotenv(".env"))

API_KEY = getenv("API_KEY")

# Event Hub configuration
CONNECTION_STR = getenv("CONNECTION_STR")
EVENT_HUB_NAME = getenv("EVENT_HUB_NAME")


producer = EventHubProducerClient.from_connection_string(
    conn_str=CONNECTION_STR,
    eventhub_name=EVENT_HUB_NAME
)

LOCATIONS = [
    {"name": "Ring Road - Maadi", "lat": 29.9544, "lon": 31.2858},
    {"name": "October Bridge", "lat": 30.0526, "lon": 31.2372},
    {"name": "26th July Corridor", "lat": 30.0381, "lon": 31.0264},
    {"name": "Abbas El Akkad", "lat": 30.0631, "lon": 31.3341},
    {"name": "Galaa Square", "lat": 30.0398, "lon": 31.2188},
    {"name": "Mosheer Tantawy Axis", "lat": 30.0242, "lon": 31.3486},
    {"name": "Rod El Farag Axis", "lat": 30.0881, "lon": 31.2185},
    {"name": "Ring Road - Marg", "lat": 30.1554, "lon": 31.3323},
    {"name": "Ring Road - Moneeb", "lat": 29.9881, "lon": 31.2227},
    {"name": "Gamaat El Dowal St", "lat": 30.0543, "lon": 31.2005},
    {"name": "Faisal Street", "lat": 30.0051, "lon": 31.1574},
    {"name": "North 90th Street", "lat": 30.0308, "lon": 31.4721},
    {"name": "South 90th Street", "lat": 30.0195, "lon": 31.4326}
]


### 2. Extract + Transform ###

def fetch_and_transform_data(locations, api_key):

    records = []
    timestamp = datetime.now().isoformat()

    for loc in locations:

        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={loc['lat']},{loc['lon']}&key={api_key}"

        try:

            response = requests.get(url)

            if response.status_code == 200:

                data = response.json()['flowSegmentData']

                real_speed = data['currentSpeed']
                free_flow_speed = data['freeFlowSpeed']

                current_speed = real_speed + random.randint(-2,2) if real_speed > 5 else real_speed

                if current_speed > 0:
                    vehicle_count = int((free_flow_speed/current_speed)*20) + random.randint(-3,3)
                else:
                    vehicle_count = 100

                # Traffic Status
                if current_speed < 20:
                    traffic_status = "Congested"
                elif current_speed < 50:
                    traffic_status = "Moderate"
                else:
                    traffic_status = "Free_Flow"

                record = {
                    "timestamp": timestamp,
                    "location_name": loc['name'],
                    "latitude": loc['lat'],
                    "longitude": loc['lon'],
                    "speed": current_speed,
                    "vehicle_count": abs(vehicle_count),
                    "traffic_status": traffic_status
                }

                records.append(record)

        except Exception as e:

            print(f"Error with {loc['name']} : {e}")

    return records

### 4. Send Data to Event Hub ###

def send_to_eventhub(records, producer):

    if not records:
        return

    try:
        batch = producer.create_batch()
        count = 0

        for record in records:

            event = EventData(json.dumps(record))

            try:
                batch.add(event)
                count += 1

            except ValueError:

                producer.send_batch(batch)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent batch of {count} records")

                batch = producer.create_batch()
                batch.add(event)
                count = 1

        if len(batch) > 0:
            producer.send_batch(batch)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent final batch of {count} records")

    except Exception as e:
        print("Event Hub Error:", e)



def main():

    print("Starting Real-Time Traffic Streaming...")
    print("---------------------------------------")

    while True:

        try:

            # Extract + Transform
            records = fetch_and_transform_data(LOCATIONS, API_KEY)

            # Send to Event Hub
            send_to_eventhub(records, producer)            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle completed - {len(records)} records")

        except Exception as e:
            print("Pipeline Error:", e)

        
        time.sleep(2)



if __name__ == "__main__":
    main()