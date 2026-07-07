# 🚦 Real-Time Traffic Analytics Pipeline for Cairo Smart Cities

An end-to-end Data Engineering project designed to ingest, process, and visualize live traffic telemetry data from major Egyptian road networks. This project leverages real-time APIs and Cloud Infrastructure to provide actionable traffic insights.

## 🏗 Project Architecture & Workflow
The pipeline is designed to handle high-velocity data streaming through the following phases:

1.  **Data Extraction (Current Phase)**: 
    * Consuming live traffic metrics (Speed, Travel Time, Confidence) from the **TomTom Traffic API**.
    * Implementing a custom **Traffic Simulator** to enrich data with realistic noise and vehicle density estimation.
2.  **Ingestion Layer**: 
    * Streaming processed JSON payloads to **Azure Event Hubs** for real-time buffering.
3.  **Stream Processing**: 
    * Utilizing **Azure Stream Analytics** to perform windowed aggregations (e.g., 5-minute average speed per road).
4.  **Storage & Serving**: 
    * Sinking processed data into **Azure SQL Database / Data Lake** for historical analysis.
5.  **Visualization & Alerts**: 
    * Developing an interactive **Power BI Dashboard** featuring:
        * Live Traffic Heatmaps.
        * Anomaly Detection (Accident/Congestion Alerts).

## 📁 Repository Structure
- `traffic_simulator.ipynb`: Development notebook for API prototyping and Feature Engineering.
- `traffic_simulator.py`: Production-ready ETL script optimized for continuous cloud execution.

## 🛠 Tech Stack
- **Language**: Python (Pandas, Requests, JSON).
- **Cloud**: Microsoft Azure (Event Hubs, Stream Analytics,Database SQL).
- **BI Tools**: Power BI.
