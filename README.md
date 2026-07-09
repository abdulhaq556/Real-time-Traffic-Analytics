# 🚦 Smart Traffic Egypt: Real-Time Analytics & ML Pipeline

[![Presentation]([https://img.shields.io/badge/Presentation-View_Slides-blue?style=for-the-badge&logo=canva](https://canva.link/t7jfgiytlm331u9))](#)


An end-to-end Data Engineering and Machine Learning ecosystem designed to ingest, process, predict, and visualize live traffic telemetry data from major Egyptian road networks. 

---

## 📖 Table of Contents
1. [The Problem & Our Solution](#-the-problem--our-solution)
2. [System Architecture](#-system-architecture)
3. [Phase 1: Data Engineering & Cloud Infrastructure](#-phase-1-data-engineering--cloud-infrastructure)
4. [Phase 2: Machine Learning Engine](#-phase-2-machine-learning-engine)
5. [Phase 3: Business Intelligence (Power BI)](#-phase-3-business-intelligence-power-bi)
6. [Phase 4: User Application (Streamlit)](#-phase-4-user-application-streamlit)
7. [Getting Started & Installation](#-getting-started--installation)

---

## 🎯 The Problem & Our Solution

**The Challenge:** 
Traffic congestion in metropolitan areas like Cairo fluctuates rapidly. Traditional routing apps tell you the current state, but they lack predictive foresight and deep analytical context for city planners.

**The Solution:**
We built a dual-purpose intelligence system:
* **For Decision Makers:** A real-time Power BI dashboard tracking live bottlenecks, average speeds, and congestion distributions across 17 major roads.
* **For Commuters:** An interactive Streamlit application powered by an XGBoost model that not only predicts future traffic states but also plans the optimal departure time within a 24-hour window.

---

## 🏗 System Architecture

Our pipeline is designed for high velocity and fault tolerance.

> **[Insert Architecture Diagram Here]**

1. **Extraction:** A custom Python ETL script continuously pulls live telemetry (Speed, Travel Time, Congestion level) from the **TomTom Traffic API**.
2. **Streaming:** Data is pushed as JSON payloads into **Azure Event Hubs**.
3. **Processing:** **Azure Stream Analytics** consumes the stream, performs windowed aggregations, and filters anomalies.
4. **Storage:** Cleaned data is sinked into an **Azure SQL Database** for historical training and live querying.
5. **Serving:** The Machine Learning model and UI layers fetch this data for real-time inference and visualization.

---

## ⚙️ Phase 1: Data Engineering & Cloud Infrastructure

The backbone of this project is a robust data pipeline built on Microsoft Azure:
* **Live Ingestion:** Handling continuous API requests without throttling.
* **Geospatial Mapping:** Mapping raw coordinates to 17 specific junctions in Cairo (e.g., Ring Road, 26th July Corridor).
* **Fault Tolerance (Auto-Fallback):** We engineered the frontend to be highly reliable. If the connection to Azure SQL drops, the system automatically falls back to parsing a local `CSV` replica, ensuring zero downtime for the end-user.

---

## 🧠 Phase 2: Machine Learning Engine

To move from monitoring to prediction, we developed a classification model:
* **Algorithm:** **XGBoost Classifier** (Chosen for its high performance with tabular data and speed during inference).
* **Feature Engineering:** We extracted temporal features (`hour`, `minute`, `day_of_week`, `is_weekend`, `is_peak_hour`) and encoded categorical locations.
* **Handling Imbalance:** Traffic data is heavily skewed towards "Free Flow". We applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset, allowing the model to accurately detect rare "Congested" states.
* **Output:** The model predicts three states: `Free_Flow`, `Moderate`, and `Congested`, along with a **Confidence Score (%)** for transparency.

---

## 📊 Phase 3: Business Intelligence (Power BI)

Designed for city officials and traffic management centers.
* **Live Monitoring:** Real-time KPI cards showing Total Records, Active Locations, and Average Network Speed.
* **Geospatial Heatmaps:** Visualizing current congestion nodes on a live map of Cairo.
* **Alerts:** Automated tracking of severe congestion events.

> **[Insert Power BI Dashboard Screenshot Here]**

---

## 💻 Phase 4: User Application (Streamlit)

A lightweight, interactive web application acting as the frontend for our ML model.

### 1. Exploratory Data Analysis (EDA)
Users can explore historical patterns through interactive visual plots:
* Congestion Heatmaps (Hour vs. Day of Week).
* Speed Distributions and Feature Correlation Matrices.

### 2. Smart Prediction & Trip Planner
* **Confidence Breakdown:** Instead of a simple output, the app displays the exact probability for each traffic state.
* **24-Hour Optimization:** The user selects a route, and the system simulates predictions for the next 24 hours, plotting a graph to highlight the **Best (Safest)** and **Worst** times to travel.
* **Multi-Route Comparison:** Users input two alternative routes. The system evaluates both concurrently and flags the `FASTEST` route based on the highest model confidence.

> **[Insert Streamlit Prediction & 24-Hour Graph Screenshots Here]**

---

## 🚀 Getting Started & Installation

To run the Streamlit application and ETL simulator locally:

### Prerequisites
* Python 3.9+
* Azure SQL Server Credentials (or use the offline CSV mode)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Egypt-Traffic-Intelligence.git](https://github.com/YourUsername/Egypt-Traffic-Intelligence.git)
   cd Egypt-Traffic-Intelligence
