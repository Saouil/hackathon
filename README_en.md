
# 🧠 AI Hackathon Project Summary Report (April 12–13, 2025)

## 📌 Project Title:
Smart Meteorological Disaster Prevention System: A Fuzzy Logic-Based Flood Prediction and Q&A Assistant Platform

---

## 📅 Event Duration
- **Start Date**: April 12, 2025
- **End Date**: April 13, 2025

## 🧑‍💻 Team Members
- Leader: Huang Kee Ngong
- Member: Andrew Lim Kim Seng

---

## 🎯 Project Objective
To build an intelligent weather advisory system that can predict flood risks based on rainfall using a Fuzzy Inference System (FIS) and Large Language Model (LLM), while also responding to user questions in real time.

---

## 🛠️ Tech Stack
- **Backend Framework**: FastAPI
- **AI Model**: LLaMA Model (DeepSeek-R1-Distill-Qwen-7B-Q2_K) via `llama_cpp`
- **Fuzzy System**: Takagi-Sugeno-Kang Fuzzy Inference System (TSK-FIS)
- **RAG Module**: Rule-based fuzzy knowledge retrieval
- **Data Analysis**: Pandas, Scikit-learn, Matplotlib
- **Model Training**: KMeans, LinearRegression
- **Deployment**: Uvicorn, VS Code

---

## 📊 Data Sources
- Local weather parameter files (`tavg`, `prcp`, `pres`, etc.)
- Open-Meteo real-time river data API
- Public meteorological dataset `_MalaysiaFloodDataset_`

---

## 🔁 System Module Layers

### ✅ Layer 1: Membership Function Construction
- Gaussian functions are used to define membership functions for each feature (e.g., `prcp`, `avg_river_discharge`)
- Automatic clustering (KMeans) to extract fuzzy centers
- Output includes images + JSON configuration

### ✅ Layer 2: Fuzzy Rule Generation
- Enumerate all fuzzy combinations (e.g., Low-Medium-High)
- Rule structure: `IF x1 is Medium AND x2 is High THEN` (without output initially)

### ✅ Layer 3: TSK Rule Construction
- Train linear regression models using subsets of data covered by each fuzzy rule
- Output linear equations (intercept + coefficients) as the `THEN` clause
- Stored as JSON files for inference

### ✅ Layer 4: Fuzzy Inference (Flood Score)
- Input: rainfall values for a specific location
- Fuzzification ➜ Rule matching ➜ Score calculation using TSK model
- Threshold-based flood classification (e.g., score > 0.7 means flood risk)

### ✅ Layer 5: Deployment & Interaction
- LLM integrates with chat API and responds based on fuzzy score and contextual rules
- Supports bilingual (English/Chinese) mode
- Allows users to input questions via frontend for flood prediction

---

## 🧠 LLM Integration and RAG
- Automatically switches language using `langdetect`
- Calls `retrieve_fuzzy_rules_context()` to extract related fuzzy rules
- Constructs a ChatML format prompt to respond as an expert

---

## 📈 Derived Application Logic
- Automatically compute monthly `avg_river_discharge` for Kuching, Miri, and Sibu
- Predict flood probability using `annual rainfall + prcp + discharge`
- Auto-generate flood classification (`flood = 0 or 1`) ➜ export as CSV

---

## 📁 Final Deliverables
- ✅ Fuzzy membership function plots (per feature)
- ✅ All TSK rules in JSON format
- ✅ Flood prediction CSV (includes risk score)
- ✅ Callable FastAPI chatbot interface (supports chat + inference)

---

## 🚀 Project Highlights
- Custom fuzzy rule system + interpretable inference model
- Local weather data integrated with external APIs
- Human-like responses powered by LLM
- Bilingual support and location-specific predictions

---

## 📌 Next-Step Recommendations
- Integrate real-time API to fetch `prcp` instead of manual input
- Use map interface to select location (auto-fetch coordinates ➜ predict)
- Generate dynamic risk trend charts for users (visualization)
- Add training data to label `flood` for better ground truth accuracy
- Expand to support multi-city predictions and dashboard interface

---

If you'd like to deploy this to a web frontend or publish a demo, I can help generate an interactive UI 🙌
