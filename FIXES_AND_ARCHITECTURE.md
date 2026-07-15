# 🌤️ weatherBOT v2.0 — Architecture & Technical Reference

`weatherBOT` is a restricted, offline AI Weather Intelligence Platform engineered to run on isolated Edge hardware (including Raspberry Pi). It synthesizes Machine Learning metrics (XGBoost), Spatial Graph Neural Network predictions (PyTorch Geometric), localized Large Language Model inference (Ollama `llama3.1:8b`), real-time IoT sensor telemetry, and a fully standard MCP (Model Context Protocol) server.

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER  (Browser)                              │
│        React + TypeScript Frontend  (port 5173)                 │
│  ┌────────────┬──────────────┬─────────────┬────────────────┐   │
│  │ Chat Module│ Raw Dataset  │  Conv. Hist.│    Settings    │   │
│  │ (NLP+XAI) │   Module     │  (Search)   │ (IoT + MCP)    │   │
│  └────────────┴──────────────┴─────────────┴────────────────┘   │
└───────────────────────┬─────────────────────────────────────────┘
                        │  HTTP REST / Axios
┌───────────────────────▼─────────────────────────────────────────┐
│                FastAPI Backend  (port 8000)                      │
│                                                                  │
│  /api/auth   /api/datasets   /api/chat   /api/predictions        │
│                              /api/mcp                            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  MCP Service Router │  ← Central Dispatcher
              │  mcp_router.py     │    (10 services / 32 tools)
              └──────┬─────────────┘
         ┌───────────┼──────────────────────────┐
         ▼           ▼           ▼              ▼
   ┌──────────┐ ┌─────────┐ ┌────────┐  ┌───────────┐
   │Conversation│ │  CSV   │ │ Graph  │  │    GNN    │
   │ Service  │ │Service  │ │Service │  │  Service  │
   │ (Ollama) │ │(Plotly) │ │(NetworkX│ │(PyTorch)  │
   └──────────┘ └─────────┘ └────────┘  └───────────┘
         ▼           ▼           ▼              ▼
   ┌──────────┐ ┌─────────┐ ┌────────┐  ┌───────────┐
   │ History  │ │   IoT   │ │  NLG   │  │ Dataset   │
   │ Service  │ │ Service │ │Service │  │  Manager  │
   │(SQLite)  │ │(RPi/MQTT│ │(NL Gen)│  │           │
   └──────────┘ └─────────┘ └────────┘  └───────────┘
         ▼
   ┌──────────┬──────────────┬───────────────┐
   │  Model   │  Research    │   NLP Engine  │
   │ Manager  │  Engine      │ (Intent+Entity│
   │(Registry)│(Stats+Corr.) │  Extraction)  │
   └──────────┴──────────────┴───────────────┘
         ▼
   ┌──────────────────────────────────────────┐
   │         AI Response Layer                │
   │  XAI Engine → Recommendation Engine →   │
   │  NLG Service → Chat Response            │
   └──────────────────────────────────────────┘
         ▼
   ┌─────────────────────────────────────────┐
   │  SQLite DB  +  MCP SDK Server           │
   │  (weatherbot.db)   (mcp_server.py)     │
   └─────────────────────────────────────────┘
         ▲
   ┌─────┴──────────────────────────────────┐
   │  Raspberry Pi Weather Station (IoT)    │
   │  Temperature · Humidity · Pressure     │
   │  Wind Speed · Rainfall · Light         │
   │  ── USB Serial (pyserial) ──────────── │
   │  ── MQTT Wireless (paho-mqtt) ─────── │
   └────────────────────────────────────────┘
```

---

## 🗂️ Full Directory Structure

```
weatherBOT/
├── backend/
│   ├── main.py                        # FastAPI app + router registration
│   ├── database.py                    # SQLAlchemy engine + session
│   ├── models.py                      # ORM: Dataset, ConversationSession, Message, ModelVersion
│   ├── schemas.py                     # Pydantic request/response schemas
│   ├── mcp_server.py                  # ★ Standard MCP SDK server (stdio/SSE)
│   ├── requirements.txt
│   │
│   ├── routers/
│   │   ├── auth.py                    # JWT login/register
│   │   ├── datasets.py                # Upload, list, delete
│   │   ├── chat.py                    # Chat sessions + XAI + search
│   │   ├── predictions.py             # ★ Live GNN, Historical XGBoost, XAI, Recs
│   │   └── mcp.py                     # ★ MCP route dispatcher + IoT connect
│   │
│   ├── services/
│   │   ├── conversation_service.py    # Ollama Tri-Mode inference
│   │   ├── dataset_service.py         # CSV parsing, sensor alias detection
│   │   ├── ml_service.py              # XGBoost training + self-learning loop
│   │   ├── mcp_router.py              # ★ MCP Service Router (central dispatcher)
│   │   ├── iot_service.py             # ★ Raspberry Pi serial/MQTT/simulator
│   │   ├── gnn_service.py             # ★ 3-layer GCN spatial prediction
│   │   ├── csv_service.py             # ★ Data cleaning + Plotly chart generators
│   │   ├── graph_service.py           # ★ Knowledge graph visualizations
│   │   ├── history_service.py         # ★ Search, stats, auto-title, delete
│   │   ├── nlg_service.py             # ★ Natural Language Generator
│   │   └── decision_engine.py         # Legacy heuristic intent router
│   │
│   ├── engines/
│   │   ├── nlp_engine.py              # ★ Full intent classification + entity extraction
│   │   ├── explainable_ai_engine.py   # ★ SHAP-style XAI + attention maps
│   │   └── recommendation_engine.py   # ★ Context-aware follow-up suggestions
│   │
│   ├── managers/
│   │   ├── dataset_manager.py         # ★ Full dataset lifecycle manager
│   │   ├── model_manager.py           # ★ Model registry: promote/demote/compare
│   │   └── research_engine.py         # ★ Statistics, correlation, outliers, trends
│   │
│   └── utils/
│       └── logger.py
│
└── frontend/
    └── src/
        ├── api.ts                     # All API calls (datasets, chat, predictions, MCP, IoT)
        └── components/
            ├── ChatWindow.tsx         # ★ Chat + recommendation chips + XAI panel
            ├── RawDataset.tsx         # Upload, list, delete, training status
            ├── History.tsx            # Conversation sessions
            ├── Settings.tsx           # ★ IoT controls + MCP viewer + system mode
            ├── Sidebar.tsx
            └── Login.tsx
```
> ★ = New in v2.0

---

## 🔄 Request Lifecycle (Detailed)

```
User types message
       │
       ▼
[Frontend: ChatWindow.tsx]
  sendMessage() → POST /api/chat/sessions/{id}/message
       │
       ▼
[Backend: routers/chat.py]
  HistoryService.auto_title_session()
  ConversationService.generate_response()   ← Ollama inference
  NLPEngine.process()                        ← Intent + entities
  RecommendationEngine.get_recommendations() ← Follow-up chips
       │
       ▼
[ConversationService — Tri-Mode Logic]
  ┌──────────────────────────────────────┐
  │ Historical Data Mode                 │
  │   → Fetch active ModelVersion (DB)  │
  │   → Build ML context string         │
  ├──────────────────────────────────────┤
  │ Live Station Mode                    │
  │   → IoTService.get_reading()         │
  │   → GNNService.predict()             │
  │   → Build IoT+GNN context string    │
  ├──────────────────────────────────────┤
  │ Prime (Default)                      │
  │   → Both contexts merged             │
  └──────────────────────────────────────┘
       │
       ▼
  Ollama API (llama3.1:8b, port 11434)
  → Returns natural text + optional JSON Plotly config
       │
       ▼
[Response enrichment]
  { content, graphs, mode,
    recommendations,    ← RecommendationEngine
    intent, entities }  ← NLPEngine
       │
       ▼
[Frontend renders]
  • Text response
  • Plotly charts (inline)
  • Recommendation chips (clickable)
  • XAI panel (collapsible, on prediction intents)
```

---

## 🔵 MCP Service Router — 32 Tools Across 10 Services

| Service | Actions |
|---|---|
| `iot` | `reading` · `multi_node` · `status` · `connect_serial` · `connect_mqtt` · `configure_simulator` |
| `gnn` | `predict_live` · `predict` |
| `csv` | `profile` · `trend` · `correlation_heatmap` · `distribution` |
| `graph` | `feature_graph` · `spatial_graph` |
| `history` | `stats` · `search` · `list_sessions` |
| `nlp` | `process` · `classify` · `entities` |
| `xai` | `explain` · `attention_map` · `actual_vs_predicted` |
| `research` | `full_report` · `correlation` · `outliers` · `trend` |
| `model` | `list` · `summary` · `promote` |
| `dataset` | `list` · `active` |

**Use via REST:** `POST /api/mcp/route` with `{"service":"gnn","action":"predict_live"}`

**Use via MCP SDK:** `python backend/mcp_server.py` → connects to any MCP client (Claude Desktop, custom agents)

---

## 🌡️ IoT / Raspberry Pi Integration

```
Raspberry Pi                        weatherBOT Backend
─────────────                       ─────────────────────
Sensor Board                        iot_service.py
  │                                   │
  ├── USB Serial (UART) ────────────► SerialWeatherReader
  │   /dev/ttyUSB0 or COM3            pyserial thread
  │   JSON lines: {"temp": 23.1}      updates _sensor_state{}
  │
  └── Wi-Fi MQTT ──────────────────► MQTTWeatherReader
      topic: weatherbot/station/      paho-mqtt subscription
             sensors                  updates _sensor_state{}

If no hardware: WeatherSimulator generates realistic
  sinusoidal diurnal cycles + configurable Gaussian noise
```

---

## 🧠 ML Pipeline — Self-Learning Loop

```
CSV Upload
    │
    ▼
DatasetService          ← Sensor alias matching, time-span detection
    │
    ├── Save metadata → SQLite (status: PROCESSING)
    │
    └── Background Task: run_self_learning()
              │
              ▼
          MLService.evaluate_and_promote()
              │
              ├── Clean NaN/Inf values
              ├── Intelligent target column matching
              ├── Train XGBoost (CUDA auto-detect)
              ├── Compute RMSE / R² / MAE
              ├── Extract feature importances (XAI)
              ├── Downsample 50 actual vs predicted points
              │
              └── Compare vs active model RMSE
                      ├── Better → Promote, demote old
                      └── Worse  → Save as inactive
              │
              └── Update DB status → COMPLETED or FAILED
```

---

## 🤖 AI Response Layer

```
Raw Structured Data
        │
        ▼
┌──────────────────────┐
│    XAI Engine        │  Feature importance → SHAP-style ranked explanations
│                      │  Attention map (Plotly bar chart)
│                      │  Actual vs Predicted scatter chart
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Recommendation Engine│  Intent + Mode → 3 contextual follow-up queries
│                      │  Data-driven: based on has_dataset/model/IoT
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    NLG Service       │  Converts structured dicts → natural language
│                      │  predict(), describe_iot_reading(), describe_metrics()
└──────────┬───────────┘
           │
           ▼
      Chat Response
```

---

## 🔑 NLP Engine — Intent Classification

8 intent types classified via keyword maps (zero external calls, Edge-optimized):

| Intent | Triggers |
|---|---|
| `GREETING` | hi, hello, hey, good morning... |
| `GRAPH_REQUEST` | plot, chart, visualize, heatmap... |
| `PREDICTION_REQUEST` | predict, forecast, tomorrow, estimate... |
| `IOT_STATUS` | live, sensor, station, raspberry, real-time... |
| `XAI_REQUEST` | why, explain, shap, feature importance... |
| `DATA_QUERY` | temperature, humidity, rmse, dataset... |
| `HISTORY_QUERY` | history, previous, last session... |
| `GENERAL_CHAT` | fallback |

Entity extraction: **sensors**, **time ranges**, **metrics** (RMSE, R², MAE), **locations** (for refusal)

---

## 🛢️ Database Schema

```
datasets
  id, filename, total_rows, total_columns, time_span,
  sampling_frequency, missing_values, duplicate_values,
  detected_sensors (JSON), data_quality_score,
  status (PROCESSING|COMPLETED|FAILED), error_message, upload_date

model_versions
  id, version, is_active, dataset_id, training_time,
  accuracy (R²), rmse, plot_data (JSON), feature_importances (JSON), created_at

conversation_sessions
  id (UUID), title, created_at

messages
  id, session_id, role (user|assistant), content,
  timestamp, graphs (JSON Plotly config)
```

---

## 🚀 How to Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Access: `http://localhost:5173` · Login: `admin` / `admin`

### MCP Server (optional — for Claude Desktop / MCP clients)
```bash
cd backend
python mcp_server.py
```

### Ollama (required for chat)
```bash
ollama run llama3.1:8b
```

### Raspberry Pi Setup
- **Serial**: Connect Pi via USB, set port `COM3` or `/dev/ttyUSB0` in Settings
- **MQTT**: Run `mosquitto` on Pi, enter Pi IP in Settings
- Pi sends JSON: `{"temperature": 23.1, "humidity": 61.2, "pressure": 1012.8, "wind_speed": 4.3, "rainfall": 0.0, "light_intensity": 720.0}`
