[![Built with Vibe Coding](https://img.shields.io/badge/built%20with-vibe%20coding-ff69b4)](https://github.com/dafal/gallussense)

> 🧪✨ This is a vibe-coded project: built with fun, curiosity, backyard audio chaos, and a lot of rooster spirit.

# 🐔 GallusSense

**GallusSense** is a real-time rooster call detection system built to work with audio streams—especially RTSP streams from IP cameras. It uses audio signal processing, MFCC features, and a trained machine learning model to detect rooster calls ("cocorico!").

The project consists of two main components:

- **Backend** – Captures and analyzes audio in real time using `ffmpeg` and `librosa`, then uses a machine learning model to detect rooster calls. Detections are logged into a local SQLite database.
  
- **Frontend** – A playful and interactive Streamlit app that displays detection events in a fun and visual way. It provides real-time feedback, sound wave previews, stats, and animation when a rooster is detected.

## 🚀 Features

- 📡 **Real-time audio capture** from any RTSP stream using `ffmpeg`
- 🎧 **MFCC-based audio analysis** powered by `librosa` for robust feature extraction
- 🤖 **Rooster call detection** using a pre-trained `RandomForestClassifier` model
- ✅ **Confidence-based logging** – only detections above a configurable threshold are stored
- 🗃️ **Detection history** stored in a local SQLite database with timestamps
- 🔊 **Optional audio clip saving** – automatically stores audio segments when a rooster is detected
- 📸 **Optional spectrogram export** for visual inspection or data science fun
- 🔁 **Automatic `ffmpeg` recovery** – restarts the stream if audio freezes or drops
- 🎛️ **ASCII waveform previews** right in the console, just for the nerdy joy of it
- 📊 **Fun and visual Streamlit dashboard** – real-time stats by hour, day, and more quirky analytics to enjoy your rooster data in style

## 🧠 Model Details

GallusSense uses a lightweight yet effective machine learning approach to detect rooster calls based on audio signal characteristics.

### 🔍 Training Pipeline

The default model was trained using a combination of:
- 📹 **Rooster audio samples** recorded directly from the author's backyard
- 🎧 **Non-rooster sounds** extracted from the ESC-50 dataset (environmental recordings)

The training workflow includes:
1. **Audio preprocessing** with `librosa`:
   - Resampling to 22,050 Hz
   - Normalization
   - MFCC extraction (40 coefficients)
2. **Feature aggregation** by averaging over the time axis
3. **Binary classification** using a `RandomForestClassifier` with class balancing

### ⚖️ Performance & Generalization

> The model performs very well in its original environment, but its generalization to other contexts (regions, audio devices) hasn't been thoroughly tested.

- ✅ Excellent detection accuracy in the trained backyard setup
- ⚠️ Occasional false positives from ambulance sirens and motorcycles due to similar acoustic patterns

For better accuracy in other settings, retraining with local audio samples is recommended.

### 🧪 Train Your Own Model

A ready-to-use **Jupyter notebook (`train_model.ipynb`)** is included in the repository to help you:
- Load your own rooster and non-rooster samples
- Extract MFCC features
- Train and export a new `.pkl` model

This makes it easy to adapt GallusSense to your own environment in just a few steps!

Très bien ! On peut créer une **section claire et pratique** pour le déploiement avec Docker, incluant :

1. 🔧 Liste des variables d’environnement nécessaires (extraites de ton backend)
2. 🐳 Commandes simples pour lancer les conteneurs
3. ⚙️ Un exemple `docker-compose.yml`
4. 🚀 Bonus : quelques idées pour aller plus loin si tu veux améliorer le setup


## 🐳 Running with Docker

GallusSense is designed to run easily using Docker containers for both backend processing and the frontend dashboard.

### 🧩 Docker Images

Prebuilt images are available on Docker Hub:

- **Backend**: [`dafal/gallussense-backend`](https://hub.docker.com/r/dafal/gallussense-backend)
- **Frontend (Streamlit app)**: [`dafal/gallussense-frontend`](https://hub.docker.com/r/dafal/gallussense-frontend)

---

### ⚙️ Environment Variables (Backend)

Make sure to define the following variables to configure your detection system:

| Variable                  | Description                                      | Default               |
|--------------------------|--------------------------------------------------|-----------------------|
| `RTSP_URL`               | RTSP stream URL with audio                       | **Required**          |
| `MODEL_PATH`             | Path to the `.pkl` model inside the container    | `./models/gallussense_rf.pkl` |
| `SR`                     | Sample rate (Hz)                                 | `22050`               |
| `WINDOW_SIZE`            | Audio window size in seconds                     | `5`                   |
| `HOP_INTERVAL`           | Time (sec) between detection windows             | `2`                   |
| `CONFIDENCE_THRESHOLD`   | Confidence threshold for logging a detection     | `0.75`                |
| `FFMPEG_RESTART_INTERVAL`| Timeout (sec) before restarting ffmpeg           | `10`                  |
| `FREEZE_TIMEOUT`         | Trigger restart if features are frozen           | `30`                  |
| `SAVE_SAMPLES`           | Generate and save audio and spectrogramm         | `false`               |
---

### ▶️ Quick Start with Docker Compose

Create a `docker-compose.yml` in your project root or use the one provided in the repo:

```yaml
version: '3.9'

services:
  backend:
    image: dafal/gallussense-backend
    environment:
      RTSP_URL: rtsp://your-camera-stream
      MODEL_PATH: ./models/gallussense_rf.pkl
      CONFIDENCE_THRESHOLD: 0.6
      SAVE_SAMPLES: true
    volumes:
      - ./models:/app/models
      - ./db:/app/db
      - ./records:/app/records
    restart: always

  frontend:
    image: dafal/gallussense-frontend
    ports:
      - "8501:8501"
    volumes:
      - ./db:/app/db
      - ./records:/app/records
    restart: always
```

Then start everything with:

```bash
docker-compose up -d
```

Visit `http://localhost:8501` to access the Streamlit dashboard 🎉
