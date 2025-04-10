import subprocess
import numpy as np
import threading
import librosa
import joblib
import time
import collections
import sqlite3
import os
import fcntl
import logging
from scipy.io.wavfile import write

# === Basic Logging ===
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# === Parameters (env with defaults) ===
rtsp_url = os.getenv("RTSP_URL")
if not rtsp_url:
    logging.error("❌ RTSP_URL is not set. Please define it in the environment or docker-compose.")
    exit(1)

model_path = os.getenv("MODEL_PATH", "./models/gallussense_rf.pkl")
save_samples = os.getenv("SAVE_SAMPLES", "false").lower() == "true"

sr = int(os.getenv("SR", 22050))
window_size = int(os.getenv("WINDOW_SIZE", 5))
hop_interval = int(os.getenv("HOP_INTERVAL", 2))
n_mfcc = int(os.getenv("N_MFCC", 40))
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
ffmpeg_restart_interval = int(os.getenv("FFMPEG_RESTART_INTERVAL", 10))
freeze_timeout = int(os.getenv("FREEZE_TIMEOUT", 30))

last_detection_time = 0
detection_cooldown = window_size

try:
    model = joblib.load(model_path)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    exit(1)

max_samples = sr * (window_size + 1)
audio_buffer = collections.deque(maxlen=max_samples)
stop_event = threading.Event()
restart_event = threading.Event()

def init_db():
    conn = sqlite3.connect("db/detections.db")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            audio_path TEXT,
            spectrogram_path TEXT,
            confidence INTEGER,
            review_status INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("📁 Database initialized.")

def log_detection(audio_path=None, spectrogram_path=None, confidence=None):
    conn = sqlite3.connect("db/detections.db")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    confidence_int = int(confidence * 100) if confidence is not None else None
    conn.execute(
        "INSERT INTO detections (timestamp, audio_path, spectrogram_path, confidence, review_status) VALUES (?, ?, ?, ?, NULL)",
        (now, audio_path, spectrogram_path, confidence_int)
    )
    conn.commit()
    conn.close()
    logging.debug(f"📌 Detection logged at {now} with confidence {confidence_int}% and files.")

def waveform_to_ascii(signal, width=60, color="gray"):
    blocks = "▁▂▃▄▅▆▇█"
    colors = {
        "gray": "\033[90m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m"
    }
    signal = signal[:len(signal) - len(signal) % width]
    chunk = len(signal) // width
    ascii_wave = ""
    for i in range(width):
        val = np.mean(np.abs(signal[i * chunk:(i + 1) * chunk]))
        idx = min(int(val * len(blocks)), len(blocks) - 1)
        ascii_wave += blocks[idx]
    return f"{colors.get(color, '')}{ascii_wave}{colors['reset']}"

def save_rooster_sample(signal, confidence):
    from datetime import datetime
    import os
    from scipy.io.wavfile import write as wav_write
    import matplotlib.pyplot as plt
    import librosa.display

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y-%m-%dT%H-%M-%S")
    confidence_str = f"{int(confidence * 100):03d}"

    dir_path = os.path.join("records", date_str)
    os.makedirs(dir_path, exist_ok=True)

    base_filename = f"{timestamp_str}_{confidence_str}"
    wav_path = os.path.join(dir_path, f"{base_filename}.wav")
    png_path = os.path.join(dir_path, f"{base_filename}.png")

    wav_signal = np.int16(signal * 32767)
    wav_write(wav_path, sr, wav_signal)

    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"GallusSense Spectrogram - {timestamp_str} - ({confidence * 100:.1f} %)")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    logging.debug(f"📂 Sample saved: {wav_path}")
    logging.debug(f"🖼️ Spectrogram saved: {png_path}")

    return wav_path, png_path

def capture_audio():
    last_audio_time = time.time()

    def start_ffmpeg():
        logging.info("🎮 Starting ffmpeg...")
        process = subprocess.Popen([
            "ffmpeg", "-i", rtsp_url,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(sr), "-vn", "-"
        ], stdout=subprocess.PIPE, stderr=open("ffmpeg_errors.log", "a"))

        fd = process.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        return process

    process = start_ffmpeg()

    try:
        while not stop_event.is_set():
            if restart_event.is_set():
                logging.warning("🔁 Forcing ffmpeg restart due to detected freeze.")
                restart_event.clear()
                process.terminate()
                process.wait()
                process = start_ffmpeg()
                last_audio_time = time.time()

            if process.poll() is not None:
                logging.warning("⚠️ ffmpeg died. Restarting...")
                time.sleep(2)
                process = start_ffmpeg()
                last_audio_time = time.time()

            try:
                raw_audio = process.stdout.read(4096)
            except BlockingIOError:
                raw_audio = None

            if raw_audio:
                last_audio_time = time.time()

                if len(raw_audio) % 2 != 0:
                    raw_audio = raw_audio[:-1]

                if len(raw_audio) >= 2:
                    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_buffer.extend(samples)

            time.sleep(0.1)

            if time.time() - last_audio_time > ffmpeg_restart_interval:
                logging.warning(f"⏱️ No audio for {ffmpeg_restart_interval}s. Restarting...")
                process.terminate()
                process.wait()
                process = start_ffmpeg()
                last_audio_time = time.time()

    finally:
        logging.info("🛌 ffmpeg stopped.")
        process.terminate()

def analyze_audio():
    global last_detection_time
    logging.info(f"🎧 Detection running (rooster only, confidence ≥ {int(confidence_threshold * 100)}%)...")
    previous_mfcc = None
    last_unique_mfcc_time = time.time()
    label_names = {1: "rooster", 0: "non_rooster"}

    pending_detections = []

    while not stop_event.is_set():
        if len(audio_buffer) >= sr * window_size:
            window = list(audio_buffer)[-sr * window_size:]
            signal = np.array(window)
            signal = librosa.util.normalize(signal)

            rms_energy = np.sqrt(np.mean(signal ** 2))
            logging.debug(f"🔊 RMS Energy: {rms_energy:.4f}")
            if rms_energy < 0.01:
                logging.warning("📉 Signal too weak, skipped.")
                ascii_wave = waveform_to_ascii(signal, color="gray")
                logging.info(f"Prediction: silent            (0.00) | {ascii_wave}")
                time.sleep(hop_interval)
                continue

            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            if previous_mfcc is not None:
                dist = np.linalg.norm(mfcc_mean - previous_mfcc)
                if dist < 0.01:
                    if time.time() - last_unique_mfcc_time > freeze_timeout:
                        restart_event.set()
                        last_unique_mfcc_time = time.time()
                    time.sleep(hop_interval)
                    continue
                else:
                    last_unique_mfcc_time = time.time()
            previous_mfcc = mfcc_mean

            prediction = model.predict([mfcc_mean])[0]
            proba = model.predict_proba([mfcc_mean])[0]
            conf = np.max(proba)
            label = label_names[prediction]

            audio_path, spec_path = (None, None)
            if prediction == 1 and save_samples:
                audio_path, spec_path = save_rooster_sample(signal, conf)

            if prediction == 1:
                color = "red" if conf > 0.9 else "yellow"
            else:
                color = "gray"

            ascii_wave = waveform_to_ascii(signal, color=color)
            current_time = time.time()

            label_padded = f"{label:<17}"
            msg = f"Prediction: {label_padded} ({conf:.2f}) | {ascii_wave}"

            # Handle consecutive detections intelligently
            if prediction == 1 and conf >= confidence_threshold:
                # Add new detection to buffer
                pending_detections.append({
                    "time": current_time,
                    "conf": conf,
                    "audio_path": audio_path,
                    "spec_path": spec_path
                })

                # If we have more than 2 pending, flush the best one
                if len(pending_detections) >= 3:
                    best = max(pending_detections, key=lambda d: d["conf"])
                    log_detection(best["audio_path"], best["spec_path"], best["conf"])
                    last_detection_time = best["time"]
                    pending_detections.clear()
                    msg += " ✅ Best of 3 logged"
                else:
                    msg += f" 🐓 COCORICOOOO !!! ⏳ Waiting for next detection ({len(pending_detections)}/2)"

            # Force logging if no new detection after X seconds
            if pending_detections:
                oldest = pending_detections[0]["time"]
                if current_time - oldest > (detection_cooldown + hop_interval):
                    best = max(pending_detections, key=lambda d: d["conf"])
                    log_detection(best["audio_path"], best["spec_path"], best["conf"])
                    last_detection_time = best["time"]
                    pending_detections.clear()
                    msg += " 🕒 Timeout reached, ✅ best logged"

            logging.info(msg)

        time.sleep(hop_interval)


if __name__ == "__main__":
    init_db()
    threading.Thread(target=capture_audio, daemon=True).start()
    threading.Thread(target=analyze_audio, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("🛌 Interrupt received. Shutting down...")
        stop_event.set()
        time.sleep(2)
        logging.info("✅ GallusSense cleanly stopped.")
