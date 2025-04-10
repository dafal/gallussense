import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calplot
import os

# === Streamlit configuration ===
st.set_page_config(page_title="GallusSense 🐓", layout="wide")
DB_PATH = "db/detections.db"

# === Load detections ===
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT timestamp, audio_path, spectrogram_path
        FROM detections
        WHERE review_status IS NULL OR review_status = 1
        ORDER BY timestamp DESC
        """,
        conn,
        parse_dates=['timestamp']
    )
    conn.close()
    return df

# === Auto-refresh every 60 seconds ===
st_autorefresh(interval=60 * 1000, key="gallus_refresh")

# === Load data ===
df = get_stats()

if df.empty:
    st.markdown("<h1 style='text-align: center'>🐓 GallusSense - Détection de Coq en Direct</h1>", unsafe_allow_html=True)
    st.warning("Aucune détection encore 🛌")
else:
    df['date'] = df['timestamp'].dt.date
    df['heure'] = df['timestamp'].dt.hour

    latest_ts = df['timestamp'].max()
    now = datetime.now()
    recent_detection = (now - latest_ts) < timedelta(seconds=10)

    # === Header with two columns (title + image) ===
    col_left, col_right = st.columns([2, 1])

    with col_left:
        if recent_detection:
            st.markdown("""
                <style>
                .pulse {
                    display: inline-block;
                    animation: pulse 0.6s infinite;
                }
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.4); }
                    100% { transform: scale(1); }
                }
                </style>
                <h1 style='text-align: left'>
                    <span class='pulse'>🐓</span> GallusSense - Détection de Coq en Direct
                </h1>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: left'>🐓 GallusSense - Détection de Coq en Direct</h1>", unsafe_allow_html=True)

    with col_right:
        valid_spec = df[df['spectrogram_path'].notna() & df['spectrogram_path'].str.endswith('.png')]
        if not valid_spec.empty:
            last_spec = valid_spec.iloc[0]
            if os.path.exists(last_spec['spectrogram_path']):
                st.markdown("<h4 style='text-align: center;'>Dernière photo 📸</h4>", unsafe_allow_html=True)
                st.image(last_spec['spectrogram_path'], use_container_width=True)
            else:
                st.info("La dernière image n'est pas disponible sur le disque.")

    # === Today's statistics ===
    today = datetime.now().date()
    today_df = df[df['date'] == today]
    if not today_df.empty:
        total_today = len(today_df)
        first_today = today_df['timestamp'].min().strftime("%H:%M")
        last_today = today_df['timestamp'].max().strftime("%H:%M")
        st.markdown(
            f"<h3 style='text-align:center'>📆 {today} — <b>{total_today} cocoricos</b> entre {first_today} et {last_today} 🐔</h3>",
            unsafe_allow_html=True
        )

    # === Hour/day heatmap ===
    pivot_df = df.groupby(['date', 'heure']).size().unstack(fill_value=0)
    pivot_df = pivot_df.sort_index(ascending=False)
    all_hours = pd.Index(range(24), name="heure")
    pivot_df = pivot_df.reindex(columns=all_hours, fill_value=0)
    pivot_df["Total"] = pivot_df.sum(axis=1)

    fig, ax = plt.subplots(figsize=(13, min(0.4 * len(pivot_df), 12)))
    sns.heatmap(
        pivot_df.drop(columns=["Total"]),
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="gray",
        annot=True,
        fmt="d",
        annot_kws={"color": "black"},
        cbar=False,
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.tick_top()
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    st.pyplot(fig)

    # === Yearly calendar ===
    st.subheader("🗓️ Vue annuelle des cocoricos")

    daily_counts = df['timestamp'].dt.date.value_counts().sort_index()
    daily_counts.index = pd.to_datetime(daily_counts.index)

    fig_cal, ax_cal = calplot.calplot(
        daily_counts,
        cmap='YlOrRd',
        colorbar=True,
        figsize=(16, 4),
        linewidth=1,
        yearlabel_kws={'fontname': 'sans-serif'},
    )

    st.pyplot(fig_cal)

    # === 📦 Export + 🔮 Prediction + 🏆 MVP Cocorico (3 columns) ===
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("📦 Export des données")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger les détections (CSV)",
            data=csv,
            file_name="detections_gallussense.csv",
            mime="text/csv",
            help="Clique pour télécharger toutes les détections 🐓"
        )

    with col2:
        st.subheader("🔮 Prédiction pour demain")
        daily_counts = df['timestamp'].dt.date.value_counts().sort_index()
        if len(daily_counts) >= 2:
            recent_days = daily_counts[-7:]
            predicted = int(round(recent_days.mean()))
            st.metric("Prévision", f"{predicted} 🐓")
        else:
            st.info("Pas assez de données pour prédire 😅")

    with col3:
        st.subheader("🏆 Cocorico de la semaine")
        last_week = datetime.now().date() - timedelta(days=7)
        week_df = df[df['date'] >= last_week]
        if not week_df.empty:
            top_day = week_df['date'].value_counts().sort_values(ascending=False).index[0]
            top_count = week_df['date'].value_counts().sort_values(ascending=False).iloc[0]
            jour_nom = top_day.strftime("%A %d %B")
            st.markdown(
                f"<h5 style='text-align: center;'>🥇 <b>{jour_nom}</b><br>{top_count} cocoricos !</h5>",
                unsafe_allow_html=True
            )
        else:
            st.info("Pas assez de cocoricos cette semaine 🛌")
