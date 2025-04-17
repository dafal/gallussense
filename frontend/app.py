import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calplot
import numpy as np
import os
import yaml
from language_detection import detect_browser_language

st.set_page_config(page_title="GallusSense", page_icon="logo.png", layout="wide")

# === Detect browser locales ===
browser_language = detect_browser_language()
language = browser_language[:2]
if language not in ["fr", "en"]:
    language = "en"

# === Load translation files ===
with open(f"locales/{language}.yaml", encoding="utf-8") as f:
    translations = yaml.safe_load(f)

_ = lambda key, **kwargs: translations.get(key, key).format(**kwargs)

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

st_autorefresh(interval=60 * 1000, key="gallus_refresh")

df = get_stats()

if df.empty:
    st.markdown(f"<h1 style='text-align: center'>🐓 {_('title')}</h1>", unsafe_allow_html=True)
    st.warning(_("no_data"))
else:
    df['date'] = df['timestamp'].dt.date
    df['heure'] = df['timestamp'].dt.hour

    latest_ts = df['timestamp'].max()
    now = datetime.now()
    recent_detection = (now - latest_ts) < timedelta(seconds=10)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        if recent_detection:
            st.markdown(f"""
                <style>
                .pulse {{
                    display: inline-block;
                    animation: pulse 0.6s infinite;
                }}
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.4); }}
                    100% {{ transform: scale(1); }}
                }}
                </style>
                <h1 style='text-align: left'>
                    <span class='pulse'>🐓</span> {_('title')}
                </h1>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='text-align: left'>🐓 {_('title')}</h1>", unsafe_allow_html=True)

    with col_right:
        valid_spec = df[df['spectrogram_path'].notna() & df['spectrogram_path'].str.endswith('.png')]
        if not valid_spec.empty:
            last_spec = valid_spec.iloc[0]
            if os.path.exists(last_spec['spectrogram_path']):
                st.markdown(f"<h4 style='text-align: center;'>{_('latest_image')}</h4>", unsafe_allow_html=True)
                st.image(last_spec['spectrogram_path'], use_container_width=True)
            else:
                st.info(_("image_missing"))

    today = datetime.now().date()
    today_df = df[df['date'] == today]
    if not today_df.empty:
        total_today = len(today_df)
        first_today = today_df['timestamp'].min().strftime("%H:%M")
        last_today = today_df['timestamp'].max().strftime("%H:%M")
        st.markdown(
            f"<h3 style='text-align:center'>📆 {today} — <b>{total_today} cocoricos</b> {_('between')} {first_today} {_('and')} {last_today} 🐔</h3>",
            unsafe_allow_html=True
        )

    pivot_df = df.groupby(['date', 'heure']).size().unstack(fill_value=0)
    pivot_df = pivot_df.sort_index(ascending=False)
    all_hours = pd.Index(range(24), name="heure")
    pivot_df = pivot_df.reindex(columns=all_hours, fill_value=0)
    pivot_df["Total"] = pivot_df.sum(axis=1)

    no_detection_mask = pd.DataFrame(False, index=pivot_df.index, columns=pivot_df.columns.drop("Total"))
    no_detection_file = "db/no_detections.txt"
    if os.path.exists(no_detection_file):
        with open(no_detection_file) as f:
            lines = f.readlines()
        no_detect_entries = [line.strip().split() for line in lines if line.strip()]
        for date_str, hour_str in no_detect_entries:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                hour = int(hour_str)
                if date in no_detection_mask.index and hour in no_detection_mask.columns:
                    no_detection_mask.at[date, hour] = True
            except Exception as e:
                st.warning(_("invalid_line", date_str=date_str, hour_str=hour_str))

    fig, ax = plt.subplots(figsize=(13, min(0.4 * len(pivot_df), 12)))

    from matplotlib.colors import ListedColormap

    base_cmap = sns.color_palette("YlOrRd", as_cmap=True)
    cmap = base_cmap
    cmap.set_bad(color="lightgray")

    data_to_plot = pivot_df.drop(columns=["Total"]).copy().astype(float)

    if os.path.exists(no_detection_file):
        for date_str, hour_str in no_detect_entries:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                hour = int(hour_str)
                if date in data_to_plot.index and hour in data_to_plot.columns:
                    data_to_plot.at[date, hour] = np.nan
            except Exception as e:
                st.warning(_("invalid_line", date_str=date_str, hour_str=hour_str))

    sns.heatmap(
        data_to_plot,
        cmap=cmap,
        linewidths=0.3,
        linecolor="gray",
        annot=pivot_df.drop(columns=["Total"]),
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

    st.subheader(_("calendar_title"))

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

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader(_("export_title"))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=_("download_label"),
            data=csv,
            file_name="detections_gallussense.csv",
            mime="text/csv",
            help=_("download_help")
        )

    with col2:
        st.subheader(_("prediction_title"))
        daily_counts = df['timestamp'].dt.date.value_counts().sort_index()
        if len(daily_counts) >= 2:
            recent_days = daily_counts[-7:]
            predicted = int(round(recent_days.mean()))
            st.metric(_("forecast"), f"{predicted} 🐓")
        else:
            st.info(_("not_enough_data"))

    with col3:
        st.subheader(_("weekly_title"))
        last_week = datetime.now().date() - timedelta(days=7)
        week_df = df[df['date'] >= last_week]
        if not week_df.empty:
            top_day = week_df['date'].value_counts().sort_values(ascending=False).index[0]
            top_count = week_df['date'].value_counts().sort_values(ascending=False).iloc[0]
            jour_nom = top_day.strftime("%A %d %B")
            st.markdown(
                f"<h5 style='text-align: center;'>🥇 <b>{jour_nom}</b><br>{top_count} cocoricos!</h5>",
                unsafe_allow_html=True
            )
        else:
            st.info(_("weekly_empty"))
