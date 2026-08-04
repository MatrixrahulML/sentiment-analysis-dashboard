"""
Sentiment Analysis Dashboard
Rahul Sah | maatrixxrahul

Real-time brand sentiment tracker using VADER sentiment analysis on live
news headlines (Google News RSS), with trend charts and threshold alerts.
"""

import streamlit as st
import numpy as np
import pandas as pd
import feedparser
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from urllib.parse import quote

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtext { color: #9ca3af; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-header">📊 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtext">Live brand/topic sentiment tracking from news headlines, '
    'with trend charts and alert thresholds.</div>',
    unsafe_allow_html=True,
)
st.info(
    "ℹ️ **Data source note:** X (Twitter)'s API now requires a paid tier, so this dashboard "
    "tracks live sentiment from **Google News RSS** headlines instead — a free, public, real-time "
    "source. The sentiment scoring, trend analysis, and alerting logic are identical to what a "
    "production Twitter-based tracker would run."
)
st.markdown("---")

analyzer = SentimentIntensityAnalyzer()

st.sidebar.header("⚙️ Settings")
brand = st.sidebar.text_input("Brand / topic to track", value="Tesla")
alert_threshold = st.sidebar.slider("Negative sentiment alert threshold", -1.0, 0.0, -0.2, 0.05)
fetch = st.sidebar.button("🔄 Fetch & Analyze Live Headlines", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline:**\n"
    "1. Fetch live headlines from Google News RSS for the given brand/topic\n"
    "2. Score each headline with VADER sentiment analysis\n"
    "3. Aggregate into trend charts and distribution views\n"
    "4. Flag headlines below the alert threshold\n\n"
    "VADER is a rule-based sentiment tool tuned for short, informal text — "
    "well suited to headlines and social posts."
)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_headlines(topic: str, limit: int = 40):
    url = f"https://news.google.com/rss/search?q={quote(topic)}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:limit]:
        published = entry.get("published_parsed")
        ts = datetime(*published[:6]) if published else datetime.now()
        source_info = entry.get("source")
        source_name = source_info.get("title", "Unknown") if isinstance(source_info, dict) else "Unknown"
        rows.append({
            "headline": entry.get("title", ""),
            "source": source_name,
            "published": ts,
            "link": entry.get("link", ""),
        })
    return pd.DataFrame(rows)


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores = df["headline"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    df["sentiment_score"] = scores
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"], bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["Negative", "Neutral", "Positive"],
    )
    return df


if "sa_data" not in st.session_state:
    st.session_state.sa_data = None

if fetch:
    with st.spinner(f"Fetching live headlines about '{brand}'..."):
        try:
            raw = fetch_headlines(brand)
        except Exception as e:
            st.error(f"Couldn't fetch live headlines right now: {e}")
            st.stop()

    if raw.empty:
        st.warning(f"No headlines found for '{brand}'. Try a broader or different search term.")
    else:
        st.session_state.sa_data = score_sentiment(raw)

if st.session_state.sa_data is not None:
    df = st.session_state.sa_data

    avg_sentiment = df["sentiment_score"].mean()
    pct_negative = (df["sentiment_label"] == "Negative").mean() * 100
    pct_positive = (df["sentiment_label"] == "Positive").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headlines analyzed", len(df))
    c2.metric("Avg. sentiment", f"{avg_sentiment:+.3f}")
    c3.metric("% Positive", f"{pct_positive:.0f}%")
    c4.metric("% Negative", f"{pct_negative:.0f}%")

    if avg_sentiment < alert_threshold:
        st.error(f"🚨 **ALERT:** Average sentiment ({avg_sentiment:+.3f}) is below your threshold ({alert_threshold:+.2f}) — negative coverage spike detected for '{brand}'.")

    tab1, tab2, tab3 = st.tabs(["📈 Trend", "📊 Distribution", "📰 Headlines"])

    with tab1:
        df_sorted = df.sort_values("published")
        fig_trend = px.scatter(
            df_sorted, x="published", y="sentiment_score", color="sentiment_label",
            color_discrete_map={"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"},
            title=f"Sentiment Over Time — '{brand}'", template="plotly_dark",
            hover_data=["headline"],
        )
        fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_trend.add_hline(y=alert_threshold, line_dash="dot", line_color="#ef4444",
                             annotation_text="Alert threshold")
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            counts = df["sentiment_label"].value_counts().reset_index()
            counts.columns = ["sentiment", "count"]
            fig_pie = px.pie(
                counts, names="sentiment", values="count",
                color="sentiment",
                color_discrete_map={"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"},
                title="Sentiment Distribution", template="plotly_dark",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            fig_hist = px.histogram(
                df, x="sentiment_score", nbins=20, title="Sentiment Score Histogram",
                template="plotly_dark", color_discrete_sequence=["#8b5cf6"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        source_summary = df.groupby("source", observed=True).agg(
            headlines=("headline", "count"), avg_sentiment=("sentiment_score", "mean")
        ).reset_index().sort_values("headlines", ascending=False).head(10)
        if not source_summary.empty:
            fig_src = px.bar(
                source_summary, x="source", y="avg_sentiment", color="avg_sentiment",
                color_continuous_scale="RdYlGn", title="Avg. Sentiment by News Source",
                template="plotly_dark",
            )
            st.plotly_chart(fig_src, use_container_width=True)

    with tab3:
        display_df = df[["published", "sentiment_label", "sentiment_score", "headline", "source"]].copy()
        display_df["sentiment_score"] = display_df["sentiment_score"].round(3)
        display_df = display_df.sort_values("sentiment_score")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    st.markdown("### 👈 Enter a brand/topic and click **Fetch & Analyze Live Headlines**")

st.markdown("---")
st.markdown("### ✍️ Or Analyze Your Own Text")
manual_text = st.text_area("Paste text (e.g. a tweet, review, or comment) to score directly", height=100)
if st.button("Analyze Text"):
    if manual_text.strip():
        score = analyzer.polarity_scores(manual_text)["compound"]
        label = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
        icon = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}[label]
        st.markdown(f"### {icon} **{label}** — score: {score:+.3f}")
    else:
        st.warning("Please enter some text first.")

st.markdown("---")
st.markdown(
    "Built by **Rahul Sah** · "
    "[GitHub](https://github.com/maatrixxrahul) · "
    "[Portfolio](https://maatrixxrahul.netlify.app)"
)
