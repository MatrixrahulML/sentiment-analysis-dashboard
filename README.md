# 📊 Sentiment Analysis Dashboard

A live brand/topic sentiment tracker using VADER sentiment analysis on real-time news headlines, with trend charts and threshold-based alerts.

**[🚀 Live Demo](#)** &nbsp;|&nbsp; **[📂 Portfolio](https://maatrixxrahul.netlify.app)**

## Overview

1. **Live data** — fetches real-time headlines for any brand/topic from Google News RSS (free, public, no API key)
2. **Sentiment scoring** — each headline scored with VADER (rule-based, tuned for short informal text)
3. **Trend & distribution analysis** — sentiment over time, positive/neutral/negative breakdown, per-source sentiment comparison
4. **Alerting** — flags when average sentiment drops below a configurable threshold, simulating a brand-monitoring alert system
5. **Manual scoring** — paste any text (tweet, review, comment) for a direct sentiment score

> **Note on approach:** the original concept referenced Twitter (Tweepy). X's API now requires a paid tier, so this dashboard uses Google News RSS as its live data source instead — the sentiment scoring, trend analysis, and alerting logic are identical to what a Twitter-based version would run, just pointed at a different real-time text feed.

## Tech Stack

- **VADER (vaderSentiment)** — rule-based sentiment scoring
- **feedparser** — RSS ingestion
- **Streamlit** — dashboard
- **Plotly** — trend, distribution, and source charts

## Run Locally

```bash
git clone https://github.com/maatrixxrahul/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Rahul Sah** — [GitHub](https://github.com/maatrixxrahul) · [Portfolio](https://maatrixxrahul.netlify.app)
