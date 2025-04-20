# 🕵️‍♂️ Anomaly Detector for App Reviews

An intelligent NLP-based Streamlit app that detects **fake or suspicious app reviews** using machine learning. Paste any review and instantly find out if it's **Genuine ✅** or **Fake ❌**!

---

## 📌 Overview

This project is a mini NLP-based anomaly detection tool that:
- Parses app reviews,
- Cleans and vectorizes text using **TF-IDF**,
- Uses **Isolation Forest** to detect fake or unusual reviews,
- Provides a friendly **Streamlit UI** for live prediction.

---

## 🧠 How It Works

1. App reviews are cleaned (lowercase, no punctuation).
2. TF-IDF Vectorizer converts text into numerical features.
3. Isolation Forest flags anomalies (fake reviews).
4. The model is trained on a labeled dataset and saved.
5. Users input new reviews through the web app to get real-time predictions.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Bharadwaj204/anomaly_app.git
cd anomaly_app
