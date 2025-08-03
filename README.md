# Plot4PV: Solar Data Visualization Tool

This Streamlit app visualizes photovoltaic performance data — including Box Plots, IPCE Curves, and JV Curves — with interactive filtering and export options.

---

## Features
- 📊 Box Plot mode with mismatch correction and variation filtering
- 🌈 IPCE Curve plotting with normalization and scaling
- ⚡ JV Curve visualization with smoothing and normalization
- 📥 High-resolution export of figures (PNG/SVG)

---

## Installation & Running Locally

1. Clone this repository:
```bash
git clone https://github.com/YOUR-USERNAME/plot4pv.git
cd plot4pv
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run SeabornV1x4.py
```

---

## Deploying to Streamlit Cloud

1. Push this project to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Click **New app** → **Deploy from GitHub**.
4. Set:
   - **Main file path**: `SeabornV1x4.py`
   - Python version: 3.9+
5. Click **Deploy**.

Your app will be available at:
```
https://YOUR-USERNAME-plot4pv.streamlit.app
```

---

## Example Data
Upload your Excel or TXT files as prompted in the sidebar to generate plots.
