import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pymongo import MongoClient
import gridfs
from streamlit_autorefresh import st_autorefresh

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan AQI Monitor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Auto refresh every 10 minutes ────────────────────────
st_autorefresh(interval=10 * 60 * 1000, key="autorefresh")

# ── Custom CSS (dark neon look) ──────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #222222 0%, #444444 50%, #222222 100%);
    font-family: 'Inter', sans-serif;
}

/* Make containers invisible */
[data-testid="metric-container"], .glass-card, .edu-card, .forecast-card {
    background: transparent;
    border: none;
    box-shadow: none;
}

/* Section titles pop in white */
.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #444;
}

/* Alerts in neon colors */
.alert-poor {
    color: #ff0033;
    background: transparent;
    font-weight: 600;
}
.alert-moderate {
    color: #ffff00;
    background: transparent;
    font-weight: 600;
}
.alert-good {
    color: #00ff00;
    background: transparent;
    font-weight: 600;
}

/* Brighten tab labels */
[data-testid="stTabs"] button {
    color: #ffffff !important;
    font-weight: 600;
}

/* Brighten metric labels and values */
[data-testid="stMetricLabel"] {
    color: #ffffff !important;
}
[data-testid="stMetricValue"] {
    color: #00ffff !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] {
    color: #ff33cc !important;
}

/* Brighten selectbox labels */
label[data-testid="stWidgetLabel"] {
    color: #ffffff !important;
    font-weight: 600;
}

/* Dark gray text for Learn About AQI section */
.glass-card p, .edu-card p, .edu-card h4 {
    color: #cccccc !important;
}
.edu-card div {
    color: #999999 !important;
}

/* Dark gray city labels in Tab 3 forecast */
div[data-testid="stMarkdownContainer"] strong {
    color: #cccccc !important;
}

/* Neon colors for forecast numbers */
.forecast-card .aqi-number-1 { color: #00ffff !important; }
.forecast-card .aqi-number-2 { color: #a6ff00 !important; }
.forecast-card .aqi-number-3 { color: #ff9900 !important; }
.forecast-card .aqi-number-4 { color: #ff3399 !important; }
.forecast-card .aqi-number-5 { color: #9933ff !important; }

/* Softer pollutant metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 8px 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
[data-testid="stMetricLabel"] {
    color: #eeeeee !important;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #66ccff !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] {
    color: #ff99cc !important;
    font-weight: 500;
}

/* Scale cards */
.scale-card {
    border-radius: 12px;
    padding: 20px 12px;
    text-align: center;
    min-height: 170px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    border: 1px solid rgba(255,255,255,0.06);
}
.scale-num  { font-family: 'Inter', sans-serif; font-size: 40px; font-weight: 800; }
.scale-lbl  { font-family: 'Inter', sans-serif; font-size: 11px; letter-spacing: 0.08em; font-weight: 700; }
.scale-desc { font-family: 'Inter', sans-serif; font-size: 11px; opacity: 0.7; line-height: 1.5; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Constants ────────────────────────────────────────────
AQI_LABELS  = {1:"Good",2:"Fair",3:"Moderate",4:"Poor",5:"Very Poor"}
AQI_COLORS  = {1:"#00ff00",2:"#ffff00",3:"#ff9900",4:"#ff0033",5:"#9933ff"}
AQI_BADGE   = {1:"good",2:"fair",3:"moderate",4:"poor",5:"verypoor"}
POLLUTANTS  = ['pm2_5','pm10','so2','o3','co','nh3','no2','no']
POLL_LABELS = {'pm2_5':'PM2.5','pm10':'PM10','so2':'SO2','o3':'O3','co':'CO','nh3':'NH3','no2':'NO2','no':'NO'}
WHO_LIMITS  = {'pm2_5':15,'pm10':45,'so2':40,'o3':100,'co':4000,'nh3':None,'no2':25,'no':None}
CITIES      = ["Karachi","Lahore","Islamabad","Faisalabad","Peshawar","Multan","Quetta"]

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#555', size=10),
    margin=dict(l=0, r=0, t=10, b=0),
)

# ── MongoDB connection ────────────────────────────────────
def get_mongo():
    MONGO_URI = os.environ.get("MONGO_URI")
    client  = MongoClient(MONGO_URI)
    db      = client["aqi_db"]
    fs_grid = gridfs.GridFS(db)
    return client, db, fs_grid


# ── Data loading from MongoDB ────────────────────────────
@st.cache_data(ttl=600)   # refresh every 10 minutes
def load_data():
    try:
        client, db, _ = get_mongo()
        collection = db["pakistan_aqi"]

        # Load last 30 days
        cutoff = datetime.utcnow() - timedelta(days=30)
        cursor = collection.find(
            {"timestamp_pk": {"$gte": cutoff}},
            {"_id": 0}
        ).sort("timestamp_pk", 1)

        df = pd.DataFrame(list(cursor))
        client.close()

        if df.empty:
            st.error("No data found in MongoDB.")
            return pd.DataFrame()

        df['timestamp_pk'] = pd.to_datetime(df['timestamp_pk'], utc=True)
        df = df.sort_values(['city', 'timestamp_pk']).reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Could not load data from MongoDB: {e}")
        return pd.DataFrame()


# ── Model loading from MongoDB GridFS ───────────────────
@st.cache_resource
def load_model(filename="random_forest.pkl"):
    try:
        client, db, fs_grid = get_mongo()
        model_file = fs_grid.find_one({"filename": filename}, sort=[("uploadDate", -1)])
        if model_file is None:
            return None, None

        tmp_model = f"/tmp/{filename}"
        with open(tmp_model, "wb") as f:
            f.write(model_file.read())

        scaler_file = fs_grid.find_one({"filename": "scaler.pkl"}, sort=[("uploadDate", -1)])
        tmp_scaler = "/tmp/scaler.pkl"
        with open(tmp_scaler, "wb") as f:
            f.write(scaler_file.read())

        client.close()

        # Load both model and scaler
        model = joblib.load(tmp_model)
        scaler = joblib.load(tmp_scaler)
        return model, scaler

    except Exception as e:
        st.error(f"Could not load model {filename}: {e}")
        return None, None



# ── Forecast & Alerts ────────────────────────────────────
def predict_forecast(df, city, model, scaler, is_lstm=False):
    city_df = df[df['city'] == city].sort_values('timestamp_pk')
    if city_df.empty or model is None:
        return [3, 3, 3]

    last_row    = city_df.iloc[-1]
    current_aqi = float(aqi_from_recent(city_df, n=6))
    forecasts   = []

    for day in range(3):
        future_dt = datetime.now() + timedelta(days=day + 1)
        features  = np.array([[
            float(last_row['pm2_5']), float(last_row['pm10']),
            float(last_row['so2']),   float(last_row['o3']),
            float(last_row['co']),    float(last_row['nh3']),
            12, future_dt.weekday(),
            1 if future_dt.weekday() >= 5 else 0,
            current_aqi
        ]])

        if is_lstm:
            features_sc = scaler.transform(features)
            features_sc = features_sc.reshape((1, 1, features_sc.shape[1]))
            pred = float(model.predict(features_sc, verbose=0)[0][0])
        else:
            pred = float(model.predict(features)[0])

        pred = max(1, min(5, round(pred)))
        forecasts.append(pred)
        current_aqi = pred

    return forecasts


def show_alert(city, aqi):
    if aqi >= 5:
        st.markdown(f'<div class="alert-poor">🚨 <b>{city}</b> air quality is <b>Very Poor</b></div>', unsafe_allow_html=True)
    elif aqi >= 4:
        st.markdown(f'<div class="alert-poor">⚠️ <b>{city}</b> air quality is <b>Poor</b></div>', unsafe_allow_html=True)
    elif aqi >= 3:
        st.markdown(f'<div class="alert-moderate">🟡 <b>{city}</b> air quality is <b>Moderate</b></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-good">✔️ <b>{city}</b> air quality is <b>{AQI_LABELS.get(aqi,"")}</b></div>', unsafe_allow_html=True)


def aqi_from_recent(city_df, n=6):
    return int(round(city_df.tail(n)['aqi'].mean()))


# ── Manual refresh button ────────────────────────────────
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ── Main App ─────────────────────────────────────────────
with st.spinner("Loading data from MongoDB..."):
    df = load_data()
    model_rf,  scaler_rf  = load_model("random_forest.pkl")
    model_gb,  scaler_gb  = load_model("gradient_boosting.pkl")
    model_rr,  scaler_rr  = load_model("ridge_regression.pkl")

if df.empty: st.stop()

tab1,tab2,tab3,tab4,tab5 = st.tabs([" Live Dashboard"," City Comparison"," Forecast"," Learn About AQI"," SHAP Explainability"])

# ── TAB 1: Live Dashboard ────────────────────────────────
with tab1:
    col_city,col_poll,col_model = st.columns(3)
    city = col_city.selectbox("🏙️ Select city",CITIES,index=5)
    selected_poll = col_poll.selectbox(" Select pollutant",["All pollutants"]+[POLL_LABELS[p] for p in POLLUTANTS])
    selected_model = col_model.selectbox(" Model used",["Random Forest","Gradient Boosting","Ridge Regression","LSTM"])

    # choose model only for Live Dashboard
    if selected_model == "Random Forest":
    model, scaler, is_lstm = model_rf, scaler_rf, False
   elif selected_model == "Gradient Boosting":
    model, scaler, is_lstm = model_gb, scaler_gb, False
   elif selected_model == "Ridge Regression":
    model, scaler, is_lstm = model_rr, scaler_rr, False
    else:
    st.error("❌ Selected model not available")
    model, scaler, is_lstm = None, None, False



    city_df = df[df['city']==city].sort_values('timestamp_pk')
    latest = city_df.iloc[-1] if not city_df.empty else None
    if latest is not None:
        aqi_val = aqi_from_recent(city_df, n=6)
        show_alert(city, aqi_val)

        # pollutant metrics filtered
        if selected_poll!="All pollutants":
            poll_key = [k for k,v in POLL_LABELS.items() if v==selected_poll][0]
            st.metric(selected_poll,f"{round(float(latest[poll_key]),1)} μg/m³",
                      delta=f"WHO limit: {WHO_LIMITS.get(poll_key,'N/A')} μg/m³", delta_color="off")
        else:
            m1,m2 = st.columns(2)
            with m1: 
                st.metric("PM2.5", f"{round(float(latest['pm2_5']),1)} μg/m³", delta="WHO limit: 15 μg/m³", delta_color="off")
            with m2: 
                st.metric("PM10", f"{round(float(latest['pm10']),1)} μg/m³", delta="WHO limit: 45 μg/m³", delta_color="off")

        # AQI trend last 24 hours
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown('<div class="section-title">AQI trend — last 24 hours</div>', unsafe_allow_html=True)
            last_24h = city_df[city_df['timestamp_pk'] >= city_df['timestamp_pk'].max() - pd.Timedelta(hours=24)]
            if not last_24h.empty:
                fig = go.Figure(go.Bar(
                    x=last_24h['timestamp_pk'], y=last_24h['aqi'],
                    marker_color=[AQI_COLORS.get(int(a), "#888") for a in last_24h['aqi']],
                    hovertemplate="<b>%{x}</b><br>AQI: %{y}<extra></extra>"
                ))
                fig.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0,5.5], tickvals=[1,2,3,4,5],
                               ticktext=["Good","Fair","Moderate","Poor","Very Poor"],
                               gridcolor='rgba(0,0,0,0.05)'),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.05)'))
                st.plotly_chart(fig, use_container_width=True)
            
        
        with c2:
            st.markdown('<div class="section-title">Current AQI level</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi_val,
                number=dict(font=dict(size=48, family='Syne, sans-serif', color=AQI_COLORS.get(aqi_val, "#888"))),
                gauge=dict(
                    axis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5],
                              ticktext=["G", "F", "M", "P", "VP"],
                              tickfont=dict(size=9, color='#444')),
                    bar=dict(color=AQI_COLORS.get(aqi_val, "#888"), thickness=0.25),
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 1], color="rgba(0,230,118,0.08)"),
                        dict(range=[1, 2], color="rgba(255,214,0,0.08)"),
                        dict(range=[2, 3], color="rgba(255,145,0,0.08)"),
                        dict(range=[3, 4], color="rgba(255,50,50,0.08)"),
                        dict(range=[4, 5], color="rgba(180,0,255,0.08)"),
                    ],
                    threshold=dict(
                        line=dict(color=AQI_COLORS.get(aqi_val, "#888"), width=3),
                        thickness=0.8, value=aqi_val
                    )
                ),
                title=dict(text=AQI_LABELS.get(aqi_val, ""),
                           font=dict(size=13, family='Space Mono, monospace', color='#555'))
            ))
            fig_gauge.update_layout(
                **{**PLOT_LAYOUT, 'margin': dict(l=20, r=20, t=30, b=0)},
                height=260,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Pollutant breakdown
        p1, p2 = st.columns([1.2, 1])
        with p1:
            st.markdown('<div class="section-title">Pollutant breakdown</div>', unsafe_allow_html=True)
            poll_rows = []
            for p in POLLUTANTS:
                val = round(float(latest[p]), 2) if p in latest.index else 0
                who = WHO_LIMITS.get(p)
                pct = min(val/who*100, 100) if who else min(val/500*100, 100)
                color = "#ff4d4f" if (who and val > who) else "#faad14" if pct > 50 else "#52c41a"
                poll_rows.append({"Pollutant": POLL_LABELS[p], "pct": pct, "color": color, "text": f"{val} μg/m³"})
            fig_poll = go.Figure()
            for row in poll_rows:
                fig_poll.add_trace(go.Bar(
                    y=[row["Pollutant"]], x=[row["pct"]], orientation='h',
                    marker_color=row["color"], text=row["text"],
                    textposition='outside', showlegend=False,
                    hovertemplate=f"<b>{row['Pollutant']}</b>: {row['text']}<extra></extra>"
                ))
            fig_poll.update_layout(
                height=280,
                margin=dict(l=0,r=80,t=10,b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[0,130], showticklabels=False, gridcolor='rgba(255,255,255,0.1)'),
                barmode='overlay'
            )
            st.plotly_chart(fig_poll, use_container_width=True)

        # Forecast (selected model)
        with p2:
            st.markdown('<div class="section-title">3-day AQI forecast</div>', unsafe_allow_html=True)
            forecasts = predict_forecast(df, city, model, scaler)
            fc1, fc2, fc3 = st.columns(3)
            for col, day, aqi_f in zip([fc1,fc2,fc3], ["Today","Tomorrow","Day 3"], forecasts):
                with col:
                    st.markdown(f"""
                    <div class="forecast-card">
                        <div style="font-size:12px;color:#00ffff;margin-bottom:6px;">{day}</div>
                        <div style="font-size:32px;font-weight:600;color:{AQI_COLORS.get(aqi_f,'#888')};">{aqi_f}</div>
                        <div style="font-size:12px;color:#ff33cc;margin-top:4px;">{AQI_LABELS.get(aqi_f,'')}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:12px;font-size:12px;color:#ffffff;">
                Predicted using <b>{selected_model}</b>. Based on current pollutant levels and time patterns.
            </div>
            """, unsafe_allow_html=True)


# ── TAB 2 ─────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">All cities — average AQI (last 30 days)</div>', unsafe_allow_html=True)
    city_avg = df.groupby('city')['aqi'].mean().sort_values(ascending=False).reset_index()
    city_avg.columns = ['City','Avg AQI']
    city_avg['Avg AQI'] = city_avg['Avg AQI'].round(2)

    fig_city = px.bar(city_avg, x='City', y='Avg AQI',
        color='Avg AQI', color_continuous_scale=[[0,'#52c41a'],[0.4,'#faad14'],[0.7,'#ff4d4f'],[1,'#722ed1']],
        range_color=[1,5], text='Avg AQI')
    fig_city.update_traces(texttemplate='%{text}', textposition='outside')
    fig_city.update_layout(height=350, coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(range=[0,6], gridcolor='rgba(0,0,0,0.05)'))
    st.plotly_chart(fig_city, use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">AQI trend — all cities over time</div>', unsafe_allow_html=True)

    daily = df.groupby(['city', pd.Grouper(key='timestamp_pk', freq='D')])['aqi'].mean().reset_index()
    fig_trend = px.line(daily, x='timestamp_pk', y='aqi', color='city',
                        markers=False, line_shape='spline')
    fig_trend.update_traces(line=dict(width=2))
    fig_trend.update_layout(
        **PLOT_LAYOUT,
        height=300,
        yaxis=dict(range=[0, 6], gridcolor='rgba(255,255,255,0.04)',
                   tickvals=[1,2,3,4,5],
                   ticktext=["Good","Fair","Moderate","Poor","Very Poor"],
                   tickfont=dict(size=9)),
        xaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(size=9)),
        legend=dict(font=dict(size=10, color='#888'), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown('<div class="section-title">AQI heatmap — city vs hour of day</div>', unsafe_allow_html=True)
    heat = df.groupby(['city','hour'])['aqi'].mean().reset_index()
    heat_pivot = heat.pivot(index='city', columns='hour', values='aqi')  # <-- missing line

    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale=["#00ff00","#ffff00","#ff9900","#ff0033","#9933ff"],  # neon scale
        zmin=1, zmax=5, aspect='auto',
        labels=dict(x="Hour of day", y="City", color="AQI Level")
    )
    fig_heat.update_layout(
        coloraxis_colorbar=dict(
            tickvals=[1,2,3,4,5],
            ticktext=["Good","Fair","Moderate","Poor","Very Poor"],
            title="AQI Level"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=10,b=0),
        font=dict(color="#ffffff")  # make labels visible
    )
    st.plotly_chart(fig_heat, use_container_width=True)




# ── TAB 3 ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">3-day AQI forecast — all cities</div>', unsafe_allow_html=True)
    st.info("Predictions made using Gradient Boosting (R²=0.9842, RMSE=0.0955) — best performing model trained on 30 days of historical data.")

    cols = st.columns(len(CITIES))
    for col, c in zip(cols, CITIES):
        forecasts = predict_forecast(df, c, model_gb, scaler_gb, is_lstm=False)
        with col:
            st.markdown(f"**{c}**")
            for day, aqi_f in zip(["Today","Tmrw","Day 3"], forecasts):
                st.markdown(f"""
                <div class="forecast-card" style="margin-bottom:8px;">
                    <div style="font-size:11px;color:#888;">{day}</div>
                    <div style="font-size:24px;font-weight:600;color:{AQI_COLORS.get(aqi_f,'#888')};">{aqi_f}</div>
                    <div style="font-size:11px;color:#666;">{AQI_LABELS.get(aqi_f,'')}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<br><div class="section-title">Model performance summary</div>', unsafe_allow_html=True)
    perf_df = pd.DataFrame({
        'Model': ['Gradient Boosting', 'Random Forest', 'Ridge Regression'],
        'RMSE':  [0.0955, 0.1007, 0.2828],
        'MAE':   [0.0226, 0.0243, 0.1846],
        'R²':    [0.9842, 0.9824, 0.8612],
    })
    st.dataframe(perf_df.style.highlight_max(subset=['R²'], color='#d9f7be')
                              .highlight_min(subset=['RMSE','MAE'], color='#d9f7be'),
                 use_container_width=True, hide_index=True)


# ── TAB 4 ─────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">What is AQI?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
    <p style="font-size:15px;color:#444;line-height:1.8;">
    The <b>Air Quality Index (AQI)</b> is a simple number that tells you how clean or polluted the air is.
    Think of it like a weather forecast — instead of temperature, it tells you how safe the air is to breathe.
    The higher the number, the worse the air quality and the greater the health concern.
    In Pakistan, cities like <b>Lahore</b> and <b>Multan</b> frequently experience Poor to Very Poor air quality,
    especially during winter months due to crop burning, vehicle emissions, and industrial activity.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">AQI scale</div>', unsafe_allow_html=True)
    scale_info = [
        (1, "Good",      "#00e676", "rgba(0,230,118,0.08)",
         "Air quality is satisfactory. Enjoy outdoor activities freely."),
        (2, "Fair",      "#ffd600", "rgba(255,214,0,0.08)",
         "Acceptable. Very sensitive people may feel mild effects."),
        (3, "Moderate",  "#ff9100", "rgba(255,145,0,0.08)",
         "Sensitive groups should limit prolonged outdoor time."),
        (4, "Poor",      "#ff3232", "rgba(255,50,50,0.08)",
         "Everyone may experience health effects. Reduce outdoor activity."),
        (5, "Very Poor", "#b400ff", "rgba(180,0,255,0.08)",
         "Health warnings. Avoid all outdoor activity."),
    ]
    for col, (num, label, color, bg, desc) in zip(st.columns(5), scale_info):
        with col:
            st.markdown(f"""
            <div class="scale-card" style="background:{bg};border-color:{color}22;">
                <div class="scale-num" style="color:{color};">{num}</div>
                <div class="scale-lbl" style="color:{color};">{label}</div>
                <div class="scale-desc" style="color:{color};">{desc}</div>
            </div>""", unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pollutants explained</div>', unsafe_allow_html=True)
    poll_guide = [
        ("PM2.5","🔴 Most dangerous","Tiny particles smaller than 2.5 microns. Enter deep into lungs. Main driver of AQI. WHO safe limit: 15 μg/m³."),
        ("PM10", "🟠 Dangerous",     "Larger dust particles from roads and construction. Causes respiratory issues. WHO safe limit: 45 μg/m³."),
        ("SO2",  "🟡 Moderate risk", "From burning coal and oil. Causes acid rain and respiratory irritation. Main source: power plants."),
        ("O3",   "🟡 Moderate risk", "Ground-level ozone from sunlight reacting with emissions. Peaks in afternoon. Worsens asthma."),
        ("CO",   "🟢 Lower risk",    "Carbon monoxide from incomplete combustion. Colorless, odorless — dangerous in enclosed spaces."),
        ("NO2",  "🟢 Lower risk",    "From vehicle exhaust and power plants. Irritates airways, contributes to smog."),
        ("NH3",  "🟢 Lower risk",    "Ammonia mainly from agriculture. Can react with other pollutants to form fine particles."),
        ("NO",   "🟢 Lower risk",    "Nitric oxide — quickly converts to NO2 in atmosphere. Primarily from combustion engines."),
    ]
    r1 = st.columns(4)
    r2 = st.columns(4)
    for col, (name, risk, desc) in zip(r1+r2, poll_guide):
        with col:
            st.markdown(f"""
            <div class="edu-card" style="margin-bottom:12px;">
                <h4>{name}</h4>
                <div style="font-size:11px;margin-bottom:6px;">{risk}</div>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">What causes bad air quality in Pakistan?</div>', unsafe_allow_html=True)
    causes = [
        ("🚗","Vehicle emissions","Millions of old vehicles burning low-quality fuel, especially in Lahore and Karachi."),
        ("🏭","Industrial activity","Factories in Faisalabad, Multan, and Karachi release pollutants with little filtering."),
        ("🌾","Crop burning","Farmers in Punjab burn leftover crops after harvest — a major cause of smog October–December."),
        ("🧱","Construction dust","Rapid urban development stirs up large amounts of PM10 and PM2.5 dust year-round."),
    ]
    for col, (icon, title, desc) in zip(st.columns(4), causes):
        with col:
            st.markdown(f"""
            <div class="edu-card">
                <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Health tips by AQI level</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "AQI Level":       ["1 — Good","2 — Fair","3 — Moderate","4 — Poor","5 — Very Poor"],
        "Who is at risk":  ["No one","Very sensitive individuals","Sensitive groups (asthma, elderly, children)","Everyone","Everyone"],
        "Outdoor activity":["✔️ Go outside freely","✔️ Fine for most","⚠️ Sensitive groups should limit","❌ Reduce outdoor time","❌ Stay indoors"],
        "Recommended action":["Enjoy fresh air","Monitor if sensitive","Wear a mask if sensitive","Wear N95 mask outdoors","Keep windows closed, use air purifier"],
    }), use_container_width=True, hide_index=True)

with tab5:
    st.markdown('<div class="section-title">🔎 AQI Explainability</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#cccccc;font-size:14px;margin-bottom:20px;">
    This section explains <b>why the model predicted the AQI value</b>.
    Think of it like a scoreboard: each feature (pollutant, time, or previous AQI) either
    pushed the prediction <span style="color:#ff3232;">up (worse air)</span> or
    <span style="color:#00e676;">down (better air)</span>.
    </p>
    """, unsafe_allow_html=True)

    try:
        import shap

        city_shap = st.selectbox("🏙️ Choose a city", CITIES, key="shap_city")
        city_df_shap = df[df['city'] == city_shap].sort_values('timestamp_pk').copy()

        if not city_df_shap.empty and model_rf is not None:

            FEATURE_NAMES = ['PM2.5','PM10','SO2','O3','CO','NH3','Hour','Weekday','Is Weekend','Prev AQI']

            # Add time features
            city_df_shap['weekday']    = pd.to_datetime(city_df_shap['timestamp_pk']).dt.weekday
            city_df_shap['is_weekend'] = (city_df_shap['weekday'] >= 5).astype(int)
            city_df_shap['hour']       = pd.to_datetime(city_df_shap['timestamp_pk']).dt.hour

            # Prepare last 100 rows
            X = city_df_shap[['pm2_5','pm10','so2','o3','co','nh3',
                               'hour','weekday','is_weekend','aqi']].dropna().tail(100)
            X.columns = FEATURE_NAMES

            explainer   = shap.TreeExplainer(model_rf)
            shap_values = explainer.shap_values(X)

            # ── Chart 1: Overall importance ──
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': mean_shap}).sort_values('Importance')

            st.markdown('<div class="section-title">📊 Which features matter most?</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#888;font-size:12px;">Bigger bars = stronger influence on AQI predictions.</p>', unsafe_allow_html=True)

            fig1 = go.Figure(go.Bar(
                x=shap_df['Importance'],
                y=shap_df['Feature'],
                orientation='h',
                marker=dict(color='#00bcd4'),
                text=[f"{float(v):.3f}" for v in shap_df['Importance']],
                textposition='outside',
                textfont=dict(color='#cccccc', size=11),
            ))
            fig1.update_layout(margin=dict(l=10, r=70, t=50, b=10), height=320,
                               xaxis=dict(showticklabels=False), yaxis=dict(tickfont=dict(color='#cccccc')))
            st.plotly_chart(fig1, use_container_width=True)

            # ── Chart 2: High vs Low effect ──
            st.markdown('<div class="section-title">⬆️⬇️ High vs Low Values</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#888;font-size:12px;">Red = higher AQI when feature is high. Green = lower AQI when feature is high.</p>', unsafe_allow_html=True)

            direction_rows = []
            for i, feat in enumerate(FEATURE_NAMES):
                sv   = shap_values[:, i]
                fv   = X.iloc[:, i].values
                med  = np.median(fv)
                avg_high = sv[fv >= med].mean() if (fv >= med).sum() > 0 else 0
                avg_low  = sv[fv < med].mean()  if (fv < med).sum()  > 0 else 0
                direction_rows.append({'Feature': feat, 'High': float(avg_high), 'Low': float(avg_low)})

            dir_df = pd.DataFrame(direction_rows).sort_values('High', key=lambda x: abs(x))

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name='High value',
                y=dir_df['Feature'],
                x=dir_df['High'],
                orientation='h',
                marker=dict(color=['#ff3232' if v > 0 else '#00e676' for v in dir_df['High']]),
                text=[f"{float(v):+.3f}" for v in dir_df['High']],
                textposition='outside',
                textfont=dict(color='#cccccc', size=10),
            ))
            fig2.add_trace(go.Bar(
                name='Low value',
                y=dir_df['Feature'],
                x=dir_df['Low'],
                orientation='h',
                marker=dict(color=['rgba(255,50,50,0.3)' if v > 0 else 'rgba(0,230,118,0.3)' for v in dir_df['Low']]),
                text=[f"{float(v):+.3f}" for v in dir_df['Low']],
                textposition='outside',
                textfont=dict(color='#888', size=10),
            ))
            fig2.update_layout(margin=dict(l=10, r=80, t=50, b=10), height=360, barmode='overlay',
                               xaxis=dict(title=dict(text='Impact on AQI', font=dict(color='#888', size=11))),
                               yaxis=dict(tickfont=dict(color='#cccccc')))
            st.plotly_chart(fig2, use_container_width=True)

            # ── Chart 3: Latest prediction breakdown ──
            st.markdown('<div class="section-title">🧾 Latest Prediction Explained</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#888;font-size:12px;">Shows how each feature pushed the final AQI up or down compared to the baseline.</p>', unsafe_allow_html=True)

            latest_sv   = shap_values[-1]
            base_val    = explainer.expected_value
            waterfall_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'SHAP': latest_sv}).sort_values('SHAP', key=lambda x: abs(x))

            fig3 = go.Figure(go.Bar(
                x=waterfall_df['SHAP'],
                y=waterfall_df['Feature'],
                orientation='h',
                marker=dict(color=['#ff3232' if float(v) > 0 else '#00e676' for v in waterfall_df['SHAP']]),
                text=[f"{float(v):+.3f}" for v in waterfall_df['SHAP']],
                textposition='outside',
                textfont=dict(color='#cccccc', size=11),
            ))
            fig3.add_vline(x=0, line_color='#555', line_width=1)
            fig3.update_layout(margin=dict(l=10, r=70, t=50, b=30), height=340,
                               xaxis=dict(title=dict(text=f'Baseline AQI: {float(base_val):.2f}', font=dict(color='#888', size=11))),
                               yaxis=dict(tickfont=dict(color='#cccccc')))
            st.plotly_chart(fig3, use_container_width=True)

        else:
            st.warning("No data available for SHAP analysis.")

    except ImportError:
        st.error("SHAP not installed. Run: pip install shap")
    except Exception as e:
        st.error(f"SHAP error: {e}")
