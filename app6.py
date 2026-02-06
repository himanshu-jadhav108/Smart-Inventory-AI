"""
🏆 ULTIMATE SMART INVENTORY TRACKER - PREMIUM UI EDITION
Hybrid AI-Powered Demand Forecasting & Inventory Optimization System
Enhanced with modern, professional frontend design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION & PREMIUM THEME
# ============================================================

st.set_page_config(
    page_title="Smart Inventory AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS with Modern Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: #ffffff;
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    /* Header Styles */
    .main-header {
        color: #667eea;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #475569;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        margin: 0;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    
    /* Recommendation Boxes */
    .recommendation-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.3);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-box::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .critical-box {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 20px 40px rgba(239, 68, 68, 0.3);
        margin: 2rem 0;
        animation: shake 0.5s ease-in-out;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    .safe-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
    
    .recommendation-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .recommendation-value {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .recommendation-details {
        font-size: 1.1rem;
        line-height: 1.8;
        opacity: 0.95;
    }
    
    /* Insight Cards */
    .insight-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .insight-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 0.25rem;
    }
    
    .badge-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    /* Progress Bars */
    .progress-container {
        background: #e2e8f0;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 1s ease;
    }
    
    /* Data Table Styles */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Charts Container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1.5rem 0;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        font-weight: 600;
        padding: 1rem;
    }
    
    /* Info/Warning/Error Boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #667eea;
        font-weight: 600;
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Dashboard Grid */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Feature Tag */
    .feature-tag {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    /* Glassmorphism Effect */
    .glass {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
    }
    
    /* Additional Visibility Improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    
    p, span, div {
        color: #334155;
    }
    
    .stMarkdown {
        color: #334155;
    }
    
    /* Improve expander visibility */
    .streamlit-expanderHeader p {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stAlert p {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. DATA GENERATION & LOADING
# ============================================================

def generate_sample_data(days=120, num_products=5):
    """Generate realistic sales data with multiple patterns"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    products = [f'PROD-{str(i).zfill(3)}' for i in range(1, num_products + 1)]

    data = []
    for i, product in enumerate(products):
        base = 30 + (i * 20) + np.random.randint(-5, 5)

        if i % 3 == 0:
            trend = np.linspace(0, 30, days)
        elif i % 3 == 1:
            trend = np.linspace(20, -10, days)
        else:
            trend = np.zeros(days)

        seasonality = 10 * np.sin(np.linspace(0, 8*np.pi, days))
        monthly = 5 * np.sin(np.linspace(0, 4*np.pi, days))
        noise = np.random.normal(0, 3, days)

        anomalies = np.zeros(days)
        anomaly_days = np.random.choice(days, size=3, replace=False)
        for day in anomaly_days:
            anomalies[day] = np.random.choice([-20, 25])

        units_sold = base + trend + seasonality + monthly + noise + anomalies
        units_sold = np.maximum(units_sold, 0).astype(int)

        for j, date in enumerate(dates):
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'product_id': product,
                'units_sold': units_sold[j]
            })

    return pd.DataFrame(data)


def load_and_validate_data(uploaded_file=None):
    """Load data from upload or generate sample"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    else:
        df = generate_sample_data()
        st.sidebar.info("📊 Using AI-generated sample data")

    required_cols = ['date', 'product_id', 'units_sold']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        return None

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').fillna(0)
    df = df.sort_values('date').reset_index(drop=True)

    return df

# ============================================================
# 2. ADVANCED FORECASTING ENGINE
# ============================================================

class ForecastingEngine:
    def __init__(self, data):
        self.data = data.copy()
        self.data['day_index'] = range(len(self.data))

    def ewma_forecast(self, span=14):
        """Exponential Weighted Moving Average"""
        return self.data['units_sold'].ewm(span=span, adjust=False).mean()

    def linear_trend(self):
        """Linear regression trend"""
        X = self.data[['day_index']].values
        y = self.data['units_sold'].values
        model = LinearRegression()
        model.fit(X, y)
        return model, model.predict(X)

    def seasonal_pattern(self):
        """Weekly seasonality"""
        self.data['day_of_week'] = pd.to_datetime(self.data['date']).dt.dayofweek
        pattern = self.data.groupby('day_of_week')['units_sold'].mean()
        return pattern / pattern.mean()

    def ensemble_forecast(self, forecast_days=14):
        """Combine multiple methods for robust forecasting"""
        ewma = self.ewma_forecast()
        model, linear_pred = self.linear_trend()
        seasonal = self.seasonal_pattern()

        recent_data = self.data.tail(14)
        recent_avg = recent_data['units_sold'].mean()
        recent_trend = (recent_data['units_sold'].iloc[-1] - recent_data['units_sold'].iloc[0]) / 14

        last_date = pd.to_datetime(self.data['date'].max())
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

        forecasts = []
        confidences = []

        for i, date in enumerate(forecast_dates):
            day_idx = len(self.data) + i
            day_of_week = date.dayofweek

            linear_future = model.predict([[day_idx]])[0]
            ewma_future = recent_avg + (recent_trend * i)
            seasonal_factor = seasonal[day_of_week]
            seasonal_future = recent_avg * seasonal_factor

            ensemble = (0.3 * linear_future + 0.4 * ewma_future + 0.3 * seasonal_future)
            ensemble = max(0, ensemble)

            historical_std = self.data['units_sold'].tail(30).std()
            confidence = 1.96 * historical_std

            forecasts.append(ensemble)
            confidences.append(confidence)

        return forecast_dates, forecasts, confidences

    def calculate_accuracy(self):
        """Calculate model accuracy on recent data"""
        if len(self.data) < 30:
            return None

        train = self.data[:-7]
        test = self.data[-7:]

        if len(train) < 14:
            return None

        train_engine = ForecastingEngine(train)
        _, pred, _ = train_engine.ensemble_forecast(forecast_days=7)
        actual = test['units_sold'].values

        mae = mean_absolute_error(actual, pred)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100 if np.all(actual > 0) else None

        return {'mae': mae, 'mape': mape, 'actual': actual, 'predicted': pred}

# ============================================================
# 3. ANOMALY DETECTION
# ============================================================

def detect_anomalies_advanced(data, window=7, std_threshold=2):
    """Advanced anomaly detection with multiple methods"""
    df = data.copy()

    df['rolling_mean'] = df['units_sold'].rolling(window=window, min_periods=1).mean()
    df['rolling_std'] = df['units_sold'].rolling(window=window, min_periods=1).std().fillna(0)

    df['upper_bound'] = df['rolling_mean'] + (std_threshold * df['rolling_std'])
    df['lower_bound'] = df['rolling_mean'] - (std_threshold * df['rolling_std'])

    Q1 = df['units_sold'].quantile(0.25)
    Q3 = df['units_sold'].quantile(0.75)
    IQR = Q3 - Q1
    iqr_upper = Q3 + 1.5 * IQR
    iqr_lower = Q1 - 1.5 * IQR

    df['is_anomaly'] = (
        (df['units_sold'] > df['upper_bound']) | 
        (df['units_sold'] < df['lower_bound']) |
        (df['units_sold'] > iqr_upper) |
        (df['units_sold'] < iqr_lower)
    )

    df['anomaly_severity'] = np.where(
        df['is_anomaly'],
        np.abs(df['units_sold'] - df['rolling_mean']) / (df['rolling_std'] + 1),
        0
    )

    return df

# ============================================================
# 4. INVENTORY OPTIMIZATION
# ============================================================

def calculate_optimal_inventory(forecast_values, confidences, current_stock, lead_time, 
                                service_level=0.95):
    """Advanced inventory optimization"""
    total_demand = sum(forecast_values)
    daily_demand = total_demand / len(forecast_values)

    forecast_std = np.std(forecast_values)
    lead_time_demand = daily_demand * lead_time
    lead_time_variance = forecast_std * np.sqrt(lead_time)

    try:
        from scipy.stats import norm
        z_score = norm.ppf(service_level)
    except:
        z_score = 1.65  # Default for 95% service level

    safety_stock = z_score * lead_time_variance

    cv = forecast_std / daily_demand if daily_demand > 0 else 0
    dynamic_buffer = max(0.15, min(0.35, cv * 0.5))

    reorder_point = lead_time_demand + safety_stock
    stock_position = current_stock

    if stock_position <= reorder_point:
        order_quantity = (reorder_point - stock_position) + (daily_demand * 7)
        should_order = True
    else:
        order_quantity = 0
        should_order = False

    return {
        'reorder_point': round(reorder_point),
        'order_quantity': round(order_quantity),
        'safety_stock': round(safety_stock),
        'should_order': should_order,
        'daily_demand': round(daily_demand, 1),
        'total_forecast': round(total_demand),
        'dynamic_buffer': round(dynamic_buffer * 100),
        'stock_position': stock_position,
        'days_until_stockout': round(stock_position / daily_demand, 1) if daily_demand > 0 else 999,
        'confidence_interval': round(np.mean(confidences), 1)
    }

# ============================================================
# 5. EXPLAINABILITY ENGINE
# ============================================================

def generate_insights(product_data, forecast_result, anomaly_data, inventory_rec, accuracy_metrics):
    """Generate comprehensive business insights"""
    insights = []

    recent_7 = product_data['units_sold'].tail(7).mean()
    prev_7 = product_data['units_sold'].tail(14).head(7).mean()
    trend_change = ((recent_7 - prev_7) / prev_7 * 100) if prev_7 > 0 else 0

    if abs(trend_change) > 15:
        trend_icon = "📈" if trend_change > 0 else "📉"
        badge = "badge-success" if trend_change > 0 else "badge-danger"
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">{trend_icon}</span>'
                       f'<strong>Significant Trend:</strong> Sales changed by '
                       f'<span class="status-badge {badge}">{abs(trend_change):.1f}%</span> '
                       f'{"up" if trend_change > 0 else "down"} compared to previous week</div>')
    else:
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">➡️</span>'
                       f'<strong>Stable Trend:</strong> Sales consistent at ~{recent_7:.0f} units/day</div>')

    cv = product_data['units_sold'].tail(30).std() / recent_7 if recent_7 > 0 else 0
    if cv > 0.4:
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">⚠️</span>'
                       f'<strong>High Volatility</strong> <span class="status-badge badge-warning">CV: {cv:.2f}</span>: '
                       f'Demand unpredictable, {inventory_rec["dynamic_buffer"]}% safety buffer applied</div>')
    elif cv > 0.25:
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">📊</span>'
                       f'<strong>Moderate Volatility</strong> <span class="status-badge badge-info">CV: {cv:.2f}</span>: '
                       f'Some demand variation expected</div>')
    else:
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">✅</span>'
                       f'<strong>Low Volatility</strong> <span class="status-badge badge-success">CV: {cv:.2f}</span>: '
                       f'Very predictable demand pattern</div>')

    anomalies = anomaly_data[anomaly_data['is_anomaly']]
    if len(anomalies) > 0:
        recent_anomalies = anomalies.tail(3)
        dates = recent_anomalies['date'].dt.strftime('%m-%d').tolist()
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">🚨</span>'
                       f'<strong>{len(anomalies)} Anomalies Detected:</strong> Recent unusual activity on '
                       f'{", ".join(dates)}. Review for promotions or stockouts.</div>')
    else:
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">✅</span>'
                       f'<strong>No Anomalies:</strong> Sales pattern is normal and predictable</div>')

    if accuracy_metrics and accuracy_metrics.get('mape'):
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">🎯</span>'
                       f'<strong>Forecast Accuracy:</strong> Historical MAPE of '
                       f'<span class="status-badge badge-success">{accuracy_metrics["mape"]:.1f}%</span> '
                       f'(±{inventory_rec["confidence_interval"]} units confidence interval)</div>')

    days_left = inventory_rec['days_until_stockout']
    lead_time = inventory_rec.get('lead_time', 7)

    if days_left < lead_time:
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">🚨</span>'
                       f'<strong>Critical Stock:</strong> Only <span class="status-badge badge-danger">'
                       f'{days_left} days</span> of inventory left! Order immediately to avoid stockout.</div>')
    elif days_left < lead_time * 2:
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">⚠️</span>'
                       f'<strong>Low Stock:</strong> <span class="status-badge badge-warning">{days_left} days</span> '
                       f'remaining. Reorder recommended.</div>')
    else:
        insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">✅</span>'
                       f'<strong>Healthy Stock:</strong> <span class="status-badge badge-success">{days_left} days</span> '
                       f'of coverage available</div>')

    dow_pattern = product_data.groupby(product_data['date'].dt.dayofweek)['units_sold'].mean()
    peak_day = dow_pattern.idxmax()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    insights.append(f'<div class="insight-card fade-in"><span class="insight-icon">📅</span>'
                   f'<strong>Peak Day:</strong> {days[peak_day]} '
                   f'<span class="status-badge badge-info">avg: {dow_pattern[peak_day]:.0f} units</span></div>')

    return insights

# ============================================================
# 6. VISUALIZATION DASHBOARD
# ============================================================

def create_dashboard(historical_df, forecast_dates, forecast_values, confidences, anomaly_df):
    """Create comprehensive visualization dashboard with premium styling"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('📊 Sales History & AI Forecast', '🔍 Anomaly Detection Timeline', 
                       '📅 Weekly Demand Pattern', '📦 Inventory Projection'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Color scheme
    colors = {
        'historical': '#667eea',
        'anomaly': '#ef4444',
        'forecast': '#10b981',
        'confidence': 'rgba(16, 185, 129, 0.15)',
        'weekly': '#f59e0b',
        'inventory': '#8b5cf6'
    }

    # 1. Main Chart: Historical + Forecast
    normal_data = anomaly_df[~anomaly_df['is_anomaly']]
    fig.add_trace(
        go.Scatter(
            x=normal_data['date'], 
            y=normal_data['units_sold'],
            mode='lines', 
            name='Historical Sales', 
            line=dict(color=colors['historical'], width=3),
            hovertemplate='<b>%{x}</b><br>Sales: %{y} units<extra></extra>'
        ),
        row=1, col=1
    )

    anomaly_data = anomaly_df[anomaly_df['is_anomaly']]
    if len(anomaly_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomaly_data['date'], 
                y=anomaly_data['units_sold'],
                mode='markers', 
                name='Anomalies',
                marker=dict(
                    color=colors['anomaly'], 
                    size=12, 
                    symbol='x',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>ANOMALY</b><br>Date: %{x}<br>Sales: %{y} units<extra></extra>'
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates, 
            y=forecast_values,
            mode='lines+markers', 
            name='AI Forecast',
            line=dict(color=colors['forecast'], width=4, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>FORECAST</b><br>%{x}<br>Predicted: %{y} units<extra></extra>'
        ),
        row=1, col=1
    )

    # Confidence interval
    upper = [f + c for f, c in zip(forecast_values, confidences)]
    lower = [max(0, f - c) for f, c in zip(forecast_values, confidences)]
    fig.add_trace(
        go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper + lower[::-1],
            fill='toself', 
            fillcolor=colors['confidence'],
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence', 
            showlegend=True,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # 2. Anomaly Timeline
    anomaly_colors = [colors['anomaly'] if x else '#cbd5e1' for x in anomaly_df['is_anomaly']]
    fig.add_trace(
        go.Bar(
            x=anomaly_df['date'], 
            y=anomaly_df['anomaly_severity'],
            marker_color=anomaly_colors,
            name='Anomaly Score',
            hovertemplate='<b>%{x}</b><br>Severity: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Weekly Pattern
    historical_df['day_of_week'] = pd.to_datetime(historical_df['date']).dt.dayofweek
    weekly = historical_df.groupby('day_of_week')['units_sold'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    bar_colors = [colors['weekly'] if i == weekly.idxmax() else '#94a3b8' for i in range(7)]
    
    fig.add_trace(
        go.Bar(
            x=days, 
            y=weekly.values, 
            marker_color=bar_colors,
            name='Weekly Pattern',
            text=[f'{v:.0f}' for v in weekly.values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Avg Sales: %{y:.1f} units<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Inventory Projection
    cumulative_demand = np.cumsum(forecast_values)
    fig.add_trace(
        go.Scatter(
            x=forecast_dates, 
            y=cumulative_demand,
            mode='lines+markers', 
            name='Cumulative Demand',
            line=dict(color=colors['inventory'], width=4),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.1)',
            hovertemplate='<b>%{x}</b><br>Total Demand: %{y:.0f} units<extra></extra>'
        ),
        row=2, col=2
    )

    # Update layout with premium styling
    fig.update_layout(
        height=900,
        showlegend=True,
        template='plotly_white',
        font=dict(family="Inter, sans-serif", size=12),
        title=dict(
            text="<b>AI-Powered Inventory Intelligence Dashboard</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#1e293b')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='white'
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(203, 213, 225, 0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(203, 213, 225, 0.3)')

    return fig


def create_metric_card(label, value, delta=None, icon="📊"):
    """Create a styled metric card"""
    delta_html = ""
    if delta:
        delta_color = "#10b981" if delta > 0 else "#ef4444"
        delta_symbol = "▲" if delta > 0 else "▼"
        delta_html = f'<div class="metric-delta" style="color: {delta_color};">{delta_symbol} {abs(delta):.1f}%</div>'
    
    return f"""
    <div class="metric-card slide-in">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Animated Header
    st.markdown('<p class="main-header fade-in">🧠 Smart Inventory AI Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enterprise-Grade Demand Forecasting & Inventory Optimization <span class="feature-tag">AI-Powered</span></p>', 
                unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.markdown("## ⚙️ Configuration Panel")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Data source
    data_source = st.sidebar.radio(
        "📂 Data Source:",
        ["📊 AI-Generated Sample", "📁 Upload Custom CSV"],
        help="Choose between demo data or upload your own sales history"
    )

    uploaded_file = None
    if data_source == "📁 Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file", 
            type=['csv'],
            help="Upload a CSV with columns: date, product_id, units_sold"
        )
        if uploaded_file is None:
            st.markdown("""
            <div class="insight-card">
                <h3>📋 Required CSV Format</h3>
                <p>Your file must include these columns:</p>
            </div>
            """, unsafe_allow_html=True)
            st.code("""date,product_id,units_sold
2024-01-01,PROD001,45
2024-01-02,PROD001,52
2024-01-03,PROD001,48""")
            return

    # Load data
    with st.spinner("🔄 Loading and validating data..."):
        df = load_and_validate_data(uploaded_file)
    
    if df is None:
        return

    # Product selection with search
    products = sorted(df['product_id'].unique())
    st.sidebar.markdown("### 🏷️ Product Selection")
    selected_product = st.sidebar.selectbox(
        "Choose Product:",
        products,
        help=f"Select from {len(products)} available products"
    )

    # Configuration
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### 🔮 Forecast Configuration")
    
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (days):",
        min_value=7,
        max_value=30,
        value=14,
        help="Number of days to forecast ahead"
    )
    
    service_level = st.sidebar.slider(
        "Service Level Target:",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f",
        help="Probability of not stocking out (95% = industry standard)"
    )

    st.sidebar.markdown("### 📦 Inventory Parameters")
    
    current_stock = st.sidebar.number_input(
        "Current Stock Level:",
        min_value=0,
        max_value=100000,
        value=100,
        step=10,
        help="Current units in inventory"
    )
    
    lead_time = st.sidebar.number_input(
        "Supplier Lead Time (days):",
        min_value=1,
        max_value=90,
        value=7,
        help="Days until new stock arrives after ordering"
    )

    # Filter product data
    product_data = df[df['product_id'] == selected_product].copy()

    if len(product_data) < 14:
        st.error("❌ **Insufficient Data**: Minimum 14 days of history required for accurate forecasting")
        return

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    
    # Run Analysis Button
    analyze_button = st.sidebar.button(
        "🚀 Run Advanced Analysis",
        type="primary",
        use_container_width=True
    )

    if analyze_button:
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Initialize
            status_text.text("🔧 Initializing AI forecasting engine...")
            progress_bar.progress(20)
            engine = ForecastingEngine(product_data)

            # Step 2: Forecast
            status_text.text("🔮 Generating ensemble predictions...")
            progress_bar.progress(40)
            forecast_dates, forecast_values, confidences = engine.ensemble_forecast(forecast_days)

            # Step 3: Anomalies
            status_text.text("🔍 Detecting anomalies...")
            progress_bar.progress(60)
            anomaly_df = detect_anomalies_advanced(product_data)

            # Step 4: Accuracy
            status_text.text("📊 Calculating model accuracy...")
            progress_bar.progress(75)
            accuracy = engine.calculate_accuracy()

            # Step 5: Optimization
            status_text.text("🎯 Optimizing inventory levels...")
            progress_bar.progress(90)
            inventory = calculate_optimal_inventory(
                forecast_values, confidences, current_stock, lead_time, service_level
            )
            inventory['lead_time'] = lead_time

            # Step 6: Insights
            status_text.text("💡 Generating business insights...")
            progress_bar.progress(95)
            insights = generate_insights(product_data, forecast_values, anomaly_df, inventory, accuracy)

            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            # Display Results
            st.markdown("## 📊 Analysis Results", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

            # Top Metrics in Premium Cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            recent_avg = product_data['units_sold'].tail(7).mean()
            prev_avg = product_data['units_sold'].tail(14).head(7).mean()
            trend_delta = ((recent_avg - prev_avg) / prev_avg * 100) if prev_avg > 0 else 0
            
            with col1:
                st.markdown(create_metric_card(
                    "Current Stock",
                    f"{current_stock:,}",
                    icon="📦"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card(
                    "7-Day Forecast",
                    f"{sum(forecast_values[:7]):,.0f}",
                    delta=trend_delta,
                    icon="📈"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card(
                    "Anomalies",
                    f"{anomaly_df['is_anomaly'].sum()}",
                    icon="⚠️"
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_metric_card(
                    "Daily Demand",
                    f"{inventory['daily_demand']:.1f}",
                    icon="📊"
                ), unsafe_allow_html=True)
            
            with col5:
                days_color = "🚨" if inventory['days_until_stockout'] < lead_time else "✅"
                st.markdown(create_metric_card(
                    "Days to Stockout",
                    f"{inventory['days_until_stockout']:.0f}",
                    icon=days_color
                ), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Main Recommendation Box
            if inventory['should_order']:
                if inventory['days_until_stockout'] < lead_time:
                    st.markdown(f"""
                    <div class="critical-box fade-in">
                        <div class="recommendation-title">🚨 CRITICAL: IMMEDIATE ACTION REQUIRED</div>
                        <div class="recommendation-value">{inventory['order_quantity']:,}</div>
                        <div style="font-size: 1.5rem; margin-bottom: 1rem;">UNITS TO ORDER NOW</div>
                        <div class="recommendation-details">
                            ⚡ <strong>Urgency:</strong> Only {inventory['days_until_stockout']:.0f} days of stock remaining<br>
                            📊 <strong>Reorder Point:</strong> {inventory['reorder_point']:,} units<br>
                            🛡️ <strong>Safety Stock:</strong> {inventory['safety_stock']:,} units ({inventory['dynamic_buffer']}% buffer)<br>
                            📈 <strong>Forecast Demand:</strong> {inventory['total_forecast']:,} units over {forecast_days} days<br>
                            🎯 <strong>Service Level:</strong> {service_level*100:.0f}% (Industry Standard)<br>
                            ⏱️ <strong>Lead Time:</strong> {lead_time} days
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="recommendation-box fade-in">
                        <div class="recommendation-title">📦 RESTOCKING RECOMMENDED</div>
                        <div class="recommendation-value">{inventory['order_quantity']:,}</div>
                        <div style="font-size: 1.5rem; margin-bottom: 1rem;">UNITS TO ORDER</div>
                        <div class="recommendation-details">
                            ✅ <strong>Stock Status:</strong> {inventory['days_until_stockout']:.0f} days remaining<br>
                            📊 <strong>Reorder Point:</strong> {inventory['reorder_point']:,} units<br>
                            🛡️ <strong>Safety Stock:</strong> {inventory['safety_stock']:,} units ({inventory['dynamic_buffer']}% buffer)<br>
                            📈 <strong>Forecast Demand:</strong> {inventory['total_forecast']:,} units over {forecast_days} days<br>
                            🎯 <strong>Service Level:</strong> {service_level*100:.0f}% confidence<br>
                            ⏱️ <strong>Lead Time:</strong> {lead_time} days
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-box fade-in">
                    <div class="recommendation-title">✅ INVENTORY LEVEL OPTIMAL</div>
                    <div class="recommendation-value">NO ORDER NEEDED</div>
                    <div class="recommendation-details">
                        🎯 <strong>Stock Coverage:</strong> {inventory['days_until_stockout']:.0f} days<br>
                        📊 <strong>Current Position:</strong> {current_stock:,} units<br>
                        📈 <strong>Daily Demand:</strong> {inventory['daily_demand']:.1f} units<br>
                        🛡️ <strong>Reorder Point:</strong> {inventory['reorder_point']:,} units<br>
                        ✅ <strong>Status:</strong> Well above reorder threshold
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Dashboard
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_dashboard(product_data, forecast_dates, forecast_values, confidences, anomaly_df)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Business Insights
            st.markdown("## 💡 AI-Generated Business Insights", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            for insight in insights:
                st.markdown(insight, unsafe_allow_html=True)

            # Detailed Forecast Data
            with st.expander("🔍 **View Detailed Forecast Data & Model Performance**", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Forecast Table")
                    forecast_df = pd.DataFrame({
                        'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                        'Forecast': [round(v) for v in forecast_values],
                        'Lower Bound': [round(max(0, v - c)) for v, c in zip(forecast_values, confidences)],
                        'Upper Bound': [round(v + c) for v, c in zip(forecast_values, confidences)],
                        'Confidence (±)': [round(c) for c in confidences]
                    })
                    st.dataframe(forecast_df, use_container_width=True, height=400)
                
                with col2:
                    st.markdown("### 🎯 Model Performance")
                    if accuracy:
                        st.metric("Mean Absolute Error (MAE)", f"{accuracy['mae']:.2f} units")
                        st.metric("Mean Absolute % Error (MAPE)", f"{accuracy['mape']:.2f}%")
                        
                        # Accuracy interpretation
                        if accuracy['mape'] < 10:
                            acc_status = "🟢 Excellent"
                        elif accuracy['mape'] < 20:
                            acc_status = "🟡 Good"
                        else:
                            acc_status = "🔴 Fair"
                        
                        st.markdown(f"**Accuracy Rating:** {acc_status}")
                        
                        # Comparison chart
                        comp_df = pd.DataFrame({
                            'Actual': accuracy['actual'],
                            'Predicted': accuracy['predicted']
                        })
                        st.line_chart(comp_df, use_container_width=True)
                    else:
                        st.info("Insufficient data for accuracy calculation (need 30+ days)")

            # Export Options
            st.markdown("## 📥 Export Options", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                forecast_csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Forecast CSV",
                    data=forecast_csv,
                    file_name=f"forecast_{selected_product}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                insights_text = "\n\n".join([insight.replace('<div class="insight-card fade-in">', '').replace('</div>', '') for insight in insights])
                st.download_button(
                    label="💡 Download Insights",
                    data=insights_text,
                    file_name=f"insights_{selected_product}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            
            with col3:
                st.markdown("📧 **Email Report** *(Coming Soon)*")

        except Exception as e:
            st.error(f"❌ **Analysis Error:** {str(e)}")
            st.exception(e)

    else:
        # Preview Mode - Enhanced
        st.markdown("## 👋 Welcome to Smart Inventory AI Pro", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card fade-in">
            <h3>🚀 Get Started</h3>
            <p>Configure your settings in the sidebar and click <strong>"Run Advanced Analysis"</strong> to:</p>
            <ul>
                <li>📈 Generate AI-powered demand forecasts</li>
                <li>🔍 Detect anomalies and unusual patterns</li>
                <li>📦 Optimize inventory levels</li>
                <li>💡 Receive actionable business insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Data Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card(
                "Total Records",
                f"{len(df):,}",
                icon="📊"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                "Products",
                f"{df['product_id'].nunique()}",
                icon="🏷️"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                "Date Range",
                f"{(df['date'].max() - df['date'].min()).days} days",
                icon="📅"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metric_card(
                "Avg Daily Sales",
                f"{df['units_sold'].mean():.0f}",
                icon="📈"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Data Preview
        st.markdown("### 📋 Data Preview", unsafe_allow_html=True)
        st.dataframe(
            df.head(20).style.background_gradient(cmap='Blues', subset=['units_sold']),
            use_container_width=True,
            height=400
        )

        # Quick Trend Visualization
        st.markdown("### 📊 Product Trends Overview", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        for i, prod in enumerate(products[:5]):  # Show top 5 products
            prod_data = df[df['product_id'] == prod]
            fig.add_trace(go.Scatter(
                x=prod_data['date'],
                y=prod_data['units_sold'],
                mode='lines',
                name=prod,
                line=dict(width=2),
                hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>Sales: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Feature Highlights
        st.markdown("### ✨ Key Features", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="insight-card">
                <h4>🤖 AI Ensemble Forecasting</h4>
                <p>Combines multiple algorithms (EWMA, Linear Regression, Seasonal Patterns) for superior accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-card">
                <h4>🔍 Advanced Anomaly Detection</h4>
                <p>Multi-method approach using statistical and IQR-based techniques</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="insight-card">
                <h4>📊 Dynamic Safety Stock</h4>
                <p>Adapts to demand volatility with intelligent buffer calculation</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
