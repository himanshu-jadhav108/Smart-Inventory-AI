# 🧠 Smart Inventory AI Pro

Enterprise-Grade AI-Powered Demand Forecasting & Inventory Optimization System

## 📋 Overview

Smart Inventory AI Pro is a comprehensive inventory management solution that leverages advanced AI and machine learning techniques to provide:

- 📈 **AI-Powered Demand Forecasting**: Ensemble forecasting combining EWMA, linear regression, and seasonal patterns
- 🔍 **Advanced Anomaly Detection**: Multi-method approach using statistical and IQR-based techniques
- 📦 **Dynamic Inventory Optimization**: Intelligent reorder point and safety stock calculations
- 💡 **Explainable AI Insights**: Business-friendly insights and recommendations
- 🎨 **Premium UI/UX**: Modern, professional interface with interactive dashboards

## 🚀 Quick Start

### Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Usage

1. **Choose Data Source**: Use AI-generated sample data or upload your own CSV file
2. **Select Product**: Choose which product to analyze
3. **Configure Parameters**: Set forecast horizon, service level, current stock, and lead time
4. **Run Analysis**: Click "Run Advanced Analysis" to generate forecasts and recommendations

## 📁 Project Structure

```
smart_inventory_ai/
│
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── config/
│   └── theme.py               # CSS styling and theme configuration
│
├── data/
│   └── sample_generator.py   # Sample data generation
│
├── models/
│   ├── forecasting.py        # Forecasting engine with ensemble methods
│   ├── anomaly.py            # Anomaly detection algorithms
│   ├── inventory.py          # Inventory optimization logic
│   └── explainability.py     # Business insights generation
│
├── dashboard/
│   ├── charts.py             # Plotly chart generation
│   ├── metrics.py            # Metric card components
│   └── layout.py             # UI layout and components
│
├── utils/
│   ├── data_loader.py        # Data loading and validation
│   └── helpers.py            # Helper utility functions
│
└── assets/
    └── styles.css            # Additional CSS styles
```

## 🎯 Features

### Forecasting Engine
- **EWMA (Exponential Weighted Moving Average)**: Captures recent trends
- **Linear Regression**: Identifies long-term patterns
- **Seasonal Decomposition**: Accounts for weekly patterns
- **Ensemble Method**: Combines multiple approaches for robust predictions

### Anomaly Detection
- **Statistical Method**: Rolling mean and standard deviation
- **IQR Method**: Interquartile range-based detection
- **Severity Scoring**: Quantifies anomaly magnitude

### Inventory Optimization
- **Dynamic Safety Stock**: Adapts to demand volatility
- **Service Level Optimization**: Configurable stockout risk
- **Reorder Point Calculation**: Considers lead time and demand variability
- **Order Quantity Recommendations**: Optimal replenishment amounts

### Business Insights
- Trend analysis and change detection
- Volatility assessment
- Anomaly summaries
- Forecast accuracy metrics
- Stock status warnings
- Seasonality patterns

## 📊 Data Format

### Required CSV Columns

```csv
date,product_id,units_sold
2024-01-01,PROD001,45
2024-01-02,PROD001,52
2024-01-03,PROD001,48
```

- **date**: Date in YYYY-MM-DD format
- **product_id**: Unique product identifier
- **units_sold**: Number of units sold (integer)

## 🛠️ Technical Details

### Technologies
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **ML/Analytics**: scikit-learn, scipy, pandas, numpy
- **Styling**: Custom CSS with modern design patterns

### Architecture Principles
- **Modular Design**: Clear separation of concerns
- **No Streamlit in Models**: Pure Python logic in model files
- **Clean Interfaces**: Well-defined function boundaries
- **Maintainability**: Easy to extend and modify

## 📈 Model Performance

The forecasting engine is validated using walk-forward validation with the following metrics:
- **MAE (Mean Absolute Error)**: Average prediction error in units
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based accuracy measure

Typical accuracy ratings:
- MAPE < 10%: Excellent
- MAPE 10-20%: Good
- MAPE > 20%: Fair

## 🎨 UI Features

- Modern gradient backgrounds
- Animated components (fade-in, slide-in effects)
- Interactive metric cards with hover effects
- Color-coded status indicators
- Responsive layout
- Professional typography
- Dark-themed sidebar

## 🔧 Customization

### Adding New Products
Simply upload a CSV with additional product_id values

### Adjusting Forecast Models
Modify weights in `models/forecasting.py` ensemble method:
```python
ensemble = (0.3 * linear_future + 0.4 * ewma_future + 0.3 * seasonal_future)
```

### Changing Themes
Edit `config/theme.py` to customize colors, fonts, and styles

### Adding New Metrics
Create new functions in `dashboard/metrics.py` and call from `dashboard/layout.py`

## 📝 License

This project is provided as-is for educational and commercial use.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Support

For questions or support, please open an issue in the repository.

---

**Built with ❤️ using Streamlit and Modern AI**
