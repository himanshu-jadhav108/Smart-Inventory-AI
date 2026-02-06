# 📦 Smart Inventory Tracker MVP

A hackathon-ready inventory forecasting and restocking recommendation tool for small retail businesses.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Upload Data
Use the provided `sample_sales_data.csv` or upload your own CSV file with columns:
- `date` (YYYY-MM-DD format)
- `product_id` (string)
- `units_sold` (integer)

## 📊 Features

### ✅ Demand Forecasting
- 7-day forecast using Linear Regression
- Simple, reliable, and explainable
- Trend analysis (increasing/decreasing/stable)

### ✅ Anomaly Detection
- Statistical spike/drop detection
- Rolling mean + standard deviation approach
- Visual markers on chart

### ✅ Restocking Recommendations
- Intelligent reorder quantity calculation
- Considers current stock and lead time
- Includes safety buffer

### ✅ Explainability Layer
- Plain-English explanations
- No black-box predictions
- Business-friendly insights

## 🎯 Usage Flow

1. **Upload CSV** - Load your sales history
2. **Select Product** - Choose which product to analyze
3. **Set Parameters** - Input current stock and lead time
4. **Run Analysis** - Get instant insights

## 📁 File Structure

```
.
├── app.py                      # Main Streamlit application
├── sample_sales_data.csv       # Demo dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🧠 How It Works

### Forecasting Method
Uses Linear Regression on historical time series data:
- X-axis: Day index (1, 2, 3, ...)
- Y-axis: Units sold
- Output: 7-day ahead predictions

### Anomaly Detection
Statistical approach using rolling windows:
- Calculate rolling mean and standard deviation
- Flag points beyond 2 standard deviations
- Works without machine learning

### Restocking Logic
```
Daily Demand = Total Forecast / 7
Expected Demand = Daily Demand × Lead Time
Safety Stock = Expected Demand × 20%
Reorder Quantity = (Expected Demand + Safety Stock) - Current Stock
```

## 🎯 Demo Tips

### Best Practices
- Use data with at least 14 days of history
- Include some sales spikes for anomaly demo
- Set realistic current stock levels
- Try different lead times to see impact

### Expected Outputs
- Interactive Plotly chart with forecast
- Key metrics dashboard
- Plain-English insights
- Detailed data tables

## 🚫 Known Limitations

- No authentication (demo only)
- Single product analysis per run
- Linear regression only (simple but stable)
- No database persistence
- CSV input only

## 📝 Sample Data Format

```csv
date,product_id,units_sold
2024-01-01,PROD001,45
2024-01-02,PROD001,52
2024-01-03,PROD001,48
```

## ⚡ Performance

- Loads in < 1 second
- Analysis runs in < 2 seconds
- Stable for demo presentations
- Works offline (no API calls)

## 🎓 Technical Stack

- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn (Linear Regression)
- **Viz**: Plotly
- **Language**: Python 3.8+

## 🏆 Hackathon Checklist

- ✅ Works end-to-end
- ✅ Clear UI/UX
- ✅ Business value
- ✅ Explainable AI
- ✅ Demo-stable
- ✅ Clean code
- ✅ Sample data included

## 🔧 Troubleshooting

**CSV not loading?**
- Check column names match exactly
- Ensure date format is YYYY-MM-DD
- Remove any extra headers

**Forecast looks wrong?**
- Need at least 7 days of history
- Check for data quality issues
- Verify units_sold are numeric

**No anomalies detected?**
- Data might be too consistent
- Try sample_sales_data.csv (has spike on day 19)

---

Built for hackathons • Simple • Stable • Demo-ready

## 👨‍💻 About the Team

Team Id : U621SF9X
TeamName: Eureka Fourge
Team Members : 
- Himanshu Jadhav
- Ritesh Gaike
- Yash Bhongale
- Onkar Kharat
