# FINN Business Intelligence Dashboard

A comprehensive Streamlit dashboard for FINN's Growth Department to monitor and analyze car subscription business performance.

## Features

### Executive Summary Tab ✅
- **6 Key Business Metrics** with target comparisons:
  - Active Subscriptions
  - Monthly Recurring Revenue (MRR)
  - Net New Subscriptions
  - Net Revenue Retention (NRR)
  - Blended Customer Acquisition Cost (CAC)
  - Fleet Utilization

- **Performance Analysis Charts**:
  - MRR over time by Business Unit (B2C vs B2B)
  - Active Subscriptions trend
  - Daily Net New Subscriptions

### Acquisition Tab ✅
- **4 Acquisition Scorecards**:
  - New Subscriptions (with B2C/B2B split)
  - New Customers/Accounts
  - Funnel Conversion Rate
  - Blended Customer Acquisition Cost (CAC)

- **Funnel Analysis**:
  - B2C Funnel: Visits → Car Configs → Checkouts → Credit → Contracts → Delivery
  - B2B Funnel: Leads → SQLs → Proposals → Contracts → Delivery
  - Interactive funnel selector for "All" business units

- **Performance Charts**:
  - Top-of-funnel volume vs New Subscriptions over time
  - Channel performance: New subscriptions by channel
  - CAC analysis by acquisition channel

- **Global Filters** (apply to all tabs):
  - Date range (default: last 30 days)
  - Business Unit: All, B2C, B2B
  - Region: German states + Other
  - Channel: Paid Search, Social, Organic, etc.

### Retention Tab ✅
- **5 Retention Scorecards**:
  - Customer Churn Rate (logo churn with B2C/B2B split)
  - Subscription Churn Rate
  - Early Churn Rate (0-3 months)
  - Renewal/Continuation Rate
  - Expansion MRR Share

- **Advanced Analytics**:
  - Churn rates over time (customer vs subscription trends)
  - Cohort retention heatmap with monthly cohorts
  - MRR movement analysis (expansion vs contraction vs churn)
  - B2B account expansion by fleet size

- **Interactive Features**:
  - Business unit selector for cohort analysis
  - Color-coded retention heatmap
  - Account distribution insights for B2B

### ✅ **Complete Business Intelligence Dashboard**
All three tabs are now fully implemented with comprehensive analytics for FINN's car subscription business!

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone or download the project**
   ```bash
   cd /path/to/growth-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

4. **Open in browser**
   - The dashboard will automatically open at `http://localhost:8501`
   - If not, click the link in the terminal

## Project Structure

```
growth-dashboard/
├── dashboard.py           # Main Streamlit app
├── src/
│   └── data_generator.py  # Simulated data generation
├── data/                  # (Future: real data files)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Data Model

The dashboard uses simulated data with these dimensions:
- **Date**: Daily data for 6+ months
- **Business Unit**: B2C, B2B
- **Region**: German states (BY, BW, NW, HE, etc.) + Other
- **Channel**: Paid Search, Paid Social, Organic, Referral, Partnerships, Outbound

### Key Metrics
- New/Ended/Active Subscriptions
- Monthly Recurring Revenue (MRR)
- Customer Acquisition Cost (CAC)
- Fleet metrics (available/subscribed cars)
- Net Revenue Retention (NRR)

## Usage Tips

1. **Filters are global** - they apply to all tabs
2. **Date range** defaults to last 30 days but can be extended
3. **Hover over charts** for detailed tooltips
4. **Scorecards show** current value vs target with color-coded deltas
5. **B2C vs B2B split** is shown when "All" business units selected

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express & Graph Objects
- **Data**: Simulated using statistical models