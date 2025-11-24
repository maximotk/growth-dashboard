"""
FINN Growth Dashboard
Main Streamlit application for Growth Department analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import DataGenerator

# Page configuration
st.set_page_config(
    page_title="FINN Business Intelligence Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > div > div > div > div {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the simulated data"""
    generator = DataGenerator()
    subs_df = generator.generate_subscription_data()
    channel_df = generator.generate_channel_data()
    funnel_df = generator.generate_funnel_data()
    customer_df = generator.generate_customer_data()
    retention_df = generator.generate_retention_data()
    cohort_df = generator.generate_cohort_data()
    b2b_account_df = generator.generate_b2b_account_data()
    targets = generator.get_targets()
    return subs_df, channel_df, funnel_df, customer_df, retention_df, cohort_df, b2b_account_df, targets

@st.cache_data
def filter_data(df, date_range, business_units, regions, channels=None):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    # Date filter
    filtered_df = filtered_df[
        (filtered_df['date'] >= date_range[0]) & 
        (filtered_df['date'] <= date_range[1])
    ]
    
    # Business unit filter
    if 'All' not in business_units:
        filtered_df = filtered_df[filtered_df['business_unit'].isin(business_units)]
    
    # Region filter
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    
    # Channel filter (if applicable)
    if channels and 'channel' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['channel'].isin(channels)]
    
    return filtered_df

def calculate_metrics(filtered_df, targets):
    """Calculate KPIs from filtered data"""
    if filtered_df.empty:
        return {
            'active_subscriptions': 0,
            'mrr': 0,
            'net_new_subscriptions': 0,
            'nrr': 0,
            'blended_cac': 0,
            'fleet_utilization': 0
        }
    
    # Get latest date metrics
    latest_date = filtered_df['date'].max()
    latest_data = filtered_df[filtered_df['date'] == latest_date]
    
    # Active subscriptions (sum for latest date)
    active_subs = latest_data['active_subscriptions'].sum()
    
    # MRR (sum for latest date)
    mrr = latest_data['mrr'].sum()
    
    # Net new subscriptions (sum over the period)
    total_new = filtered_df['new_subscriptions'].sum()
    total_ended = filtered_df['ended_subscriptions'].sum()
    net_new = total_new - total_ended
    
    # NRR (simplified calculation - assume 95% + noise)
    nrr = 0.95 + np.random.normal(0, 0.02)  # Simplified for demo
    nrr = max(0.8, min(1.2, nrr))  # Reasonable bounds
    
    # Blended CAC
    total_acq_cost = filtered_df['acquisition_cost'].sum()
    blended_cac = total_acq_cost / total_new if total_new > 0 else 0
    
    # Fleet utilization (latest date)
    total_available = latest_data['fleet_available'].sum()
    total_subscribed = latest_data['fleet_subscribed'].sum()
    fleet_util = total_subscribed / total_available if total_available > 0 else 0
    
    return {
        'active_subscriptions': int(active_subs),
        'mrr': mrr,
        'net_new_subscriptions': int(net_new),
        'nrr': nrr,
        'blended_cac': blended_cac,
        'fleet_utilization': fleet_util
    }

def display_scorecard(title, value, target, format_type="number", delta_inverse=False):
    """Display a metric scorecard with target comparison"""
    
    # Format value based on type
    if format_type == "currency":
        formatted_value = f"€{value:,.0f}"
        formatted_target = f"€{target:,.0f}"
    elif format_type == "percentage":
        formatted_value = f"{value:.1%}"
        formatted_target = f"{target:.1%}"
        value = value * 100  # For delta calculation
        target = target * 100
    else:
        formatted_value = f"{value:,.0f}"
        formatted_target = f"{target:,.0f}"
    
    # Calculate delta
    if target > 0:
        delta_pct = ((value - target) / target) * 100
        if delta_inverse:  # For metrics where lower is better (like CAC)
            delta_pct = -delta_pct
    else:
        delta_pct = 0
    
    # Format delta
    delta_color = "normal"
    if delta_pct > 5:
        delta_color = "normal"  # Green
        delta_text = f"+{delta_pct:.1f}% vs target"
    elif delta_pct < -5:
        delta_color = "inverse"  # Red
        delta_text = f"{delta_pct:.1f}% vs target"
    else:
        delta_color = "off"  # Gray
        delta_text = f"{delta_pct:.1f}% vs target"
    
    st.metric(
        label=title,
        value=formatted_value,
        delta=delta_text,
        delta_color=delta_color
    )
    
    st.caption(f"Target: {formatted_target}")

def create_mrr_chart(filtered_df):
    """Create MRR over time chart by business unit"""
    if filtered_df.empty:
        return go.Figure()
    
    # Aggregate by date and business unit
    mrr_data = filtered_df.groupby(['date', 'business_unit'])['mrr'].sum().reset_index()
    
    fig = px.line(
        mrr_data, 
        x='date', 
        y='mrr', 
        color='business_unit',
        title='Monthly Recurring Revenue Over Time',
        labels={'mrr': 'MRR (€)', 'date': 'Date'}
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="MRR (€)"
    )
    
    return fig

def create_subscriptions_chart(filtered_df):
    """Create active subscriptions and net new chart"""
    if filtered_df.empty:
        return go.Figure()
    
    # Aggregate by date
    daily_data = filtered_df.groupby('date').agg({
        'active_subscriptions': 'sum',
        'new_subscriptions': 'sum',
        'ended_subscriptions': 'sum'
    }).reset_index()
    
    daily_data['net_new'] = daily_data['new_subscriptions'] - daily_data['ended_subscriptions']
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Active Subscriptions Over Time', 'Net New Subscriptions (Daily)'),
        vertical_spacing=0.15
    )
    
    # Active subscriptions line
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'], 
            y=daily_data['active_subscriptions'],
            mode='lines',
            name='Active Subscriptions',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Net new subscriptions bars
    colors = ['red' if x < 0 else 'green' for x in daily_data['net_new']]
    fig.add_trace(
        go.Bar(
            x=daily_data['date'], 
            y=daily_data['net_new'],
            name='Net New Subscriptions',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Active Subscriptions", row=1, col=1)
    fig.update_yaxes(title_text="Net New Subscriptions", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig

def calculate_acquisition_metrics(filtered_subs_df, filtered_channel_df, filtered_funnel_df, filtered_customer_df):
    """Calculate acquisition KPIs from filtered data"""
    if filtered_subs_df.empty:
        return {
            'new_subscriptions': 0,
            'new_customers': 0,
            'funnel_conversion': 0,
            'blended_cac': 0,
            'b2c_new_subs': 0,
            'b2b_new_subs': 0,
            'b2c_new_customers': 0,
            'b2b_new_customers': 0
        }
    
    # New subscriptions (sum over the period)
    total_new = filtered_subs_df['new_subscriptions'].sum()
    b2c_new = filtered_subs_df[filtered_subs_df['business_unit'] == 'B2C']['new_subscriptions'].sum()
    b2b_new = filtered_subs_df[filtered_subs_df['business_unit'] == 'B2B']['new_subscriptions'].sum()
    
    # New customers (from customer data)
    total_customers = filtered_customer_df['new_customers'].sum() if not filtered_customer_df.empty else int(total_new * 0.7)
    b2c_customers = filtered_customer_df[filtered_customer_df['business_unit'] == 'B2C']['new_customers'].sum() if not filtered_customer_df.empty else int(b2c_new * 0.95)
    b2b_customers = filtered_customer_df[filtered_customer_df['business_unit'] == 'B2B']['new_customers'].sum() if not filtered_customer_df.empty else int(b2b_new * 0.3)
    
    # Funnel conversion (from funnel data)
    if not filtered_funnel_df.empty:
        b2c_funnel = filtered_funnel_df[filtered_funnel_df['business_unit'] == 'B2C']
        b2b_funnel = filtered_funnel_df[filtered_funnel_df['business_unit'] == 'B2B']
        
        if not b2c_funnel.empty:
            b2c_conversion = b2c_funnel['contracts_signed'].sum() / b2c_funnel['visits'].sum() if b2c_funnel['visits'].sum() > 0 else 0
        else:
            b2c_conversion = 0.05  # Default
            
        if not b2b_funnel.empty:
            b2b_conversion = b2b_funnel['contracts_signed'].sum() / b2b_funnel['leads'].sum() if b2b_funnel['leads'].sum() > 0 else 0
        else:
            b2b_conversion = 0.08  # Default
        
        # Combined conversion (weighted by volume)
        if total_new > 0:
            funnel_conversion = (b2c_conversion * b2c_new + b2b_conversion * b2b_new) / total_new
        else:
            funnel_conversion = 0
    else:
        funnel_conversion = 0.06  # Default
    
    # Blended CAC
    total_acq_cost = filtered_subs_df['acquisition_cost'].sum()
    blended_cac = total_acq_cost / total_new if total_new > 0 else 0
    
    return {
        'new_subscriptions': int(total_new),
        'new_customers': int(total_customers),
        'funnel_conversion': funnel_conversion,
        'blended_cac': blended_cac,
        'b2c_new_subs': int(b2c_new),
        'b2b_new_subs': int(b2b_new),
        'b2c_new_customers': int(b2c_customers),
        'b2b_new_customers': int(b2b_customers)
    }

def create_funnel_chart(filtered_funnel_df, business_unit):
    """Create funnel visualization"""
    if filtered_funnel_df.empty:
        return go.Figure()
    
    # Filter by business unit if not "All"
    if business_unit == "All":
        # Show B2C funnel by default, add radio button in the main function
        funnel_data = filtered_funnel_df[filtered_funnel_df['business_unit'] == 'B2C']
        bu_title = "B2C"
    else:
        funnel_data = filtered_funnel_df[filtered_funnel_df['business_unit'] == business_unit]
        bu_title = business_unit
    
    if funnel_data.empty:
        return go.Figure()
    
    # Aggregate funnel data
    if bu_title == "B2C":
        steps_data = {
            'Visits': funnel_data['visits'].sum(),
            'Car Configs': funnel_data['car_configs'].sum(),
            'Checkouts Started': funnel_data['checkouts_started'].sum(),
            'Credit Passed': funnel_data['credit_passed'].sum(),
            'Contracts Signed': funnel_data['contracts_signed'].sum(),
            'Cars Delivered': funnel_data['cars_delivered'].sum()
        }
    else:  # B2B
        steps_data = {
            'Leads': funnel_data['leads'].sum(),
            'SQLs': funnel_data['sqls'].sum(),
            'Proposals': funnel_data['proposals'].sum(),
            'Contracts Signed': funnel_data['contracts_signed'].sum(),
            'Cars Delivered': funnel_data['cars_delivered'].sum()
        }
    
    # Create DataFrame for plotting
    steps_df = pd.DataFrame([
        {'step': step, 'value': value, 'index': i} 
        for i, (step, value) in enumerate(steps_data.items())
    ])
    
    # Calculate conversion rates
    steps_df['conversion'] = 0.0
    for i in range(1, len(steps_df)):
        if steps_df.iloc[i-1]['value'] > 0:
            steps_df.iloc[i, steps_df.columns.get_loc('conversion')] = (steps_df.iloc[i]['value'] / steps_df.iloc[i-1]['value']) * 100
    
    # Create horizontal bar chart (funnel-like)
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=steps_df['step'],
        x=steps_df['value'],
        orientation='h',
        marker=dict(
            color=px.colors.sequential.Blues_r[1:len(steps_df)+1],
            line=dict(color='white', width=1)
        ),
        text=[f"{int(val):,}" for val in steps_df['value']],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))
    
    fig.update_layout(
        title=f"{bu_title} Acquisition Funnel",
        xaxis_title="Volume",
        yaxis_title="Funnel Step",
        height=400,
        yaxis={'categoryorder': 'array', 'categoryarray': steps_df['step'].tolist()[::-1]}
    )
    
    return fig

def create_tofu_vs_subs_chart(filtered_funnel_df, filtered_subs_df, business_unit):
    """Create top-of-funnel vs new subscriptions time series"""
    if filtered_funnel_df.empty or filtered_subs_df.empty:
        return go.Figure()
    
    # Aggregate by date and business unit
    if business_unit == "All":
        # Show both B2C and B2B
        b2c_funnel = filtered_funnel_df[filtered_funnel_df['business_unit'] == 'B2C'].groupby('date').agg({
            'visits': 'sum', 'contracts_signed': 'sum'
        }).reset_index()
        b2c_funnel['business_unit'] = 'B2C'
        b2c_funnel['tofu'] = b2c_funnel['visits']
        
        b2b_funnel = filtered_funnel_df[filtered_funnel_df['business_unit'] == 'B2B'].groupby('date').agg({
            'leads': 'sum', 'contracts_signed': 'sum'
        }).reset_index()
        b2b_funnel['business_unit'] = 'B2B' 
        b2b_funnel['tofu'] = b2b_funnel['leads']
        
        # Combine data
        combined_data = []
        for _, row in b2c_funnel.iterrows():
            combined_data.append({'date': row['date'], 'metric': 'B2C Visits', 'value': row['tofu']})
            combined_data.append({'date': row['date'], 'metric': 'B2C New Subs', 'value': row['contracts_signed']})
        
        for _, row in b2b_funnel.iterrows():
            combined_data.append({'date': row['date'], 'metric': 'B2B Leads', 'value': row['tofu']})
            combined_data.append({'date': row['date'], 'metric': 'B2B New Subs', 'value': row['contracts_signed']})
        
        plot_df = pd.DataFrame(combined_data)
        
        fig = px.line(plot_df, x='date', y='value', color='metric',
                     title='Top-of-Funnel Volume vs New Subscriptions Over Time')
        
    else:
        # Show single business unit
        bu_funnel = filtered_funnel_df[filtered_funnel_df['business_unit'] == business_unit]
        
        if business_unit == "B2C":
            daily_data = bu_funnel.groupby('date').agg({
                'visits': 'sum', 'contracts_signed': 'sum'
            }).reset_index()
            
            # Create combined data for plotting
            combined_data = []
            for _, row in daily_data.iterrows():
                combined_data.append({'date': row['date'], 'metric': 'Visits', 'value': row['visits']})
                combined_data.append({'date': row['date'], 'metric': 'New Subscriptions', 'value': row['contracts_signed']})
            
        else:  # B2B
            daily_data = bu_funnel.groupby('date').agg({
                'leads': 'sum', 'contracts_signed': 'sum'
            }).reset_index()
            
            combined_data = []
            for _, row in daily_data.iterrows():
                combined_data.append({'date': row['date'], 'metric': 'Leads', 'value': row['leads']})
                combined_data.append({'date': row['date'], 'metric': 'New Subscriptions', 'value': row['contracts_signed']})
        
        plot_df = pd.DataFrame(combined_data)
        fig = px.line(plot_df, x='date', y='value', color='metric',
                     title=f'{business_unit} Top-of-Funnel vs New Subscriptions Over Time')
    
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Volume")
    return fig

def create_channel_performance_charts(filtered_channel_df):
    """Create channel performance charts for subscriptions and CAC"""
    if filtered_channel_df.empty:
        return go.Figure(), go.Figure()
    
    # Aggregate by channel
    channel_data = filtered_channel_df.groupby('channel').agg({
        'new_subscriptions': 'sum',
        'acquisition_cost': 'sum'
    }).reset_index()
    
    # Calculate CAC per channel
    channel_data['cac'] = channel_data['acquisition_cost'] / channel_data['new_subscriptions']
    channel_data['cac'] = channel_data['cac'].fillna(0)
    
    # Sort by new subscriptions (descending)
    channel_data = channel_data.sort_values('new_subscriptions', ascending=True)  # True for horizontal bars
    
    # New subscriptions chart
    fig_subs = px.bar(
        channel_data, 
        x='new_subscriptions', 
        y='channel',
        orientation='h',
        title='New Subscriptions by Channel',
        labels={'new_subscriptions': 'New Subscriptions', 'channel': 'Channel'}
    )
    fig_subs.update_layout(height=300)
    
    # CAC chart (same order)
    fig_cac = px.bar(
        channel_data, 
        x='cac', 
        y='channel',
        orientation='h',
        title='Customer Acquisition Cost by Channel',
        labels={'cac': 'CAC (€)', 'channel': 'Channel'},
        color='cac',
        color_continuous_scale='RdYlBu_r'
    )
    fig_cac.update_layout(height=300)
    
    return fig_subs, fig_cac

def calculate_retention_metrics(filtered_retention_df, selected_business_units):
    """Calculate retention KPIs from filtered data"""
    if filtered_retention_df.empty:
        return {
            'customer_churn_rate': 0,
            'subscription_churn_rate': 0,
            'early_churn_rate': 0,
            'renewal_rate': 0,
            'expansion_mrr_share': 0,
            'b2c_customer_churn': 0,
            'b2b_customer_churn': 0,
            'b2c_subscription_churn': 0,
            'b2b_subscription_churn': 0
        }
    
    # Average metrics over the selected period
    if 'All' in selected_business_units:
        # Weighted average across business units
        b2c_data = filtered_retention_df[filtered_retention_df['business_unit'] == 'B2C']
        b2b_data = filtered_retention_df[filtered_retention_df['business_unit'] == 'B2B']
        
        b2c_customer_churn = b2c_data['customer_churn_rate'].mean() if not b2c_data.empty else 0
        b2b_customer_churn = b2b_data['customer_churn_rate'].mean() if not b2b_data.empty else 0
        b2c_subscription_churn = b2c_data['subscription_churn_rate'].mean() if not b2c_data.empty else 0
        b2b_subscription_churn = b2b_data['subscription_churn_rate'].mean() if not b2b_data.empty else 0
        
        # Overall averages (simple average for demo)
        customer_churn_rate = filtered_retention_df['customer_churn_rate'].mean()
        subscription_churn_rate = filtered_retention_df['subscription_churn_rate'].mean()
        
    else:
        # Single business unit
        customer_churn_rate = filtered_retention_df['customer_churn_rate'].mean()
        subscription_churn_rate = filtered_retention_df['subscription_churn_rate'].mean()
        b2c_customer_churn = customer_churn_rate if selected_business_units[0] == 'B2C' else 0
        b2b_customer_churn = customer_churn_rate if selected_business_units[0] == 'B2B' else 0
        b2c_subscription_churn = subscription_churn_rate if selected_business_units[0] == 'B2C' else 0
        b2b_subscription_churn = subscription_churn_rate if selected_business_units[0] == 'B2B' else 0
    
    early_churn_rate = filtered_retention_df['early_churn_rate'].mean()
    renewal_rate = filtered_retention_df['renewal_rate'].mean()
    expansion_mrr_share = filtered_retention_df['expansion_mrr_share'].mean()
    
    return {
        'customer_churn_rate': customer_churn_rate,
        'subscription_churn_rate': subscription_churn_rate,
        'early_churn_rate': early_churn_rate,
        'renewal_rate': renewal_rate,
        'expansion_mrr_share': expansion_mrr_share,
        'b2c_customer_churn': b2c_customer_churn,
        'b2b_customer_churn': b2b_customer_churn,
        'b2c_subscription_churn': b2c_subscription_churn,
        'b2b_subscription_churn': b2b_subscription_churn
    }

def create_churn_trends_chart(filtered_retention_df, selected_business_units):
    """Create churn rates over time chart"""
    if filtered_retention_df.empty:
        return go.Figure()
    
    # Prepare data for plotting
    plot_data = []
    
    if 'All' in selected_business_units:
        # Show both B2C and B2B
        for _, row in filtered_retention_df.iterrows():
            plot_data.append({
                'date': row['date'],
                'churn_type': f"{row['business_unit']} Customer Churn",
                'churn_rate': row['customer_churn_rate'] * 100
            })
            plot_data.append({
                'date': row['date'],
                'churn_type': f"{row['business_unit']} Subscription Churn",
                'churn_rate': row['subscription_churn_rate'] * 100
            })
    else:
        # Single business unit
        bu = selected_business_units[0] if selected_business_units else 'B2C'
        bu_data = filtered_retention_df[filtered_retention_df['business_unit'] == bu]
        
        for _, row in bu_data.iterrows():
            plot_data.append({
                'date': row['date'],
                'churn_type': 'Customer Churn',
                'churn_rate': row['customer_churn_rate'] * 100
            })
            plot_data.append({
                'date': row['date'],
                'churn_type': 'Subscription Churn',
                'churn_rate': row['subscription_churn_rate'] * 100
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    fig = px.line(plot_df, x='date', y='churn_rate', color='churn_type',
                  title='Churn Rates Over Time',
                  labels={'churn_rate': 'Churn Rate (%)', 'date': 'Date'})
    
    fig.update_layout(height=400)
    return fig

def create_cohort_heatmap(filtered_cohort_df, business_unit):
    """Create cohort retention heatmap"""
    if filtered_cohort_df.empty:
        return go.Figure()
    
    # Filter by business unit
    if business_unit == "All":
        # Default to B2C, will be controlled by radio button
        cohort_data = filtered_cohort_df[filtered_cohort_df['business_unit'] == 'B2C']
        title_bu = "B2C"
    else:
        cohort_data = filtered_cohort_df[filtered_cohort_df['business_unit'] == business_unit]
        title_bu = business_unit
    
    if cohort_data.empty:
        return go.Figure()
    
    # Pivot data for heatmap
    heatmap_data = cohort_data.pivot_table(
        index='cohort_month',
        columns='months_since_start',
        values='retention_rate',
        aggfunc='mean'
    )
    
    # Convert to percentage and round
    heatmap_data = (heatmap_data * 100).round(1)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"Month {i}" for i in heatmap_data.columns],
        y=[date.strftime('%Y-%m') for date in heatmap_data.index],
        colorscale='RdYlGn',
        zmin=0,
        zmax=100,
        colorbar=dict(title="Retention Rate (%)")
    ))
    
    fig.update_layout(
        title=f'{title_bu} Cohort Retention Heatmap',
        xaxis_title='Months Since Start',
        yaxis_title='Cohort Month',
        height=500
    )
    
    return fig

def create_mrr_movement_chart(filtered_retention_df):
    """Create MRR movement chart (expansion vs contraction vs churn)"""
    if filtered_retention_df.empty:
        return go.Figure()
    
    # Prepare data for stacked bar chart
    plot_data = []
    
    for _, row in filtered_retention_df.iterrows():
        plot_data.append({
            'date': row['date'],
            'component': 'Expansion MRR',
            'value': row['expansion_mrr'],
            'business_unit': row['business_unit']
        })
        plot_data.append({
            'date': row['date'],
            'component': 'Contraction MRR',
            'value': row['contraction_mrr'],
            'business_unit': row['business_unit']
        })
        plot_data.append({
            'date': row['date'],
            'component': 'Churn MRR',
            'value': row['churn_mrr'],
            'business_unit': row['business_unit']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Aggregate by date and component
    agg_df = plot_df.groupby(['date', 'component'])['value'].sum().reset_index()
    
    # Create stacked bar chart
    fig = px.bar(agg_df, x='date', y='value', color='component',
                 title='MRR Movement Over Time',
                 labels={'value': 'MRR (€)', 'date': 'Date'},
                 color_discrete_map={
                     'Expansion MRR': '#2E8B57',
                     'Contraction MRR': '#FF8C00', 
                     'Churn MRR': '#DC143C'
                 })
    
    fig.update_layout(height=400, barmode='relative')
    return fig

def create_b2b_account_expansion_chart(filtered_b2b_df):
    """Create B2B account expansion visualization"""
    if filtered_b2b_df.empty:
        return go.Figure()
    
    # Get latest period data
    latest_date = filtered_b2b_df['date'].max()
    latest_data = filtered_b2b_df[filtered_b2b_df['date'] == latest_date]
    
    # Create bar chart of account distribution
    fig = px.bar(latest_data, x='fleet_size_bucket', y='account_count',
                 title='B2B Account Distribution by Fleet Size',
                 labels={'account_count': 'Number of Accounts', 'fleet_size_bucket': 'Fleet Size'})
    
    fig.update_layout(height=300)
    return fig

def retention_tab(filtered_retention_df, filtered_cohort_df, filtered_b2b_df, selected_business_units):
    """Display Retention tab content"""
    st.header("🔄 Retention")
    
    # Calculate retention metrics
    metrics = calculate_retention_metrics(filtered_retention_df, selected_business_units)
    
    # Display scorecards
    st.subheader("Retention Performance")
    
    # First row (3 columns)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Customer Churn Rate", f"{metrics['customer_churn_rate']:.1%}")
        if 'All' in selected_business_units:
            st.caption(f"B2C: {metrics['b2c_customer_churn']:.1%} | B2B: {metrics['b2b_customer_churn']:.1%}")
    
    with col2:
        st.metric("Subscription Churn Rate", f"{metrics['subscription_churn_rate']:.1%}")
        if 'All' in selected_business_units:
            st.caption(f"B2C: {metrics['b2c_subscription_churn']:.1%} | B2B: {metrics['b2b_subscription_churn']:.1%}")
    
    with col3:
        st.metric("Early Churn (0-3 months)", f"{metrics['early_churn_rate']:.1%}")
        st.caption("Subscriptions ending within 3 months")
    
    # Second row (2 columns)
    col4, col5 = st.columns(2)
    
    with col4:
        st.metric("Renewal Rate", f"{metrics['renewal_rate']:.1%}")
        st.caption("Continuation rate at term end")
    
    with col5:
        st.metric("Expansion MRR Share", f"{metrics['expansion_mrr_share']:.1%}")
        st.caption("Expansion / Total MRR movement")
    
    st.divider()
    
    # Charts section
    st.subheader("📊 Retention Analysis")
    
    # Chart 1: Churn rates over time
    churn_fig = create_churn_trends_chart(filtered_retention_df, selected_business_units)
    st.plotly_chart(churn_fig, use_container_width=True)
    
    # Chart 2: Cohort retention heatmap
    st.subheader("👥 Cohort Retention Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if 'All' in selected_business_units:
            cohort_bu = st.radio("Cohort View", ["B2C", "B2B"], key="cohort_selector")
        else:
            cohort_bu = selected_business_units[0] if selected_business_units else "B2C"
            st.markdown(f"**Cohort View:** {cohort_bu}")
    
    with col1:
        cohort_fig = create_cohort_heatmap(filtered_cohort_df, cohort_bu)
        st.plotly_chart(cohort_fig, use_container_width=True)
    
    # Chart 3: MRR movement
    st.subheader("💰 MRR Movement Analysis")
    mrr_fig = create_mrr_movement_chart(filtered_retention_df)
    st.plotly_chart(mrr_fig, use_container_width=True)
    
    # Optional B2B account expansion
    if 'B2B' in selected_business_units or 'All' in selected_business_units:
        st.subheader("🏢 B2B Account Expansion")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            b2b_fig = create_b2b_account_expansion_chart(filtered_b2b_df)
            st.plotly_chart(b2b_fig, use_container_width=True)
        
        with col2:
            if not filtered_b2b_df.empty:
                latest_date = filtered_b2b_df['date'].max()
                latest_data = filtered_b2b_df[filtered_b2b_df['date'] == latest_date]
                
                total_accounts = latest_data['account_count'].sum()
                avg_cars = (latest_data['total_cars'].sum() / total_accounts) if total_accounts > 0 else 0
                
                st.metric("Total B2B Accounts", f"{total_accounts:,}")
                st.metric("Avg Cars per Account", f"{avg_cars:.1f}")
                
                # Show distribution
                st.markdown("**Fleet Size Distribution:**")
                for _, row in latest_data.iterrows():
                    pct = (row['account_count'] / total_accounts) * 100 if total_accounts > 0 else 0
                    st.write(f"• {row['fleet_size_bucket']}: {pct:.1f}%")

def acquisition_tab(filtered_subs_df, filtered_channel_df, filtered_funnel_df, filtered_customer_df, selected_business_units):
    """Display Acquisition tab content"""
    st.header("📈 Acquisition")
    
    # Calculate acquisition metrics
    metrics = calculate_acquisition_metrics(filtered_subs_df, filtered_channel_df, filtered_funnel_df, filtered_customer_df)
    
    # Display scorecards
    st.subheader("Acquisition Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("New Subscriptions", f"{metrics['new_subscriptions']:,}")
        if 'All' in selected_business_units:
            st.caption(f"B2C: {metrics['b2c_new_subs']:,} | B2B: {metrics['b2b_new_subs']:,}")
    
    with col2:
        st.metric("New Customers", f"{metrics['new_customers']:,}")
        if 'All' in selected_business_units:
            st.caption(f"B2C: {metrics['b2c_new_customers']:,} | B2B: {metrics['b2b_new_customers']:,}")
    
    with col3:
        st.metric("Funnel Conversion", f"{metrics['funnel_conversion']:.2%}")
        st.caption("Contracts / Top-of-funnel")
    
    with col4:
        st.metric("Blended CAC", f"€{metrics['blended_cac']:,.0f}")
        st.caption("Total cost / New subscriptions")
    
    st.divider()
    
    # Charts section
    st.subheader("📊 Acquisition Analysis")
    
    # Funnel chart and business unit selector
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if 'All' in selected_business_units:
            funnel_bu = st.radio("Funnel View", ["B2C", "B2B"], key="funnel_selector")
        else:
            funnel_bu = selected_business_units[0] if selected_business_units else "B2C"
            st.markdown(f"**Funnel View:** {funnel_bu}")
    
    with col1:
        funnel_fig = create_funnel_chart(filtered_funnel_df, funnel_bu)
        st.plotly_chart(funnel_fig, use_container_width=True)
    
    # Top-of-funnel vs new subscriptions
    st.subheader("🔄 Volume Trends")
    tofu_bu = 'All' if 'All' in selected_business_units else (selected_business_units[0] if selected_business_units else 'B2C')
    tofu_fig = create_tofu_vs_subs_chart(filtered_funnel_df, filtered_subs_df, tofu_bu)
    st.plotly_chart(tofu_fig, use_container_width=True)
    
    # Channel performance charts
    st.subheader("📢 Channel Performance")
    subs_fig, cac_fig = create_channel_performance_charts(filtered_channel_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(subs_fig, use_container_width=True)
    with col2:
        st.plotly_chart(cac_fig, use_container_width=True)

def executive_summary_tab(filtered_df, targets):
    """Display Executive Summary tab content"""
    st.header("📊 Summary")
    
    # Calculate metrics
    metrics = calculate_metrics(filtered_df, targets)
    
    # Display scorecards in two rows of 3 columns each
    st.subheader("Key Health Metrics")
    
    # First row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_scorecard(
            "Active Subscriptions", 
            metrics['active_subscriptions'], 
            targets['active_subscriptions']
        )
    
    with col2:
        display_scorecard(
            "Monthly Recurring Revenue", 
            metrics['mrr'], 
            targets['mrr'], 
            format_type="currency"
        )
    
    with col3:
        display_scorecard(
            "Net New Subscriptions", 
            metrics['net_new_subscriptions'], 
            targets['net_new_subscriptions']
        )
    
    # Second row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        display_scorecard(
            "Net Revenue Retention", 
            metrics['nrr'], 
            targets['nrr'], 
            format_type="percentage"
        )
    
    with col5:
        display_scorecard(
            "Blended CAC", 
            metrics['blended_cac'], 
            targets['blended_cac'], 
            format_type="currency",
            delta_inverse=True  # Lower is better for CAC
        )
    
    with col6:
        display_scorecard(
            "Fleet Utilization", 
            metrics['fleet_utilization'], 
            targets['fleet_utilization'], 
            format_type="percentage"
        )
    
    st.divider()
    
    # Charts section
    st.subheader("📈 Trends")
    
    # MRR chart
    col1, col2 = st.columns([2, 1])
    with col1:
        mrr_fig = create_mrr_chart(filtered_df)
        st.plotly_chart(mrr_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Revenue Breakdown")
        if not filtered_df.empty:
            latest_date = filtered_df['date'].max()
            latest_data = filtered_df[filtered_df['date'] == latest_date]
            b2c_mrr = latest_data[latest_data['business_unit'] == 'B2C']['mrr'].sum()
            b2b_mrr = latest_data[latest_data['business_unit'] == 'B2B']['mrr'].sum()
            
            st.metric("B2C MRR", f"€{b2c_mrr:,.0f}")
            st.metric("B2B MRR", f"€{b2b_mrr:,.0f}")
            
            if b2c_mrr + b2b_mrr > 0:
                b2c_pct = (b2c_mrr / (b2c_mrr + b2b_mrr)) * 100
                st.write(f"**B2C Share:** {b2c_pct:.1f}%")
        else:
            st.write("No data available for selected filters")
    
    # Subscriptions chart
    subs_fig = create_subscriptions_chart(filtered_df)
    st.plotly_chart(subs_fig, use_container_width=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🚗 Growth Business Intelligence Dashboard")
    #st.markdown("*Growth Analytics & Performance Monitoring*")
    
    # Load data
    with st.spinner("Loading data..."):
        subs_df, channel_df, funnel_df, customer_df, retention_df, cohort_df, b2b_account_df, targets = load_data()
    
    # Sidebar filters
    st.sidebar.title("🔍 Filters")
    
    # Date range filter
    min_date = subs_df['date'].min().date()
    max_date = subs_df['date'].max().date()
    default_start = max_date - timedelta(days=30)
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Convert to datetime for filtering
    if len(date_range) == 2:
        date_range = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    else:
        date_range = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[0]))
    
    # Business unit filter
    business_units = st.sidebar.multiselect(
        "Business Unit",
        options=['All', 'B2C', 'B2B'],
        default=['All']
    )
    
    # Region filter
    available_regions = sorted(subs_df['region'].unique())
    selected_regions = st.sidebar.multiselect(
        "Region",
        options=available_regions,
        default=available_regions
    )
    
    # Channel filter
    available_channels = sorted(channel_df['channel'].unique())
    selected_channels = st.sidebar.multiselect(
        "Channel",
        options=available_channels,
        default=available_channels
    )
    
    # Filter data
    filtered_subs_df = filter_data(subs_df, date_range, business_units, selected_regions)
    filtered_channel_df = filter_data(channel_df, date_range, business_units, [], selected_channels)
    filtered_funnel_df = filter_data(funnel_df, date_range, business_units, [])
    filtered_customer_df = filter_data(customer_df, date_range, business_units, [])
    filtered_retention_df = filter_data(retention_df, date_range, business_units, [])
    filtered_cohort_df = filter_data(cohort_df, date_range, business_units, [])
    filtered_b2b_df = filter_data(b2b_account_df, date_range, ['B2B'], [])
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter Summary:**")
    st.sidebar.write(f"📅 {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
    st.sidebar.write(f"🏢 {', '.join(business_units)}")
    st.sidebar.write(f"🌍 {len(selected_regions)} regions")
    st.sidebar.write(f"📢 {len(selected_channels)} channels")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Acquisition", "Retention"])
    
    with tab1:
        executive_summary_tab(filtered_subs_df, targets)
    
    with tab2:
        acquisition_tab(filtered_subs_df, filtered_channel_df, filtered_funnel_df, filtered_customer_df, business_units)
    
    with tab3:
        retention_tab(filtered_retention_df, filtered_cohort_df, filtered_b2b_df, business_units)

if __name__ == "__main__":
    main()