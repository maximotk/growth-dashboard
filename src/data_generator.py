"""
Data generator for FINN Growth Dashboard
Creates simulated subscription and business data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List


class DataGenerator:
    def __init__(self, start_date: str = "2024-05-01", end_date: str = "2024-11-24"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Business dimensions
        self.business_units = ["B2C", "B2B"]
        self.regions = ["BY", "BW", "NW", "HE", "BE", "HH", "SN", "TH", "RP", "SL", "Other"]
        self.channels = ["Paid Search", "Paid Social", "Organic", "Referral", "Partnerships", "Outbound"]
        
        # Set random seed for reproducible data
        np.random.seed(42)
    
    def generate_subscription_data(self) -> pd.DataFrame:
        """Generate daily subscription data with trends"""
        data = []
        
        # Base metrics by business unit (B2C is larger)
        base_metrics = {
            "B2C": {
                "base_active": 8000,
                "base_new": 45,
                "base_ended": 25,
                "base_mrr": 350000,
                "base_acq_cost": 180,
                "growth_rate": 0.0015  # Daily growth rate
            },
            "B2B": {
                "base_active": 2500,
                "base_new": 15,
                "base_ended": 8,
                "base_mrr": 120000,
                "base_acq_cost": 420,
                "growth_rate": 0.002  # Slightly higher growth for B2B
            }
        }
        
        # Regional distribution weights
        region_weights = {
            "BY": 0.20, "BW": 0.15, "NW": 0.18, "HE": 0.12, 
            "BE": 0.08, "HH": 0.06, "SN": 0.05, "TH": 0.04,
            "RP": 0.04, "SL": 0.03, "Other": 0.05
        }
        
        # Channel distribution weights
        channel_weights = {
            "Paid Search": 0.25, "Paid Social": 0.20, "Organic": 0.18,
            "Referral": 0.15, "Partnerships": 0.12, "Outbound": 0.10
        }
        
        for date in self.date_range:
            days_from_start = (date - self.start_date).days
            
            for bu in self.business_units:
                bu_metrics = base_metrics[bu]
                
                # Apply growth trend
                growth_factor = 1 + (bu_metrics["growth_rate"] * days_from_start)
                
                # Seasonal effects (higher in autumn/winter)
                seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * (days_from_start + 90) / 365)
                
                # Weekend effects (lower activity)
                weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
                
                combined_factor = growth_factor * seasonal_factor * weekend_factor
                
                for region in self.regions:
                    region_factor = region_weights[region]
                    
                    # Add some noise
                    noise = np.random.normal(0.95, 0.15)
                    noise = max(0.5, min(1.5, noise))  # Cap the noise
                    
                    final_factor = combined_factor * region_factor * noise
                    
                    # Calculate metrics
                    new_subs = max(0, int(bu_metrics["base_new"] * final_factor))
                    ended_subs = max(0, int(bu_metrics["base_ended"] * final_factor * 0.9))  # Slightly lower churn
                    
                    # Active subs with cumulative effect
                    if days_from_start == 0:
                        active_subs = int(bu_metrics["base_active"] * region_factor)
                    else:
                        # This is simplified - in reality we'd track cumulative changes
                        active_subs = int(bu_metrics["base_active"] * region_factor * combined_factor)
                    
                    mrr = bu_metrics["base_mrr"] * region_factor * combined_factor
                    acq_cost = bu_metrics["base_acq_cost"] * np.random.normal(1.0, 0.2)
                    
                    # Fleet metrics (simplified)
                    if bu == "B2C":
                        fleet_available = int(active_subs * 1.2)  # 20% buffer
                        fleet_subscribed = active_subs
                    else:
                        fleet_available = int(active_subs * 1.15)  # 15% buffer for B2B
                        fleet_subscribed = active_subs
                    
                    data.append({
                        'date': date,
                        'business_unit': bu,
                        'region': region,
                        'new_subscriptions': new_subs,
                        'ended_subscriptions': ended_subs,
                        'active_subscriptions': active_subs,
                        'mrr': mrr,
                        'acquisition_cost': max(0, acq_cost) * new_subs if new_subs > 0 else 0,
                        'fleet_available': fleet_available,
                        'fleet_subscribed': fleet_subscribed
                    })
        
        return pd.DataFrame(data)
    
    def generate_channel_data(self) -> pd.DataFrame:
        """Generate acquisition data by channel"""
        data = []
        
        channel_metrics = {
            "Paid Search": {"cost_per_acq": 160, "conversion_rate": 0.045},
            "Paid Social": {"cost_per_acq": 140, "conversion_rate": 0.035},
            "Organic": {"cost_per_acq": 25, "conversion_rate": 0.065},
            "Referral": {"cost_per_acq": 45, "conversion_rate": 0.085},
            "Partnerships": {"cost_per_acq": 85, "conversion_rate": 0.055},
            "Outbound": {"cost_per_acq": 320, "conversion_rate": 0.025}
        }
        
        for date in self.date_range:
            for bu in self.business_units:
                for channel in self.channels:
                    # Get base new subscriptions for this date/BU (sum across regions)
                    base_new = np.random.poisson(20 if bu == "B2C" else 8)
                    
                    # Distribute across channels
                    if channel == "Paid Search":
                        new_subs = int(base_new * 0.25)
                    elif channel == "Paid Social":
                        new_subs = int(base_new * 0.20)
                    elif channel == "Organic":
                        new_subs = int(base_new * 0.18)
                    elif channel == "Referral":
                        new_subs = int(base_new * 0.15)
                    elif channel == "Partnerships":
                        new_subs = int(base_new * 0.12)
                    else:  # Outbound
                        new_subs = int(base_new * 0.10)
                    
                    # Add noise
                    new_subs = max(0, int(new_subs * np.random.normal(1.0, 0.3)))
                    
                    cost = new_subs * channel_metrics[channel]["cost_per_acq"] * np.random.normal(1.0, 0.2)
                    
                    data.append({
                        'date': date,
                        'business_unit': bu,
                        'channel': channel,
                        'new_subscriptions': new_subs,
                        'acquisition_cost': max(0, cost)
                    })
        
        return pd.DataFrame(data)
    
    def generate_funnel_data(self) -> pd.DataFrame:
        """Generate funnel data for B2C and B2B"""
        data = []
        
        # Funnel conversion rates (realistic percentages)
        funnel_rates = {
            "B2C": {
                "visits_to_configs": 0.12,
                "configs_to_checkout": 0.35,
                "checkout_to_credit": 0.75,
                "credit_to_contract": 0.85,
                "contract_to_delivery": 0.92
            },
            "B2B": {
                "leads_to_sqls": 0.25,
                "sqls_to_proposals": 0.45,
                "proposals_to_contracts": 0.35,
                "contracts_to_delivery": 0.88
            }
        }
        
        for date in self.date_range:
            days_from_start = (date - self.start_date).days
            
            # Seasonal and weekend effects
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * (days_from_start + 90) / 365)
            weekend_factor = 0.6 if date.weekday() >= 5 else 1.0
            combined_factor = seasonal_factor * weekend_factor
            
            for bu in self.business_units:
                # Get new subscriptions for this date/BU (sum across regions)
                base_new_subs = np.random.poisson(45 if bu == "B2C" else 15) * combined_factor
                
                if bu == "B2C":
                    # B2C Funnel (working backwards from contracts signed)
                    contracts_signed = int(base_new_subs)
                    credit_passed = int(contracts_signed / funnel_rates["B2C"]["credit_to_contract"])
                    checkouts_started = int(credit_passed / funnel_rates["B2C"]["checkout_to_credit"])
                    car_configs = int(checkouts_started / funnel_rates["B2C"]["configs_to_checkout"])
                    visits = int(car_configs / funnel_rates["B2C"]["visits_to_configs"])
                    cars_delivered = int(contracts_signed * funnel_rates["B2C"]["contract_to_delivery"])
                    
                    # Add some noise
                    noise = np.random.normal(1.0, 0.1)
                    visits = max(1, int(visits * noise))
                    car_configs = max(1, int(car_configs * noise))
                    checkouts_started = max(1, int(checkouts_started * noise))
                    
                    data.append({
                        'date': date,
                        'business_unit': bu,
                        'visits': visits,
                        'car_configs': car_configs,
                        'checkouts_started': checkouts_started,
                        'credit_passed': credit_passed,
                        'contracts_signed': contracts_signed,
                        'cars_delivered': cars_delivered,
                        'leads': 0,  # B2C doesn't use leads
                        'sqls': 0,
                        'proposals': 0
                    })
                
                else:  # B2B
                    # B2B Funnel (working backwards from contracts signed)
                    contracts_signed = int(base_new_subs)
                    proposals = int(contracts_signed / funnel_rates["B2B"]["proposals_to_contracts"])
                    sqls = int(proposals / funnel_rates["B2B"]["sqls_to_proposals"])
                    leads = int(sqls / funnel_rates["B2B"]["leads_to_sqls"])
                    first_cars_delivered = int(contracts_signed * funnel_rates["B2B"]["contracts_to_delivery"])
                    
                    # Add some noise
                    noise = np.random.normal(1.0, 0.1)
                    leads = max(1, int(leads * noise))
                    sqls = max(1, int(sqls * noise))
                    
                    data.append({
                        'date': date,
                        'business_unit': bu,
                        'visits': 0,  # B2B doesn't use visits
                        'car_configs': 0,
                        'checkouts_started': 0,
                        'credit_passed': 0,
                        'contracts_signed': contracts_signed,
                        'cars_delivered': first_cars_delivered,
                        'leads': leads,
                        'sqls': sqls,
                        'proposals': proposals
                    })
        
        return pd.DataFrame(data)
    
    def generate_customer_data(self) -> pd.DataFrame:
        """Generate customer/account data"""
        data = []
        
        for date in self.date_range:
            for bu in self.business_units:
                # Get new subscriptions for this date/BU
                base_new_subs = np.random.poisson(45 if bu == "B2C" else 15)
                
                # B2C: roughly 1:1 ratio (one customer per subscription)
                # B2B: multiple subscriptions per account (fleet accounts)
                if bu == "B2C":
                    new_customers = max(1, int(base_new_subs * np.random.normal(0.95, 0.1)))
                else:  # B2B
                    # B2B accounts typically have multiple subscriptions
                    new_customers = max(1, int(base_new_subs * np.random.normal(0.3, 0.1)))
                
                data.append({
                    'date': date,
                    'business_unit': bu,
                    'new_customers': new_customers,
                    'new_subscriptions': base_new_subs
                })
        
        return pd.DataFrame(data)
    
    def generate_retention_data(self) -> pd.DataFrame:
        """Generate retention and churn data"""
        data = []
        
        # Monthly churn rates (base rates that vary over time)
        base_churn_rates = {
            "B2C": {"customer": 0.08, "subscription": 0.12},  # 8% customer, 12% subscription monthly churn
            "B2B": {"customer": 0.04, "subscription": 0.06}   # Lower churn for B2B
        }
        
        # Generate monthly retention data
        for date in pd.date_range(start=self.start_date, end=self.end_date, freq='MS'):
            days_from_start = (date - self.start_date).days
            
            # Seasonal variation in churn (higher in winter)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (days_from_start + 180) / 365)
            
            for bu in self.business_units:
                bu_rates = base_churn_rates[bu]
                
                # Apply seasonal variation with some noise
                noise = np.random.normal(1.0, 0.2)
                customer_churn_rate = bu_rates["customer"] * seasonal_factor * noise
                subscription_churn_rate = bu_rates["subscription"] * seasonal_factor * noise
                
                # Ensure reasonable bounds
                customer_churn_rate = max(0.01, min(0.25, customer_churn_rate))
                subscription_churn_rate = max(0.02, min(0.35, subscription_churn_rate))
                
                # Early churn (0-3 months) - higher than overall
                early_churn_rate = subscription_churn_rate * 1.8
                early_churn_rate = max(0.05, min(0.5, early_churn_rate))
                
                # Renewal rate (complement of churn for term completers)
                renewal_rate = 1 - (subscription_churn_rate * 0.7)  # Better retention at term end
                renewal_rate = max(0.6, min(0.95, renewal_rate))
                
                # MRR movement components (expansion is positive, contraction/churn negative)
                base_mrr = 350000 if bu == "B2C" else 120000
                
                expansion_mrr = base_mrr * np.random.normal(0.15, 0.05)  # 15% expansion
                contraction_mrr = base_mrr * np.random.normal(-0.08, 0.03)  # 8% contraction
                churn_mrr = base_mrr * (-subscription_churn_rate)
                
                expansion_mrr = max(0, expansion_mrr)
                contraction_mrr = min(0, contraction_mrr)
                churn_mrr = min(0, churn_mrr)
                
                # Expansion MRR share
                total_movement = abs(expansion_mrr) + abs(contraction_mrr) + abs(churn_mrr)
                expansion_share = abs(expansion_mrr) / total_movement if total_movement > 0 else 0
                
                data.append({
                    'date': date,
                    'business_unit': bu,
                    'customer_churn_rate': customer_churn_rate,
                    'subscription_churn_rate': subscription_churn_rate,
                    'early_churn_rate': early_churn_rate,
                    'renewal_rate': renewal_rate,
                    'expansion_mrr_share': expansion_share,
                    'expansion_mrr': expansion_mrr,
                    'contraction_mrr': contraction_mrr,
                    'churn_mrr': churn_mrr
                })
        
        return pd.DataFrame(data)
    
    def generate_cohort_data(self) -> pd.DataFrame:
        """Generate cohort retention data"""
        data = []
        
        # Generate cohorts (monthly cohorts)
        cohort_months = pd.date_range(start=self.start_date, end=self.end_date - timedelta(days=90), freq='MS')
        
        for cohort_start in cohort_months:
            for bu in self.business_units:
                # Initial cohort size
                if bu == "B2C":
                    cohort_size = np.random.poisson(1200)
                else:
                    cohort_size = np.random.poisson(400)
                
                # Generate retention rates for each month since start
                for months_since_start in range(13):  # 0 to 12 months
                    if months_since_start == 0:
                        retention_rate = 1.0  # 100% at start
                    else:
                        # Decay function - retention decreases over time
                        if bu == "B2C":
                            # B2C has steeper decay
                            base_retention = 0.92 ** months_since_start  # ~8% monthly decay
                        else:
                            # B2B has better retention
                            base_retention = 0.95 ** months_since_start  # ~5% monthly decay
                        
                        # Add some noise and seasonal effects
                        noise = np.random.normal(1.0, 0.1)
                        retention_rate = base_retention * noise
                        retention_rate = max(0.1, min(1.0, retention_rate))
                    
                    # Calculate period (month) being measured
                    period_date = cohort_start + pd.DateOffset(months=months_since_start)
                    
                    # Only include if within our data range
                    if period_date <= self.end_date:
                        data.append({
                            'date': period_date,  # Use 'date' for consistency with filter function
                            'cohort_month': cohort_start,
                            'period_month': period_date,
                            'months_since_start': months_since_start,
                            'business_unit': bu,
                            'cohort_size': cohort_size,
                            'retention_rate': retention_rate,
                            'retained_count': int(cohort_size * retention_rate)
                        })
        
        return pd.DataFrame(data)
    
    def generate_b2b_account_data(self) -> pd.DataFrame:
        """Generate B2B account expansion data"""
        data = []
        
        # Generate account data for each month
        for date in pd.date_range(start=self.start_date, end=self.end_date, freq='MS'):
            # Simulate B2B accounts with different fleet sizes
            account_distributions = {
                '1-3 cars': np.random.poisson(150),
                '4-10 cars': np.random.poisson(80),
                '11-25 cars': np.random.poisson(30),
                '26+ cars': np.random.poisson(12)
            }
            
            for fleet_size, account_count in account_distributions.items():
                # Calculate average cars per account for each bucket
                if fleet_size == '1-3 cars':
                    avg_cars = np.random.normal(2.2, 0.5)
                elif fleet_size == '4-10 cars':
                    avg_cars = np.random.normal(6.8, 1.5)
                elif fleet_size == '11-25 cars':
                    avg_cars = np.random.normal(16.5, 3.2)
                else:  # 26+ cars
                    avg_cars = np.random.normal(38.2, 8.5)
                
                avg_cars = max(1, avg_cars)
                
                data.append({
                    'date': date,
                    'business_unit': 'B2B',  # Add business_unit for filtering consistency
                    'fleet_size_bucket': fleet_size,
                    'account_count': account_count,
                    'avg_cars_per_account': avg_cars,
                    'total_cars': int(account_count * avg_cars)
                })
        
        return pd.DataFrame(data)

    def get_targets(self) -> Dict[str, float]:
        """Return target values for KPIs"""
        return {
            'active_subscriptions': 12000,
            'mrr': 520000,
            'net_new_subscriptions': 800,  # Monthly target
            'nrr': 0.95,  # 95% Net Revenue Retention
            'blended_cac': 200,
            'fleet_utilization': 0.88
        }