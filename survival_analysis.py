import sys
import inspect
import scipy.integrate

# --- MONKEY PATCHES (Must be at the absolute top!) ---
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
if not hasattr(scipy.integrate, 'trapz'):
    scipy.integrate.trapz = scipy.integrate.trapezoid
# -----------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import os

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

class ChurnSurvivalAnalyzer:
    """
    Applies Survival Analysis (Kaplan-Meier) to non-contractual E-Commerce data.
    Uses a heuristic to define "churn" and handles right-censoring.
    """
    def __init__(self, data_dir: str = 'data/raw', output_dir: str = 'plots'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.churn_threshold_days = 90 
        self.observation_end = pd.to_datetime('2023-12-31')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        print("Loading data for Survival Analysis... (This might take 10-20 seconds. Do not abort!)")
        customers = pd.read_csv(f"{self.data_dir}/customers.csv", parse_dates=['join_date'])
        transactions = pd.read_csv(f"{self.data_dir}/transactions.csv", parse_dates=['transaction_date'])
        
        # 1. Get first and last purchase date per customer
        tx_stats = transactions.groupby('customer_id')['transaction_date'].agg(['min', 'max']).reset_index()
        tx_stats.columns = ['customer_id', 'first_purchase', 'last_purchase']
        
        self.df = pd.merge(customers, tx_stats, on='customer_id')
        
        # 2. Define the Churn Event (E)
        self.df['days_since_last_txn'] = (self.observation_end - self.df['last_purchase']).dt.days
        self.df['churn_event'] = (self.df['days_since_last_txn'] > self.churn_threshold_days).astype(int)
        
        # 3. Define the Duration (T)
        self.df['duration_days'] = np.where(
            self.df['churn_event'] == 1,
            (self.df['last_purchase'] - self.df['first_purchase']).dt.days,
            (self.observation_end - self.df['first_purchase']).dt.days
        )
        
        # Explicitly cast to float to prevent Pandas 2.0+ LossySetitemError
        self.df['duration_days'] = self.df['duration_days'].astype(float)
        self.df.loc[self.df['duration_days'] == 0, 'duration_days'] = 0.5

    def plot_overall_survival_curve(self):
        print("Calculating overall Kaplan-Meier curve...")
        kmf = KaplanMeierFitter()
        
        T = self.df['duration_days']
        E = self.df['churn_event']
        
        kmf.fit(T, event_observed=E, label='All Customers')
        
        plt.figure(figsize=(10, 6))
        kmf.plot_survival_function(ci_show=True, linewidth=2.5)
        
        plt.title('Kaplan-Meier Survival Curve (Overall Customer Base)', fontsize=16)
        plt.ylabel('Probability of Customer Remaining Active', fontsize=12)
        plt.xlabel('Days Since First Purchase', fontsize=12)
        
        plt.axvline(x=365, color='grey', linestyle='--', alpha=0.7)
        plt.text(375, 0.5, 'Year 1 Cliff', color='grey')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/survival_curve_overall.png", dpi=300)
        plt.close()

    def plot_survival_by_channel(self):
        print("Calculating Kaplan-Meier curves by Acquisition Channel...")
        plt.figure(figsize=(12, 7))
        ax = plt.subplot(111)
        
        channels = self.df['acquisition_channel'].unique()
        
        for channel in channels:
            mask = self.df['acquisition_channel'] == channel
            kmf = KaplanMeierFitter()
            
            T = self.df[mask]['duration_days']
            E = self.df[mask]['churn_event']
            
            kmf.fit(T, event_observed=E, label=channel)
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2)
            
        plt.title('Survival Probability by Acquisition Channel', fontsize=16)
        plt.ylabel('Probability of Customer Remaining Active', fontsize=12)
        plt.xlabel('Days Since First Purchase', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/survival_curve_by_channel.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    analyzer = ChurnSurvivalAnalyzer()
    analyzer.load_and_preprocess_data()
    analyzer.plot_overall_survival_curve()
    analyzer.plot_survival_by_channel()
    print(f"Survival Analysis complete! Check the '{analyzer.output_dir}' folder for the KM curves.")