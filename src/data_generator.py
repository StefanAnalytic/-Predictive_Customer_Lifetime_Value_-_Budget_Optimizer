import pandas as pd
import numpy as np
import uuid
from datetime import timedelta
from typing import Tuple, Dict
import os

# Set random seed for reproducibility (Crucial for a portfolio piece!)
np.random.seed(42)

class ECommerceDataGenerator:
    """
    Generates synthetic but highly realistic B2C E-Commerce transaction data.
    Simulates a non-contractual setting (customers can churn silently).
    """
    
    def __init__(self, start_date: str = '2022-01-01', end_date: str = '2023-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.days_active = (self.end_date - self.start_date).days
        
        # Define acquisition channels with latent behavioral characteristics
        # Interview-Tipp: Erkläre hier, dass echte Kanäle unterschiedliche Kohorten-Qualitäten haben.
        # Organic bringt treue Kunden (niedriges Churn-Risiko), Paid Social bringt Impulskäufer.
        self.channels: Dict[str, dict] = {
            'Organic Search': {'cac_mean': 0, 'purchase_rate': 0.05, 'churn_prob': 0.01, 'aov_mean': 60, 'daily_volume': 20},
            'Paid Social': {'cac_mean': 25, 'purchase_rate': 0.08, 'churn_prob': 0.05, 'aov_mean': 45, 'daily_volume': 50},
            'Google Ads': {'cac_mean': 15, 'purchase_rate': 0.06, 'churn_prob': 0.02, 'aov_mean': 55, 'daily_volume': 35},
            'Influencer': {'cac_mean': 40, 'purchase_rate': 0.12, 'churn_prob': 0.08, 'aov_mean': 70, 'daily_volume': 15}
        }

    def generate_customers(self) -> pd.DataFrame:
        """Generates customer signups over time with baseline latent parameters."""
        print("Generating customers...")
        customers = []
        
        dates = pd.date_range(self.start_date, self.end_date)
        
        for date in dates:
            # Simulate seasonality: Q4 (Oct-Dec) has 50% more volume
            seasonality_multiplier = 1.5 if date.month in [10, 11, 12] else 1.0
            
            for channel, props in self.channels.items():
                # Poisson distribution for daily signups adds realistic variance
                daily_signups = np.random.poisson(props['daily_volume'] * seasonality_multiplier)
                
                for _ in range(daily_signups):
                    customers.append({
                        # BUGFIX: Full UUID to avoid birthday paradox collisions
                        'customer_id': str(uuid.uuid4()),
                        'join_date': date,
                        'acquisition_channel': channel,
                        # Latent variables for DGP (Data Generating Process)
                        '_latent_purchase_rate': np.random.gamma(shape=2.0, scale=props['purchase_rate']/2.0),
                        '_latent_churn_prob': np.clip(np.random.normal(props['churn_prob'], 0.01), 0.001, 0.2),
                        '_aov': max(10, np.random.normal(props['aov_mean'], props['aov_mean']*0.2))
                    })
                    
        return pd.DataFrame(customers)

    def generate_transactions_and_spend(self, df_customers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulates transactions based on the "Buy 'Til You Die" (BG/NBD) philosophy.
        Also aggregates marketing spend per channel.
        """
        print("Generating transactions and marketing spend...")
        transactions = []
        marketing_spend = []
        
        # 1. Marketing Spend Generation
        dates = pd.date_range(self.start_date, self.end_date)
        for date in dates:
            for channel, props in self.channels.items():
                if props['cac_mean'] > 0:
                    # Daily spend is roughly CAC * expected volume + noise
                    expected_signups = props['daily_volume'] * (1.5 if date.month in [10, 11, 12] else 1.0)
                    daily_spend = expected_signups * props['cac_mean'] * np.random.normal(1.0, 0.1)
                    marketing_spend.append({
                        'date': date,
                        'channel': channel,
                        'spend': round(daily_spend, 2)
                    })
                    
        # 2. Transaction Generation
        # Interview-Tipp: Erkläre hier, warum wir für LTV nicht einfach (Durchschnittlicher Bestellwert * Kaufhäufigkeit) rechnen.
        # Im E-Commerce (non-contractual) kündigen Kunden nicht explizit. Wir müssen modellieren, 
        # dass ein Kunde nach jedem Kauf mit einer bestimmten Wahrscheinlichkeit inaktiv wird.
        
        for _, customer in df_customers.iterrows():
            current_date = customer['join_date']
            
            # Everyone makes a first purchase on join date (simplification for acquisition)
            transactions.append({
                # BUGFIX: Full UUID
                'transaction_id': f"TXN-{str(uuid.uuid4())}",
                'customer_id': customer['customer_id'],
                'transaction_date': current_date,
                'revenue': round(np.random.normal(customer['_aov'], 5), 2)
            })
            
            is_alive = True
            
            while is_alive and current_date <= self.end_date:
                # Time until next purchase (Exponential distribution based on latent rate)
                days_to_next_purchase = int(np.random.exponential(1 / customer['_latent_purchase_rate']))
                current_date += timedelta(days=days_to_next_purchase)
                
                if current_date > self.end_date:
                    break
                    
                transactions.append({
                    # BUGFIX: Full UUID
                    'transaction_id': f"TXN-{str(uuid.uuid4())}",
                    'customer_id': customer['customer_id'],
                    'transaction_date': current_date,
                    'revenue': round(max(5, np.random.normal(customer['_aov'], 10)), 2)
                })
                
                # BG/NBD Assumption: After each purchase, there's a chance the customer drops out
                if np.random.random() < customer['_latent_churn_prob']:
                    is_alive = False

        df_transactions = pd.DataFrame(transactions)
        df_spend = pd.DataFrame(marketing_spend)
        
        # Clean up latent columns from customers df so we don't leak data into later modeling!
        cols_to_drop = [c for c in df_customers.columns if c.startswith('_')]
        df_customers = df_customers.drop(columns=cols_to_drop)
        
        return df_customers, df_transactions, df_spend

if __name__ == "__main__":
    # Ensure directories exist before saving
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    generator = ECommerceDataGenerator(start_date='2022-01-01', end_date='2023-12-31')
    
    df_cust = generator.generate_customers()
    df_cust, df_txn, df_spend = generator.generate_transactions_and_spend(df_cust)
    
    # Save to CSV
    df_cust.to_csv('data/raw/customers.csv', index=False)
    df_txn.to_csv('data/raw/transactions.csv', index=False)
    df_spend.to_csv('data/raw/marketing_spends.csv', index=False)
    
    print(f"Data generation complete!")
    print(f"Total Customers: {len(df_cust)}")
    print(f"Total Transactions: {len(df_txn)}")
    print(f"Total Spend Records: {len(df_spend)}")