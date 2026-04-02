import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

class CohortAnalyzer:
    def __init__(self, data_dir: str = 'data/raw', output_dir: str = 'plots'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Loads and prepares the raw data for cohort analysis."""
        print("Loading data...")
        self.customers = pd.read_csv(f"{self.data_dir}/customers.csv", parse_dates=['join_date'])
        self.transactions = pd.read_csv(f"{self.data_dir}/transactions.csv", parse_dates=['transaction_date'])
        
        # Merge channel info into transactions
        self.df = self.transactions.merge(
            self.customers[['customer_id', 'join_date', 'acquisition_channel']], 
            on='customer_id', 
            how='left'
        )
        
        # Create Month periods for cohort grouping
        self.df['cohort_month'] = self.df['join_date'].dt.to_period('M')
        self.df['txn_month'] = self.df['transaction_date'].dt.to_period('M')
        
        # Calculate Cohort Index (Months since acquisition)
        # Interview-Tipp: Erkläre, dass der Cohort Index (0, 1, 2...) robuster ist als 
        # absolute Kalendermonate, um das Verhalten unterschiedlicher Kohorten zu vergleichen.
        self.df['cohort_index'] = (self.df['txn_month'] - self.df['cohort_month']).apply(lambda x: x.n)
        
    def build_retention_matrix(self):
        """Calculates the monthly retention rate per cohort."""
        print("Building Retention Matrix...")
        
        # Count unique active customers per cohort per month
        cohort_data = self.df.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()
        
        # Pivot to create the classic triangle
        cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customer_id')
        
        # BUGFIX: Divide by explicit Month 0 (Cohort Size) to get percentage
        # Do not use positional indexing like iloc[:, 0] in case negative months sneak in!
        cohort_sizes = cohort_pivot[0] 
        self.retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)
        
        return self.retention_matrix

    def plot_retention_heatmap(self):
        """Generates a Seaborn heatmap for the retention matrix."""
        print("Plotting Retention Heatmap...")
        plt.figure(figsize=(16, 10))
        
        # Plotting only the first 12 months for better readability
        # Also ensuring we only plot from index 0 onwards
        plot_cols = [col for col in self.retention_matrix.columns if 0 <= col <= 12]
        plot_data = self.retention_matrix[plot_cols]
        
        sns.heatmap(plot_data, annot=True, fmt='.1%', cmap='YlGnBu', vmin=0.0, vmax=0.5)
        plt.title('Monthly Customer Retention Rate by Cohort', fontsize=16)
        plt.ylabel('Cohort Month')
        plt.xlabel('Months Since Acquisition (Cohort Index)')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/retention_heatmap.png", dpi=300)
        plt.close()

    def plot_cumulative_ltv_by_channel(self):
        """
        Calculates and plots the historical Cumulative ARPU (LTV) by channel.
        Interview-Tipp: Hier zeigt sich der DGP (Data Generating Process) aus Phase 1!
        Paid Social startet vielleicht stark, aber Organic oder Influencer überholt langfristig.
        """
        print("Calculating Cumulative LTV by Channel...")
        
        # Get monthly revenue per cohort and channel
        revenue_data = self.df.groupby(['acquisition_channel', 'cohort_index'])['revenue'].sum().reset_index()
        
        # Get cohort sizes per channel to calculate per-user averages
        cohort_sizes = self.customers.groupby('acquisition_channel')['customer_id'].nunique().reset_index()
        cohort_sizes.rename(columns={'customer_id': 'cohort_size'}, inplace=True)
        
        merged = revenue_data.merge(cohort_sizes, on='acquisition_channel')
        merged['arpu_month'] = merged['revenue'] / merged['cohort_size']
        
        # Calculate cumulative sum for LTV
        # We also sort to ensure cumulative sum processes correctly
        merged = merged.sort_values(['acquisition_channel', 'cohort_index'])
        merged['cumulative_ltv'] = merged.groupby('acquisition_channel')['arpu_month'].cumsum()
        
        plt.figure(figsize=(12, 7))
        # Filter for positive cohort indices up to 12 months
        plot_df = merged[(merged['cohort_index'] >= 0) & (merged['cohort_index'] <= 12)]
        sns.lineplot(data=plot_df, 
                     x='cohort_index', y='cumulative_ltv', 
                     hue='acquisition_channel', marker='o', linewidth=2.5)
        
        plt.title('Historical Cumulative LTV by Acquisition Channel (First 12 Months)', fontsize=16)
        plt.ylabel('Cumulative Revenue per User ($)')
        plt.xlabel('Months Since Acquisition')
        plt.legend(title='Channel')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cumulative_ltv_by_channel.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    analyzer = CohortAnalyzer()
    analyzer.load_data()
    
    # 1. Retention Heatmap
    matrix = analyzer.build_retention_matrix()
    analyzer.plot_retention_heatmap()

    # 2. Cumulative LTV
    analyzer.plot_cumulative_ltv_by_channel()
    
    print(f"Analysis complete! Check the '{analyzer.output_dir}' folder for your charts.")