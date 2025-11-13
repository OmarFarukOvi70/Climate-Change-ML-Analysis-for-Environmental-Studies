# Climate Data Analysis Project
# Created by: Data Science Team
# Last updated: 2024-12-05
# Project: Climate Change ML Analysis for Environmental Studies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


# TODO: Add functionality for real-time data updates
# NOTE: This was originally built for academic research, might need optimization for production

# Configuration - keeping it simple for now
class ClimateConfig:
    """Config settings for the climate analysis project"""

    def __init__(self):
        self.start_year = 1850  # Data gets sketchy before this
        self.test_size = 0.3
        self.random_state = 42
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F7CAC']
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


config = ClimateConfig()

# Plot setup - these settings work well for reports
plt.style.use('default')
sns.set_palette(config.colors)
plt.rcParams['figure.figsize'] = (12, 8)

print("Starting Climate Data Analysis...")
print("=" * 70)


# ============================================================================
# DATA LOADING - messy but gets the job done
# ============================================================================

class DataLoader:
    """Handles loading climate datasets - needs refactoring later"""

    def __init__(self):
        self.files_found = []
        self.data_cache = {}  # cache for faster reloading

    def find_data_files(self):
        """Look for our climate data files"""
        expected_files = [
            'GlobalTemperatures.csv',
            'GlobalLandTemperaturesByCountry.csv',
            'GlobalLandTemperaturesByState.csv',
            'GlobalLandTemperaturesByMajorCity.csv',
            'GlobalLandTemperaturesByCity.csv'
        ]

        print("Checking for data files...")

        for file in expected_files:
            try:
                # Quick check if file exists and is readable
                temp_df = pd.read_csv(file, nrows=2)
                self.files_found.append(file)
                print(f"✓ Found {file}")
            except FileNotFoundError:
                print(f"✗ Missing {file} - might need to download")
            except Exception as e:
                print(f"! Problem with {file}: {str(e)[:50]}...")

        print(f"Found {len(self.files_found)}/{len(expected_files)} files")
        return self.files_found

    def load_main_data(self):
        """Load the main temperature dataset"""
        print("\nLoading global temperature data...")

        try:
            # Using parse_dates because the date format is consistent
            df = pd.read_csv('GlobalTemperatures.csv', parse_dates=['dt'])
            print(f"Loaded {len(df):,} temperature records")

            # Clean it up
            df = self.clean_global_data(df)
            self.data_cache['global'] = df
            return df

        except Exception as e:
            print(f"Failed to load data: {e}")
            # TODO: Add fallback to sample data
            return None

    def load_country_data(self):
        """Load country data if available"""
        if 'GlobalLandTemperaturesByCountry.csv' not in self.files_found:
            return None

        print("\nLoading country data...")

        try:
            df = pd.read_csv('GlobalLandTemperaturesByCountry.csv', parse_dates=['dt'])
            df = self.clean_country_data(df)
            self.data_cache['country'] = df
            print(f"Loaded {len(df):,} country records")
            return df

        except Exception as e:
            print(f"Country data issue: {e}")
            return None

    def clean_global_data(self, df):
        """Clean up the global temperature data"""
        # Extract time features - this is useful for seasonal analysis
        df['year'] = df['dt'].dt.year
        df['month'] = df['dt'].dt.month
        df['decade'] = (df['year'] // 10) * 10

        # Filter out old data that's less reliable
        df = df[df['year'] >= config.start_year].copy()

        # Remove rows with missing temperature data
        df = df.dropna(subset=['LandAverageTemperature', 'LandAverageTemperatureUncertainty'])

        # Quick feature engineering
        if all(col in df.columns for col in ['LandMaxTemperature', 'LandMinTemperature']):
            df['TempRange'] = df['LandMaxTemperature'] - df['LandMinTemperature']

        print(f"Cleaned data: {len(df):,} records from {df['year'].min()} to {df['year'].max()}")
        return df

    def clean_country_data(self, df):
        """Basic cleaning for country data"""
        df['year'] = df['dt'].dt.year
        df = df[df['year'] >= config.start_year]
        df = df.dropna(subset=['AverageTemperature'])  # Remove missing temps
        return df


# ============================================================================
# EXPLORATORY ANALYSIS - this is where the fun begins
# ============================================================================

class ClimateExplorer:
    """Explore the climate data and find interesting patterns"""

    def __init__(self, global_df, country_df=None):
        self.global_data = global_df
        self.country_data = country_df
        self.findings = {}  # store interesting insights

    def run_full_analysis(self):
        """Run all the exploratory analysis"""
        print("\nStarting exploratory analysis...")
        print("=" * 50)

        self.analyze_trends_over_time()
        self.look_at_seasonal_patterns()
        self.check_distributions()
        self.examine_correlations()

        if self.country_data is not None:
            self.compare_countries()

        self.summarize_findings()
        return self.findings

    def analyze_trends_over_time(self):
        """Look at how temperatures are changing over time"""
        print("\nAnalyzing temperature trends...")

        # Group by year for the big picture
        yearly_avg = self.global_data.groupby('year')['LandAverageTemperature'].mean()

        # Calculate the warming trend - this is the important part!
        trend_coef = np.polyfit(yearly_avg.index, yearly_avg.values, 1)[0]
        warming_per_century = trend_coef * 100

        # Store for later
        self.findings['warming_rate'] = warming_per_century
        self.findings['starting_temp'] = yearly_avg.iloc[0]
        self.findings['current_temp'] = yearly_avg.iloc[-1]
        self.findings['total_change'] = yearly_avg.iloc[-1] - yearly_avg.iloc[0]

        # Create the visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Main temperature plot
        axes[0, 0].plot(yearly_avg.index, yearly_avg.values,
                        linewidth=2, color='#2E86AB', alpha=0.8)
        axes[0, 0].set_title('Global Temperature Trend\n(1850 to Present)',
                             fontsize=14, fontweight='bold', pad=15)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Average Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(yearly_avg.index, yearly_avg.values, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(yearly_avg.index, p(yearly_avg.index), '--',
                        color='red', linewidth=2,
                        label=f'Trend: {warming_per_century:+.2f}°C/century')
        axes[0, 0].legend()

        # Decadal view
        decade_avg = self.global_data.groupby('decade')['LandAverageTemperature'].mean()
        axes[0, 1].bar(decade_avg.index, decade_avg.values,
                       color='#F18F01', alpha=0.7)
        axes[0, 1].set_title('Temperature by Decade', fontsize=14, fontweight='bold', pad=15)
        axes[0, 1].set_xlabel('Decade')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Measurement uncertainty
        yearly_uncertainty = self.global_data.groupby('year')['LandAverageTemperatureUncertainty'].mean()
        axes[1, 0].plot(yearly_uncertainty.index, yearly_uncertainty.values,
                        color='#C73E1D', linewidth=2)
        axes[1, 0].set_title('Measurement Uncertainty Over Time',
                             fontsize=14, fontweight='bold', pad=15)
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Uncertainty (°C)')
        axes[1, 0].grid(True, alpha=0.3)

        # Recent trend (last 50 years)
        recent_data = yearly_avg[yearly_avg.index >= 1970]
        if len(recent_data) > 1:
            recent_trend = np.polyfit(recent_data.index, recent_data.values, 1)[0] * 100
            axes[1, 1].plot(recent_data.index, recent_data.values,
                            linewidth=2, color='#A23B72')
            axes[1, 1].set_title(f'Recent Warming (1970+)\n{recent_trend:+.2f}°C/century',
                                 fontsize=14, fontweight='bold', pad=15)
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Temperature (°C)')
            axes[1, 1].grid(True, alpha=0.3)

            self.findings['recent_warming'] = recent_trend

        plt.tight_layout()
        plt.show()

    def look_at_seasonal_patterns(self):
        """Check seasonal temperature patterns"""
        print("\nChecking seasonal patterns...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Monthly averages
        monthly_avg = self.global_data.groupby('month')['LandAverageTemperature'].mean()

        axes[0].plot(config.months, monthly_avg.values, marker='o',
                     linewidth=2, markersize=6, color='#2E86AB')
        axes[0].fill_between(config.months, monthly_avg.values, alpha=0.3, color='#2E86AB')
        axes[0].set_title('Seasonal Temperature Pattern', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].grid(True, alpha=0.3)

        # Temperature range over time
        seasonal_range = self.global_data.groupby('year')['LandAverageTemperature'].agg(['max', 'min'])
        seasonal_range['range'] = seasonal_range['max'] - seasonal_range['min']

        axes[1].plot(seasonal_range.index, seasonal_range['range'],
                     linewidth=2, color='#A23B72')
        axes[1].set_title('Seasonal Temperature Range Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Temperature Range (°C)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.findings['avg_seasonal_range'] = seasonal_range['range'].mean()

    def check_distributions(self):
        """Look at temperature distributions"""
        print("\nAnalyzing temperature distributions...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of temperatures
        axes[0].hist(self.global_data['LandAverageTemperature'], bins=40,
                     alpha=0.7, color='#2E86AB', edgecolor='black')
        mean_temp = self.global_data['LandAverageTemperature'].mean()
        axes[0].axvline(mean_temp, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_temp:.2f}°C')
        axes[0].set_title('Temperature Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Monthly box plots
        monthly_data = [self.global_data[self.global_data['month'] == month]['LandAverageTemperature']
                        for month in range(1, 13)]
        axes[1].boxplot(monthly_data, labels=config.months)
        axes[1].set_title('Monthly Temperature Distributions', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Temperature (°C)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def examine_correlations(self):
        """Check relationships between variables"""
        print("\nExamining correlations...")

        # Select numeric columns
        numeric_cols = ['LandAverageTemperature', 'LandAverageTemperatureUncertainty',
                        'year', 'month']

        # Add optional columns if they exist
        extra_cols = ['LandMaxTemperature', 'LandMinTemperature', 'TempRange']
        for col in extra_cols:
            if col in self.global_data.columns:
                numeric_cols.append(col)

        corr_matrix = self.global_data[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                    center=0, fmt='.2f', square=True)
        plt.title('Feature Correlations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        self.findings['temp_year_corr'] = corr_matrix.loc['LandAverageTemperature', 'year']

    def compare_countries(self):
        """Compare temperatures across countries"""
        if self.country_data is None:
            return

        print("\nComparing countries...")

        # Get countries with most data
        country_counts = self.country_data['Country'].value_counts()
        top_countries = country_counts.head(8).index.tolist()

        # Use recent data
        recent_data = self.country_data[self.country_data['year'] >= 2000]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot trends for major countries
        for country in top_countries[:4]:  # Just top 4 to avoid clutter
            country_temps = recent_data[recent_data['Country'] == country]
            if len(country_temps) > 0:
                yearly_avg = country_temps.groupby('year')['AverageTemperature'].mean()
                axes[0].plot(yearly_avg.index, yearly_avg.values,
                             label=country, linewidth=2, marker='o', markersize=3)

        axes[0].set_title('Country Temperature Trends (2000+)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Average Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Current temperatures by country
        current_temps = recent_data.groupby('Country')['AverageTemperature'].mean().sort_values(ascending=False)
        current_temps.head(8).plot(kind='bar', ax=axes[1], color='#3F7CAC')
        axes[1].set_title('Average Temperatures by Country (2000+)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Country')
        axes[1].set_ylabel('Temperature (°C)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def summarize_findings(self):
        """Print out key findings"""
        print("\n" + "=" * 50)
        print("KEY FINDINGS")
        print("=" * 50)

        findings = self.findings
        yearly_avg = self.global_data.groupby('year')['LandAverageTemperature'].mean()

        print(f"• Warming rate: {findings.get('warming_rate', 0):+.2f}°C per century")
        print(f"• Total warming since 1850: {findings.get('total_change', 0):.2f}°C")
        if 'recent_warming' in findings:
            print(f"• Recent acceleration: {findings['recent_warming']:+.2f}°C/century since 1970")
        print(f"• Average temperature: {yearly_avg.mean():.2f}°C")
        print(f"• Seasonal range: {findings.get('avg_seasonal_range', 0):.2f}°C")
        print(f"• Temperature-Year correlation: {findings.get('temp_year_corr', 0):.3f}")


# ============================================================================
# FEATURE ENGINEERING - making the data ready for ML
# ============================================================================

class FeatureBuilder:
    """Build features for machine learning models"""

    def __init__(self, df):
        self.df = df.copy()
        self.features = []

    def add_time_features(self):
        """Add time-based features"""
        print("Adding time features...")

        # Simple time index
        self.df['time_index'] = np.arange(len(self.df))

        # Cyclical features for seasons
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        self.features.extend(['time_index', 'month_sin', 'month_cos'])

    def add_lags(self, lags=[1, 12, 60]):
        """Add lagged temperature features"""
        print("Adding lag features...")

        for lag in lags:
            self.df[f'lag_{lag}'] = self.df['LandAverageTemperature'].shift(lag)
            self.features.append(f'lag_{lag}')

    def add_rolling_stats(self, windows=[12, 60]):
        """Add rolling statistics"""
        print("Adding rolling features...")

        for window in windows:
            self.df[f'roll_mean_{window}'] = self.df['LandAverageTemperature'].rolling(window, min_periods=1).mean()
            self.df[f'roll_std_{window}'] = self.df['LandAverageTemperature'].rolling(window, min_periods=1).std()

            self.features.extend([f'roll_mean_{window}', f'roll_std_{window}'])

    def add_basic_features(self):
        """Add basic climate features"""
        print("Adding basic features...")

        # Keep uncertainty as a feature
        if 'LandAverageTemperatureUncertainty' in self.df.columns:
            self.features.append('LandAverageTemperatureUncertainty')

        # Year captures long-term trend
        self.features.append('year')

        # Month for seasonality
        self.features.append('month')

    def build_features(self):
        """Build all features"""
        self.add_time_features()
        self.add_lags()
        self.add_rolling_stats()
        self.add_basic_features()

        # Clean up NaN values from lag features
        self.df = self.df.dropna()

        X = self.df[self.features]
        y = self.df['LandAverageTemperature']

        print(f"Feature engineering complete - {len(self.features)} features")
        print(f"Final dataset: {X.shape}")

        return X, y, self.features


# ============================================================================
# MACHINE LEARNING MODELS - the predictive part
# ============================================================================

class ModelRunner:
    """Train and evaluate ML models for temperature prediction"""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {}
        self.predictions = {}
        self.results = {}

    def train_all_models(self):
        """Train multiple models for comparison"""
        print("\nTraining machine learning models...")
        print("=" * 40)

        # Define our models
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
        }

        # Split data - using time series split (no shuffling)
        split_point = int(len(self.X) * (1 - config.test_size))
        X_train, X_test = self.X.iloc[:split_point], self.X.iloc[split_point:]
        y_train, y_test = self.y.iloc[:split_point], self.y.iloc[split_point:]

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Train each model
        for name, model in models.items():
            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            self.models[name] = model
            self.predictions[name] = y_pred

            # Calculate performance
            self.results[name] = self.calc_metrics(y_test, y_pred, name)

        return X_train, X_test, y_train, y_test

    def calc_metrics(self, y_true, y_pred, model_name):
        """Calculate model performance metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"{model_name} results:")
        print(f"  MAE:  {mae:.4f}°C")
        print(f"  RMSE: {rmse:.4f}°C")
        print(f"  R²:   {r2:.4f}")

        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    def evaluate_models(self, X_test, y_test):
        """Create evaluation visualizations"""
        print("\nCreating evaluation plots...")

        self.plot_predictions(X_test, y_test)
        self.plot_residuals(X_test, y_test)
        self.plot_comparison()

        return self.results

    def plot_predictions(self, X_test, y_test):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            if idx >= 4:
                break

            axes[idx].scatter(y_test, y_pred, alpha=0.6, s=20, color=config.colors[idx])
            perfect = np.linspace(y_test.min(), y_test.max(), 100)
            axes[idx].plot(perfect, perfect, 'r--', alpha=0.8, linewidth=2)

            r2 = self.results[name]['R2']
            axes[idx].set_title(f'{name} (R²={r2:.3f})', fontweight='bold')
            axes[idx].set_xlabel('Actual Temperature (°C)')
            axes[idx].set_ylabel('Predicted Temperature (°C)')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_residuals(self, X_test, y_test):
        """Plot residual analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            if idx >= 4:
                break

            residuals = y_test - y_pred
            axes[idx].scatter(y_pred, residuals, alpha=0.6, s=20, color=config.colors[idx])
            axes[idx].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[idx].set_title(f'{name} - Residuals')
            axes[idx].set_xlabel('Predicted Temperature (°C)')
            axes[idx].set_ylabel('Residuals')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_comparison(self):
        """Compare model performance"""
        metrics_df = pd.DataFrame(self.results).T

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Error comparison
        metrics_df[['RMSE', 'MAE']].plot(kind='bar', ax=axes[0], color=['#2E86AB', '#A23B72'])
        axes[0].set_title('Model Errors (Lower is Better)')
        axes[0].set_ylabel('Error (°C)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # R² comparison
        metrics_df['R2'].plot(kind='bar', ax=axes[1], color='#F18F01')
        axes[1].set_title('R² Scores (Higher is Better)')
        axes[1].set_ylabel('R² Score')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ============================================================================
# MODEL OPTIMIZATION - fine-tuning
# ============================================================================

class ModelTuner:
    """Optimize model hyperparameters"""

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.best_models = {}

    def tune_random_forest(self):
        """Optimize Random Forest parameters"""
        print("\nTuning Random Forest...")

        # Simpler grid for faster tuning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5]
        }

        rf = RandomForestRegressor(random_state=42)

        # Use time series split
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {-grid_search.best_score_:.4f} MSE")

        self.best_models['RF'] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def show_feature_importance(self, model, feature_names):
        """Display feature importance"""
        print("\nFeature importance analysis...")

        if hasattr(model, 'feature_importances_'):
            importance_data = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_data.head(12), x='importance', y='feature', palette='viridis')
            plt.title('Top Feature Importances', fontsize=16, fontweight='bold')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()

            print("\nMost important features:")
            for idx, row in importance_data.head(8).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

            return importance_data

        return None


# ============================================================================
# MAIN EXECUTION - putting it all together
# ============================================================================

def main():
    """Run the complete analysis"""
    print("Climate Change Analysis - Starting...")
    print("=" * 60)

    try:
        # Step 1: Load data
        print("\n[1/6] Loading data...")
        loader = DataLoader()
        available_files = loader.find_data_files()

        main_data = loader.load_main_data()
        if main_data is None:
            print("No data available - stopping.")
            return

        country_data = loader.load_country_data()

        # Step 2: Exploratory analysis
        print("\n[2/6] Exploratory analysis...")
        explorer = ClimateExplorer(main_data, country_data)
        insights = explorer.run_full_analysis()

        # Step 3: Feature engineering
        print("\n[3/6] Feature engineering...")
        feature_builder = FeatureBuilder(main_data)
        X, y, feature_names = feature_builder.build_features()

        # Step 4: Model training
        print("\n[4/6] Training models...")
        model_runner = ModelRunner(X, y)
        X_train, X_test, y_train, y_test = model_runner.train_all_models()
        results = model_runner.evaluate_models(X_test, y_test)

        # Step 5: Model optimization
        print("\n[5/6] Optimizing models...")
        tuner = ModelTuner(X_train, y_train)
        best_rf = tuner.tune_random_forest()

        # Initialize top_predictors as empty list
        top_predictors = []

        # Get feature importance if available
        importance_data = tuner.show_feature_importance(best_rf, feature_names)
        if importance_data is not None:
            top_predictors = importance_data.head(3)['feature'].tolist()
            print(f"\nKey predictors:")
            for i, feature in enumerate(top_predictors, 1):
                print(f"  {i}. {feature}")

        # Test optimized model
        y_pred_optimized = best_rf.predict(X_test)
        optimized_results = model_runner.calc_metrics(y_test, y_pred_optimized, "Optimized RF")

        # Step 6: Final report
        print("\n[6/6] Generating final report...")
        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        # Performance summary
        print("\nModel performance:")
        performance_df = pd.DataFrame(results).T
        performance_df.loc['Optimized RF'] = optimized_results
        print(performance_df.round(4))

        # Best model - FIXED: Check which model actually performed best
        best_model_name = performance_df['R2'].idxmax()
        best_r2 = performance_df.loc[best_model_name, 'R2']
        best_rmse = performance_df.loc[best_model_name, 'RMSE']

        print(f"\nBest model: {best_model_name}")
        print(f"  R²: {best_r2:.4f}")
        print(f"  RMSE: {best_rmse:.4f}°C")

        # Climate insights
        print(f"\nClimate insights:")
        print(f"  • Warming trend: {insights.get('warming_rate', 0):+.2f}°C/century")
        print(f"  • Total change: {insights.get('total_change', 0):.2f}°C")
        if 'recent_warming' in insights:
            print(f"  • Recent rate: {insights['recent_warming']:+.2f}°C/century")

        # Key predictors - FIXED: Check if top_predictors list has items
        if top_predictors:
            print(f"\nKey predictors:")
            for i, feature in enumerate(top_predictors, 1):
                print(f"  {i}. {feature}")

        # Recommendations - FIXED: Use actual best model and check top_predictors
        print(f"\nNext steps:")
        print(f"  1. Use {best_model_name} for predictions")

        # FIXED: Check if top_predictors list has items before accessing index 0
        if top_predictors:
            print(f"  2. Focus on {top_predictors[0]}")
        else:
            print(f"  2. Focus on time-based patterns")

        print(f"  3. Consider adding more climate variables")
        print(f"  4. Validate with external datasets")

        print(f"\nAnalysis complete!")
        print(f"Processed {len(available_files)} data files")
        print(f"Trained {len(results) + 1} models")

    except Exception as e:
        print(f"Something went wrong: {e}")
        import traceback
        traceback.print_exc()


# Run the analysis
if __name__ == "__main__":
    # Check for required packages
    required = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn']

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("Missing packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
    else:
        main()