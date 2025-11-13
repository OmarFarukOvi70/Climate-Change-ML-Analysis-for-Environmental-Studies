# Climate Data Analysis Project
# Machine Learning Analysis of Global Temperature Trends
# Last updated: November 2025

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


class ClimateAnalysisConfig:
    """Configuration parameters for climate data analysis"""

    def __init__(self):
        self.start_year = 1850  # Historical data reliability threshold
        self.test_size = 0.3
        self.random_state = 42
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F7CAC']
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


config = ClimateAnalysisConfig()

# Visualization configuration
plt.style.use('default')
sns.set_palette(config.colors)
plt.rcParams['figure.figsize'] = (12, 8)

print("Climate Data Analysis Initialization")
print("=" * 70)


class ClimateDataLoader:
    """Data loading and preprocessing for climate datasets"""

    def __init__(self):
        self.available_files = []
        self.datasets = {}

    def discover_dataset_files(self):
        """Identify and validate available climate data files"""
        expected_files = [
            'GlobalTemperatures.csv',
            'GlobalLandTemperaturesByCountry.csv',
            'GlobalLandTemperaturesByState.csv',
            'GlobalLandTemperaturesByMajorCity.csv',
            'GlobalLandTemperaturesByCity.csv'
        ]

        print("Validating dataset availability...")

        for file in expected_files:
            try:
                pd.read_csv(file, nrows=2)
                self.available_files.append(file)
                print(f"Available: {file}")
            except FileNotFoundError:
                print(f"Not found: {file}")
            except Exception as e:
                print(f"Validation error {file}: {str(e)[:50]}")

        print(f"Dataset validation complete: {len(self.available_files)}/{len(expected_files)} files available")
        return self.available_files

    def load_global_temperature_data(self):
        """Load and preprocess primary global temperature dataset"""
        print("\nLoading global temperature dataset...")

        try:
            df = pd.read_csv('GlobalTemperatures.csv', parse_dates=['dt'])
            print(f"Successfully loaded {len(df):,} temperature records")

            df = self.preprocess_global_data(df)
            self.datasets['global'] = df
            return df

        except Exception as e:
            print(f"Data loading error: {e}")
            return None

    def load_country_temperature_data(self):
        """Load country-level temperature data if available"""
        if 'GlobalLandTemperaturesByCountry.csv' not in self.available_files:
            return None

        print("\nLoading country-level temperature data...")

        try:
            df = pd.read_csv('GlobalLandTemperaturesByCountry.csv', parse_dates=['dt'])
            df = self.preprocess_country_data(df)
            self.datasets['country'] = df
            print(f"Successfully loaded {len(df):,} country records")
            return df

        except Exception as e:
            print(f"Country data loading error: {e}")
            return None

    def preprocess_global_data(self, df):
        """Preprocess global temperature dataset"""
        # Temporal feature extraction
        df['year'] = df['dt'].dt.year
        df['month'] = df['dt'].dt.month
        df['decade'] = (df['year'] // 10) * 10

        # Data quality filtering
        df = df[df['year'] >= config.start_year].copy()

        # Handle missing values in critical columns
        critical_columns = ['LandAverageTemperature', 'LandAverageTemperatureUncertainty']
        df = df.dropna(subset=critical_columns)

        # Derived feature creation
        if all(col in df.columns for col in ['LandMaxTemperature', 'LandMinTemperature']):
            df['TemperatureRange'] = df['LandMaxTemperature'] - df['LandMinTemperature']

        print(f"Data preprocessing complete: {len(df):,} records from {df['year'].min()} to {df['year'].max()}")
        return df

    def preprocess_country_data(self, df):
        """Preprocess country-level temperature data"""
        df['year'] = df['dt'].dt.year
        df = df[df['year'] >= config.start_year]
        df = df.dropna(subset=['AverageTemperature'])
        return df


class ClimateDataAnalyzer:
    """Comprehensive exploratory analysis of climate data"""

    def __init__(self, global_df, country_df=None):
        self.global_data = global_df
        self.country_data = country_df
        self.analytical_insights = {}

    def execute_comprehensive_analysis(self):
        """Execute complete exploratory data analysis workflow"""
        print("\nInitiating comprehensive data analysis")
        print("=" * 50)

        self.analyze_temporal_trends()
        self.analyze_seasonal_patterns()
        self.analyze_temperature_distributions()
        self.analyze_feature_correlations()

        if self.country_data is not None:
            self.analyze_geographical_patterns()

        self.summarize_analytical_findings()
        return self.analytical_insights

    def analyze_temporal_trends(self):
        """Analyze temperature trends and changes over time"""
        print("\nAnalyzing temporal temperature trends...")

        yearly_average = self.global_data.groupby('year')['LandAverageTemperature'].mean()

        # Calculate warming trend using linear regression
        trend_coefficient = np.polyfit(yearly_average.index, yearly_average.values, 1)[0]
        warming_rate_century = trend_coefficient * 100

        # Store analytical insights
        self.analytical_insights['warming_rate'] = warming_rate_century
        self.analytical_insights['baseline_temperature'] = yearly_average.iloc[0]
        self.analytical_insights['current_temperature'] = yearly_average.iloc[-1]
        self.analytical_insights['total_temperature_change'] = yearly_average.iloc[-1] - yearly_average.iloc[0]

        # Create comprehensive temporal analysis visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Primary temperature trend visualization
        axes[0, 0].plot(yearly_average.index, yearly_average.values,
                        linewidth=2.5, color=config.colors[0], alpha=0.8)
        axes[0, 0].set_title('Global Land Temperature Trend Analysis (1850-Present)',
                             fontsize=14, fontweight='bold', pad=20)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Average Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)

        # Linear trend line
        regression_coefficients = np.polyfit(yearly_average.index, yearly_average.values, 1)
        trend_line = np.poly1d(regression_coefficients)
        axes[0, 0].plot(yearly_average.index, trend_line(yearly_average.index), '--',
                        color=config.colors[1], linewidth=2,
                        label=f'Linear Trend: {warming_rate_century:+.2f}°C/century')
        axes[0, 0].legend()

        # Decadal average analysis
        decadal_average = self.global_data.groupby('decade')['LandAverageTemperature'].mean()
        axes[0, 1].bar(decadal_average.index, decadal_average.values,
                       color=config.colors[2], alpha=0.7)
        axes[0, 1].set_title('Decadal Temperature Averages', fontsize=14, fontweight='bold', pad=20)
        axes[0, 1].set_xlabel('Decade')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Measurement uncertainty analysis
        yearly_uncertainty = self.global_data.groupby('year')['LandAverageTemperatureUncertainty'].mean()
        axes[1, 0].plot(yearly_uncertainty.index, yearly_uncertainty.values,
                        color=config.colors[3], linewidth=2)
        axes[1, 0].set_title('Temperature Measurement Uncertainty Timeline',
                             fontsize=14, fontweight='bold', pad=20)
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Measurement Uncertainty (°C)')
        axes[1, 0].grid(True, alpha=0.3)

        # Recent trend analysis (post-1970)
        recent_data = yearly_average[yearly_average.index >= 1970]
        if len(recent_data) > 1:
            recent_trend = np.polyfit(recent_data.index, recent_data.values, 1)[0] * 100
            axes[1, 1].plot(recent_data.index, recent_data.values,
                            linewidth=2, color=config.colors[4])
            axes[1, 1].set_title(f'Recent Warming Analysis (1970-Present)\n{recent_trend:+.2f}°C/century',
                                 fontsize=14, fontweight='bold', pad=20)
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Temperature (°C)')
            axes[1, 1].grid(True, alpha=0.3)

            self.analytical_insights['recent_warming_rate'] = recent_trend

        plt.tight_layout()
        plt.show()

    def analyze_seasonal_patterns(self):
        """Analyze seasonal temperature patterns and variations"""
        print("\nAnalyzing seasonal temperature patterns...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Monthly seasonal pattern analysis
        monthly_average = self.global_data.groupby('month')['LandAverageTemperature'].mean()

        axes[0].plot(config.months, monthly_average.values, marker='o',
                     linewidth=2.5, markersize=8, color=config.colors[0])
        axes[0].fill_between(config.months, monthly_average.values, alpha=0.3, color=config.colors[0])
        axes[0].set_title('Global Seasonal Temperature Pattern Analysis',
                          fontsize=14, fontweight='bold', pad=20)
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].grid(True, alpha=0.3)

        # Seasonal amplitude temporal analysis
        seasonal_analysis = self.global_data.groupby('year')['LandAverageTemperature'].agg(['max', 'min'])
        seasonal_analysis['range'] = seasonal_analysis['max'] - seasonal_analysis['min']

        axes[1].plot(seasonal_analysis.index, seasonal_analysis['range'],
                     linewidth=2, color=config.colors[1])
        axes[1].set_title('Seasonal Temperature Range Temporal Analysis',
                          fontsize=14, fontweight='bold', pad=20)
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Temperature Range (°C)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.analytical_insights['average_seasonal_amplitude'] = seasonal_analysis['range'].mean()

    def analyze_temperature_distributions(self):
        """Analyze temperature distribution characteristics"""
        print("\nAnalyzing temperature distributions...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Temperature distribution histogram
        axes[0].hist(self.global_data['LandAverageTemperature'], bins=40,
                     alpha=0.7, color=config.colors[0], edgecolor='black')
        mean_temperature = self.global_data['LandAverageTemperature'].mean()
        axes[0].axvline(mean_temperature, color=config.colors[1], linestyle='--', linewidth=2,
                        label=f'Mean Temperature: {mean_temperature:.2f}°C')
        axes[0].set_title('Global Temperature Distribution Analysis',
                          fontsize=14, fontweight='bold', pad=20)
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Monthly distribution analysis
        monthly_distributions = [self.global_data[self.global_data['month'] == month]['LandAverageTemperature']
                                for month in range(1, 13)]
        axes[1].boxplot(monthly_distributions, labels=config.months)
        axes[1].set_title('Monthly Temperature Distribution Analysis',
                          fontsize=14, fontweight='bold', pad=20)
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Temperature (°C)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def analyze_feature_correlations(self):
        """Analyze correlations between climate variables"""
        print("\nAnalyzing feature correlations...")

        # Select numerical features for correlation analysis
        numerical_features = ['LandAverageTemperature', 'LandAverageTemperatureUncertainty',
                              'year', 'month']

        # Include additional features if available
        optional_features = ['LandMaxTemperature', 'LandMinTemperature', 'TemperatureRange']
        for feature in optional_features:
            if feature in self.global_data.columns:
                numerical_features.append(feature)

        correlation_matrix = self.global_data[numerical_features].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                    center=0, fmt='.2f', square=True,
                    cbar_kws={'shrink': 0.8})
        plt.title('Climate Feature Correlation Matrix Analysis',
                  fontsize=16, fontweight='bold', pad=25)
        plt.tight_layout()
        plt.show()

        self.analytical_insights['temperature_year_correlation'] = correlation_matrix.loc['LandAverageTemperature', 'year']

    def analyze_geographical_patterns(self):
        """Analyze geographical temperature patterns and variations"""
        if self.country_data is None:
            return

        print("\nAnalyzing geographical temperature patterns...")

        # Identify countries with sufficient data
        country_data_counts = self.country_data['Country'].value_counts()
        major_countries = country_data_counts.head(8).index.tolist()

        # Focus on recent data for geographical analysis
        recent_country_data = self.country_data[self.country_data['year'] >= 2000]

        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # Country-level temperature trend analysis
        for country in major_countries[:4]:
            country_temperatures = recent_country_data[recent_country_data['Country'] == country]
            if len(country_temperatures) > 0:
                yearly_country_average = country_temperatures.groupby('year')['AverageTemperature'].mean()
                axes[0].plot(yearly_country_average.index, yearly_country_average.values,
                             label=country, linewidth=2, marker='o', markersize=4)

        axes[0].set_title('Country-Level Temperature Trend Analysis (2000-Present)',
                          fontsize=14, fontweight='bold', pad=20)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Average Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Current temperature distribution by country
        current_temperatures = recent_country_data.groupby('Country')['AverageTemperature'].mean().sort_values(ascending=False)
        current_temperatures.head(10).plot(kind='bar', ax=axes[1], color=config.colors[3])
        axes[1].set_title('Current Temperature Distribution by Country (2000-Present)',
                          fontsize=14, fontweight='bold', pad=20)
        axes[1].set_xlabel('Country')
        axes[1].set_ylabel('Average Temperature (°C)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def summarize_analytical_findings(self):
        """Summarize key analytical findings from EDA"""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS SUMMARY")
        print("=" * 60)

        insights = self.analytical_insights
        yearly_average = self.global_data.groupby('year')['LandAverageTemperature'].mean()

        print(f"Global Warming Rate: {insights.get('warming_rate', 0):+.2f}°C per century")
        print(f"Total Temperature Increase: {insights.get('total_temperature_change', 0):.2f}°C since 1850")
        if 'recent_warming_rate' in insights:
            print(f"Recent Warming Acceleration: {insights['recent_warming_rate']:+.2f}°C/century since 1970")
        print(f"Historical Average Temperature: {yearly_average.mean():.2f}°C")
        print(f"Average Seasonal Amplitude: {insights.get('average_seasonal_amplitude', 0):.2f}°C")
        print(f"Temperature-Year Correlation: {insights.get('temperature_year_correlation', 0):.3f}")


class ClimateFeatureEngineer:
    """Feature engineering for climate prediction models"""

    def __init__(self, df):
        self.dataset = df.copy()
        self.feature_set = []

    def engineer_temporal_features(self):
        """Create comprehensive temporal features"""
        print("Engineering temporal features...")

        # Temporal indexing
        self.dataset['temporal_index'] = np.arange(len(self.dataset))

        # Cyclical encoding for seasonal patterns
        self.dataset['month_sine'] = np.sin(2 * np.pi * self.dataset['month'] / 12)
        self.dataset['month_cosine'] = np.cos(2 * np.pi * self.dataset['month'] / 12)

        self.feature_set.extend(['temporal_index', 'month_sine', 'month_cosine'])

    def engineer_lag_features(self, lag_periods=[1, 12, 60]):
        """Create lagged temperature features"""
        print("Engineering lag features...")

        for lag in lag_periods:
            self.dataset[f'temperature_lag_{lag}'] = self.dataset['LandAverageTemperature'].shift(lag)
            self.feature_set.append(f'temperature_lag_{lag}')

    def engineer_rolling_features(self, window_sizes=[12, 60]):
        """Create rolling statistical features"""
        print("Engineering rolling features...")

        for window in window_sizes:
            self.dataset[f'rolling_mean_{window}'] = self.dataset['LandAverageTemperature'].rolling(
                window, min_periods=1).mean()
            self.dataset[f'rolling_std_{window}'] = self.dataset['LandAverageTemperature'].rolling(
                window, min_periods=1).std()

            self.feature_set.extend([f'rolling_mean_{window}', f'rolling_std_{window}'])

    def incorporate_climate_features(self):
        """Incorporate climate-specific predictive features"""
        print("Incorporating climate-specific features...")

        # Temperature measurement uncertainty
        if 'LandAverageTemperatureUncertainty' in self.dataset.columns:
            self.feature_set.append('LandAverageTemperatureUncertainty')

        # Temporal trend capture
        self.feature_set.append('year')

        # Seasonal pattern capture
        self.feature_set.append('month')

    def execute_feature_engineering(self):
        """Execute complete feature engineering pipeline"""
        self.engineer_temporal_features()
        self.engineer_lag_features()
        self.engineer_rolling_features()
        self.incorporate_climate_features()

        # Remove records with missing values from lag operations
        self.dataset = self.dataset.dropna()

        features = self.dataset[self.feature_set]
        target = self.dataset['LandAverageTemperature']

        print(f"Feature engineering completed: {len(self.feature_set)} features created")
        print(f"Final dataset dimensions: {features.shape}")

        return features, target, self.feature_set


class ClimateModelFramework:
    """Machine learning framework for climate prediction"""

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.trained_models = {}
        self.model_predictions = {}
        self.performance_metrics = {}

    def train_prediction_models(self):
        """Train multiple machine learning models for comparative analysis"""
        print("\nTraining predictive models...")
        print("=" * 50)

        # Model definitions
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=config.random_state, max_depth=10),
            'Random Forest': RandomForestRegressor(random_state=config.random_state, n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(random_state=config.random_state, n_estimators=100)
        }

        # Temporal train-test split
        split_threshold = int(len(self.features) * (1 - config.test_size))
        X_train, X_test = self.features.iloc[:split_threshold], self.features.iloc[split_threshold:]
        y_train, y_test = self.target.iloc[:split_threshold], self.target.iloc[split_threshold:]

        print(f"Training dataset: {X_train.shape[0]} observations")
        print(f"Testing dataset: {X_test.shape[0]} observations")

        # Model training and evaluation
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            self.trained_models[model_name] = model
            self.model_predictions[model_name] = predictions

            # Performance evaluation
            self.performance_metrics[model_name] = self.calculate_performance_metrics(y_test, predictions, model_name)

        return X_train, X_test, y_train, y_test

    def calculate_performance_metrics(self, actual, predicted, model_name):
        """Calculate comprehensive model performance metrics"""
        mean_absolute_err = mean_absolute_error(actual, predicted)
        root_mean_squared_err = np.sqrt(mean_squared_error(actual, predicted))
        r_squared = r2_score(actual, predicted)

        print(f"{model_name} Performance Evaluation:")
        print(f"  Mean Absolute Error: {mean_absolute_err:.4f}°C")
        print(f"  Root Mean Squared Error: {root_mean_squared_err:.4f}°C")
        print(f"  R-squared: {r_squared:.4f}")

        return {'MAE': mean_absolute_err, 'RMSE': root_mean_squared_err, 'R2': r_squared}

    def evaluate_model_performance(self, X_test, y_test):
        """Comprehensive model performance evaluation with visualizations"""
        print("\nModel Performance Evaluation")
        print("-" * 40)

        self.visualize_prediction_accuracy(X_test, y_test)
        self.analyze_residuals(X_test, y_test)
        self.compare_model_performance()

        return self.performance_metrics

    def visualize_prediction_accuracy(self, X_test, y_test):
        """Visualize model prediction accuracy"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        for index, (model_name, predictions) in enumerate(self.model_predictions.items()):
            if index >= 4:
                break

            axes[index].scatter(y_test, predictions, alpha=0.6, s=30, color=config.colors[index])
            perfect_prediction_line = np.linspace(y_test.min(), y_test.max(), 100)
            axes[index].plot(perfect_prediction_line, perfect_prediction_line, 'r--', alpha=0.8, linewidth=2)

            r2 = self.performance_metrics[model_name]['R2']
            axes[index].set_title(f'{model_name}\nR² = {r2:.3f}', fontsize=13, fontweight='bold', pad=15)
            axes[index].set_xlabel('Actual Temperature (°C)')
            axes[index].set_ylabel('Predicted Temperature (°C)')
            axes[index].grid(True, alpha=0.3)

        plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=3.0)
        plt.show()

    def analyze_residuals(self, X_test, y_test):
        """Analyze model prediction residuals"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        for index, (model_name, predictions) in enumerate(self.model_predictions.items()):
            if index >= 4:
                break

            residuals = y_test - predictions
            axes[index].scatter(predictions, residuals, alpha=0.6, s=30, color=config.colors[index])
            axes[index].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[index].set_title(f'{model_name} Residual Analysis', fontsize=13, fontweight='bold', pad=15)
            axes[index].set_xlabel('Predicted Temperature (°C)')
            axes[index].set_ylabel('Residuals')
            axes[index].grid(True, alpha=0.3)

        plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=3.0)
        plt.show()

    def compare_model_performance(self):
        """Comparative model performance analysis"""
        performance_dataframe = pd.DataFrame(self.performance_metrics).T

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Error metric comparison
        performance_dataframe[['RMSE', 'MAE']].plot(kind='bar', ax=axes[0], color=[config.colors[0], config.colors[1]])
        axes[0].set_title('Model Performance: Error Metrics\n(Lower Values Indicate Better Performance)',
                          fontsize=14, fontweight='bold', pad=20)
        axes[0].set_ylabel('Error Magnitude (°C)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # R-squared comparison
        performance_dataframe['R2'].plot(kind='bar', ax=axes[1], color=config.colors[2])
        axes[1].set_title('Model Performance: R-squared Values\n(Higher Values Indicate Better Performance)',
                          fontsize=14, fontweight='bold', pad=20)
        axes[1].set_ylabel('R-squared Coefficient')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class ModelOptimizationFramework:
    """Hyperparameter optimization and model refinement"""

    def __init__(self, X_train, y_train):
        self.training_features = X_train
        self.training_target = y_train
        self.optimized_models = {}

    def optimize_random_forest_parameters(self):
        """Optimize Random Forest hyperparameters"""
        print("\nInitiating Random Forest hyperparameter optimization...")

        parameter_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        random_forest = RandomForestRegressor(random_state=config.random_state)

        # Temporal cross-validation strategy
        temporal_cv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            estimator=random_forest,
            param_grid=parameter_grid,
            cv=temporal_cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.training_features, self.training_target)

        print(f"Optimization completed")
        print(f"Optimal parameters: {grid_search.best_params_}")
        print(f"Optimal cross-validation score: {-grid_search.best_score_:.4f} MSE")

        self.optimized_models['Random Forest'] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def analyze_feature_importance(self, model, feature_names):
        """Analyze and visualize feature importance"""
        print("\nConducting feature importance analysis...")

        if hasattr(model, 'feature_importances_'):
            importance_analysis = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(14, 10))
            sns.barplot(data=importance_analysis.head(15), x='importance', y='feature',
                        palette='viridis')
            plt.title('Feature Importance Analysis: Optimized Random Forest',
                     fontsize=16, fontweight='bold', pad=25)
            plt.xlabel('Feature Importance Score')
            plt.tight_layout()
            plt.show()

            print("\nTop Predictive Features:")
            for idx, row in importance_analysis.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

            return importance_analysis

        return None


def execute_climate_analysis():
    """Execute comprehensive climate data analysis pipeline"""
    print("Climate Change Analysis Pipeline Initialization")
    print("=" * 70)

    try:
        # Phase 1: Data Acquisition and Preparation
        print("\n[1/6] Data Acquisition and Validation")
        data_loader = ClimateDataLoader()
        available_datasets = data_loader.discover_dataset_files()

        primary_data = data_loader.load_global_temperature_data()
        if primary_data is None:
            print("Critical data unavailable. Analysis terminated.")
            return

        supplementary_data = data_loader.load_country_temperature_data()

        # Phase 2: Exploratory Data Analysis
        print("\n[2/6] Exploratory Data Analysis")
        data_analyzer = ClimateDataAnalyzer(primary_data, supplementary_data)
        analytical_insights = data_analyzer.execute_comprehensive_analysis()

        # Phase 3: Feature Engineering
        print("\n[3/6] Feature Engineering")
        feature_engineer = ClimateFeatureEngineer(primary_data)
        X, y, feature_names = feature_engineer.execute_feature_engineering()

        # Phase 4: Model Training and Evaluation
        print("\n[4/6] Model Training and Evaluation")
        model_framework = ClimateModelFramework(X, y)
        X_train, X_test, y_train, y_test = model_framework.train_prediction_models()
        performance_results = model_framework.evaluate_model_performance(X_test, y_test)

        # Phase 5: Model Optimization
        print("\n[5/6] Model Optimization")
        optimization_framework = ModelOptimizationFramework(X_train, y_train)
        optimized_rf = optimization_framework.optimize_random_forest_parameters()

        # Initialize feature importance tracking
        significant_predictors = []

        # Feature importance analysis
        importance_analysis = optimization_framework.analyze_feature_importance(optimized_rf, feature_names)
        if importance_analysis is not None:
            significant_predictors = importance_analysis.head(3)['feature'].tolist()
            print(f"\nMost Significant Predictive Features:")
            for i, feature in enumerate(significant_predictors, 1):
                print(f"  {i}. {feature}")

        # Optimized model evaluation
        optimized_predictions = optimized_rf.predict(X_test)
        optimized_performance = model_framework.calculate_performance_metrics(y_test, optimized_predictions, "Optimized Random Forest")

        # Phase 6: Final Analysis and Reporting
        print("\n[6/6] Final Analysis and Reporting")
        print("=" * 70)
        print("COMPREHENSIVE ANALYSIS RESULTS")
        print("=" * 70)

        # Performance summary
        print("\nModel Performance Summary:")
        performance_summary = pd.DataFrame(performance_results).T
        performance_summary.loc['Optimized Random Forest'] = optimized_performance
        print(performance_summary.round(4))

        # Best model identification
        best_performing_model = performance_summary['R2'].idxmax()
        best_r_squared = performance_summary.loc[best_performing_model, 'R2']
        best_rmse = performance_summary.loc[best_performing_model, 'RMSE']

        print(f"\nOptimal Predictive Model: {best_performing_model}")
        print(f"  R-squared: {best_r_squared:.4f}")
        print(f"  Root Mean Squared Error: {best_rmse:.4f}°C")
        print(f"  Mean Absolute Error: {performance_summary.loc[best_performing_model, 'MAE']:.4f}°C")

        # Climate change insights
        print(f"\nClimate Change Analytical Insights:")
        print(f"  Global Warming Trend: {analytical_insights.get('warming_rate', 0):+.2f}°C per century")
        print(f"  Total Temperature Increase: {analytical_insights.get('total_temperature_change', 0):.2f}°C")
        if 'recent_warming_rate' in analytical_insights:
            print(f"  Recent Warming Acceleration: {analytical_insights['recent_warming_rate']:+.2f}°C/century since 1970")

        # Key predictive features
        if significant_predictors:
            print(f"\nMost Significant Predictive Features:")
            for i, feature in enumerate(significant_predictors, 1):
                print(f"  {i}. {feature}")

        # Recommendations
        print(f"\nRecommendations for Further Analysis:")
        print(f"  1. Implement {best_performing_model} for temperature forecasting")
        print(f"  2. Prioritize monitoring of {significant_predictors[0] if significant_predictors else 'temporal patterns'}")
        print(f"  3. Explore ensemble methods for enhanced predictive accuracy")
        print(f"  4. Incorporate additional climate variables for comprehensive modeling")

        print(f"\nAnalysis successfully completed")
        print(f"Processed datasets: {len(available_datasets)}")
        print(f"Trained models: {len(performance_results) + 1}")

    except Exception as e:
        print(f"Analysis execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Dependency verification
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn']

    missing_dependencies = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_dependencies.append(package)

    if missing_dependencies:
        print("Required dependencies missing:")
        for package in missing_dependencies:
            print(f"  pip install {package}")
    else:
        execute_climate_analysis()
