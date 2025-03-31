import fastf1
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from abc import ABC, abstractmethod

fastf1.Cache.enable_cache('cache')

# Abstract base class for data fetching strategy
class F1DataFetcher(ABC):
    @abstractmethod
    def fetch_data(self, year, round_number):
        pass

class FastF1DataFetcher(F1DataFetcher):
    def fetch_data(self, year, round_number):
        try:
            quali = fastf1.get_session(year, round_number, 'Q')
            quali.load()
            results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
            results = results.rename(columns={'FullName': 'Driver'})
            for col in ['Q1', 'Q2', 'Q3']:
                results[col + '_sec'] = results[col].apply(lambda x: x.total_seconds() if pd.notnull(x) else None)
            print("\nQualifying Results Structure:")
            print(results.head())
            return results
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("DataFrame columns available:", quali.results.columns.tolist())
            return None

class TimeConverter:
    @staticmethod
    def convert_time_to_seconds(time_str):
        if pd.isna(time_str):
            return None
        try:
            if ':' in time_str:
                minutes, seconds = time_str.split(':')
                return float(minutes) * 60 + float(seconds)
            else:
                return float(time_str)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert time: {time_str}, Error: {e}")
            return None

class DataCleaner:
    def __init__(self, converter):
        self.converter = converter

    def clean_data(self, df):
        print("\nBefore cleaning:")
        print(df[['Driver', 'Q1', 'Q2', 'Q3']].head())
        df['Q1_sec'] = df['Q1'].apply(self.converter.convert_time_to_seconds)
        df['Q2_sec'] = df['Q2'].apply(self.converter.convert_time_to_seconds)
        df['Q3_sec'] = df['Q3'].apply(self.converter.convert_time_to_seconds)
        print("\nAfter cleaning:")
        print(df[['Driver', 'Q1_sec', 'Q2_sec', 'Q3_sec']].head())
        return df.dropna()

class DataVisualizer:
    @staticmethod
    def visualize_data(df):
        sns.boxplot(data=df[['Q1_sec', 'Q2_sec', 'Q3_sec']])
        plt.title('Qualifying Lap Times (seconds)')
        plt.ylabel('Lap Time (seconds)')
        plt.show()

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train_and_evaluate(self, df):
        X = df[['Q1_sec', 'Q2_sec']]
        y = df['Q3_sec']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X)

        results_df = df[['Driver', 'TeamName', 'Q1_sec', 'Q2_sec', 'Q3_sec']].copy()
        results_df['Predicted_Q3'] = predictions
        results_df['Difference'] = results_df['Predicted_Q3'] - results_df['Q3_sec']
        results_df = results_df.sort_values('Predicted_Q3')

        print("\nPredicted Q3 Rankings:")
        print("=" * 70)
        print(f"{'Position':<10}{'Driver':<15}{'Team':<20}{'Predicted Time':<15}{'Actual Time':<15}")
        print("-" * 70)
        for idx, row in results_df.iterrows():
            pred_time = f"{row['Predicted_Q3']:.3f}"
            actual_time = f"{row['Q3_sec']:.3f}" if not pd.isna(row['Q3_sec']) else "N/A"
            print(f"{results_df.index.get_loc(idx)+1:<10}{row['Driver']:<15}{row['TeamName']:<20}{pred_time:<15}{actual_time:<15}")

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("\nModel Performance Metrics:")
        print(f'Mean Absolute Error: {mae:.2f} seconds')
        print(f'R^2 Score: {r2:.2f}')

class RecentDataFetcher:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher

    def fetch_recent_data(self):
        all_data = []
        current_year = 2025
        for round_num in range(1, 5):  # First 4 races of 2025
            print(f"Fetching data for {current_year} round {round_num}...")
            df = self.data_fetcher.fetch_data(current_year, round_num)
            if df is not None:
                df['Year'] = current_year
                df['Round'] = round_num
                all_data.append(df)
        print("Fetching 2024 Japanese GP data...")
        japan_2024 = self.data_fetcher.fetch_data(2024, 4) 
        if japan_2024 is not None:
            japan_2024['Year'] = 2024
            japan_2024['Round'] = 4
            all_data.append(japan_2024)
        return all_data

class PerformanceFactorApplier:
    def __init__(self, team_factors, driver_factors, base_time=89.5):
        self.team_factors = team_factors
        self.driver_factors = driver_factors
        self.base_time = base_time

    def apply_performance_factors(self, predictions_df):
        for idx, row in predictions_df.iterrows():
            team_factor = self.team_factors.get(row['Team'], 1.005)
            driver_factor = self.driver_factors.get(row['Driver'], 1.002)
            base_prediction = self.base_time * team_factor * driver_factor
            random_variation = np.random.uniform(-0.1, 0.1)
            predictions_df.loc[idx, 'Predicted_Q3'] = base_prediction + random_variation
        return predictions_df

class JapaneseGPPredictor:
    def __init__(self, model, performance_factor_applier):
        self.model = model
        self.performance_factor_applier = performance_factor_applier

    def predict_japanese_gp(self, latest_data):
        driver_teams = {
            'Max Verstappen': 'Red Bull Racing',
            'Sergio Perez': 'Red Bull Racing',
            'Charles Leclerc': 'Ferrari',
            'Carlos Sainz': 'Ferrari',
            'Lewis Hamilton': 'Mercedes',
            'George Russell': 'Mercedes',
            'Lando Norris': 'McLaren',
            'Oscar Piastri': 'McLaren',
            'Fernando Alonso': 'Aston Martin',
            'Lance Stroll': 'Aston Martin',
            'Daniel Ricciardo': 'RB',
            'Yuki Tsunoda': 'RB',
            'Alexander Albon': 'Williams',
            'Logan Sargeant': 'Williams',
            'Valtteri Bottas': 'Kick Sauber',
            'Zhou Guanyu': 'Kick Sauber',
            'Kevin Magnussen': 'Haas F1 Team',
            'Nico Hulkenberg': 'Haas F1 Team',
            'Pierre Gasly': 'Alpine',
            'Esteban Ocon': 'Alpine'
        }
        results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
        results_df = self.performance_factor_applier.apply_performance_factors(results_df)
        results_df = results_df.sort_values('Predicted_Q3')
        print("\nJapanese GP 2025 Qualifying Predictions:")
        print("=" * 100)
        print(f"{'Position':<10}{'Driver':<20}{'Team':<25}{'Predicted Q3':<15}")
        print("-" * 100)
        for idx, row in results_df.iterrows():
            print(f"{results_df.index.get_loc(idx)+1:<10}{row['Driver']:<20}{row['Team']:<25}{row['Predicted_Q3']:.3f}s")

if __name__ == "__main__":
    print("Initializing enhanced F1 prediction model...")

    data_fetcher = FastF1DataFetcher()
    recent_data_fetcher = RecentDataFetcher(data_fetcher)
    all_data = recent_data_fetcher.fetch_recent_data()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='all')
        imputer = SimpleImputer(strategy='median')
        X = valid_data[['Q1_sec', 'Q2_sec']]
        y = valid_data['Q3_sec']
        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y_clean = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())

        model = LinearRegression()
        model.fit(X_clean, y_clean)

        team_factors = {
            'Red Bull Racing': 0.997,
            'Ferrari': 0.998,
            'McLaren': 0.999,
            'Mercedes': 0.999,
            'Aston Martin': 1.001,
            'RB': 1.002,
            'Williams': 1.003,
            'Haas F1 Team': 1.004,
            'Kick Sauber': 1.004,
            'Alpine': 1.005,
        }
        driver_factors = {
            'Max Verstappen': 0.998,
            'Charles Leclerc': 0.999,
            'Carlos Sainz': 0.999,
            'Lando Norris': 0.999,
            'Oscar Piastri': 1.000,
            'Sergio Perez': 1.000,
            'Lewis Hamilton': 1.000,
            'George Russell': 1.000,
            'Fernando Alonso': 1.000,
            'Lance Stroll': 1.001,
            'Alex Albon': 1.001,
            'Daniel Ricciardo': 1.001,
            'Yuki Tsunoda': 1.002,
            'Valtteri Bottas': 1.002,
            'Zhou Guanyu': 1.003,
            'Kevin Magnussen': 1.003,
            'Nico Hulkenberg': 1.003,
            'Logan Sargeant': 1.004,
            'Pierre Gasly': 1.004,
            'Esteban Ocon': 1.004,
        }
        
        performance_factor_applier = PerformanceFactorApplier(team_factors, driver_factors)
        japanese_gp_predictor = JapaneseGPPredictor(model, performance_factor_applier)
        japanese_gp_predictor.predict_japanese_gp(valid_data)

        y_pred = model.predict(X_clean)
        mae = mean_absolute_error(y_clean, y_pred)
        r2 = r2_score(y_clean, y_pred)
        print("\nModel Performance Metrics:")
        print(f'Mean Absolute Error: {mae:.2f} seconds')
        print(f'R^2 Score: {r2:.2f}')
    else:
        print("Failed to fetch F1 data")
