# modelo_entorno.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class EnergyTradingEnvDaily:
    def __init__(self, df, trends_df, price_col, consumption_col, capacity=10.0, degradation_rate=0.005):
        self.raw_df = df.copy()
        self.raw_trends = trends_df.copy()
        self.price_col = price_col
        self.consumption_col = consumption_col
        self.capacity = capacity
        self.degradation_rate = degradation_rate

        self.scaler = MinMaxScaler()
        self.df = self.raw_df[[price_col, consumption_col]].copy()
        self.df[['price_scaled', 'consumption_scaled']] = self.scaler.fit_transform(self.df)
        self.trends_df = self.raw_trends[['price_future', 'consumo_futuro']].copy()
        self.trends_df[['price_scaled', 'consumption_scaled']] = self.scaler.transform(
            self.trends_df.rename(columns={'price_future': price_col, 'consumo_futuro': consumption_col})
        )

        self.hours_per_day = 24
        self.n_days = len(self.df) // self.hours_per_day
        self.df = self.df.iloc[:self.n_days * self.hours_per_day].reset_index(drop=True)

        self.results_df = pd.DataFrame(columns=['día', 'baseline', 'modelo', 'penalizaciones'])
        self.reset()

    def reset(self):
        self.current_day = 0
        self.current_hour = 0
        self.current_step = 0
        self.battery = 0.0
        self.agent_energy_cost = 0.0
        self.agent_penalty_cost = 0.0
        self.baseline_day_cost = 0.0
        self.daily_records = []
        self.done = False
        return self._get_state()

    def _get_forecast(self):
        if self.current_day < len(self.trends_df):
            forecast_price = self.trends_df.loc[self.current_day, 'price_scaled']
            forecast_consumption = self.trends_df.loc[self.current_day, 'consumption_scaled']
        else:
            forecast_price = self.df['price_scaled'].iloc[-1]
            forecast_consumption = self.df['consumption_scaled'].iloc[-1]
        return forecast_price, forecast_consumption

    def _get_state(self):
        row = self.df.loc[self.current_step]
        current_price = row['price_scaled']
        current_consumption = row['consumption_scaled']
        forecast_price, forecast_consumption = self._get_forecast()
        battery_level = self.battery / self.capacity
        state = np.array([current_price, current_consumption, forecast_price, forecast_consumption, battery_level], dtype=np.float32)
        return state

    def step(self, action):
        row = self.df.loc[self.current_step]
        price_real = self.raw_df.loc[self.current_step, self.price_col]
        consumption_real = self.raw_df.loc[self.current_step, self.consumption_col]

        penalty = 10.0
        cost_action_energy = 0.0
        cost_action_penalty = 0.0

        if self.battery > 0:
            lost_energy = self.battery * self.degradation_rate
            self.battery = max(0.0, self.battery - lost_energy)

        if action == 0:
            forecast_price_real = self.raw_trends.loc[self.current_day, 'price_future']
            if forecast_price_real <= price_real:
                cost_action_penalty = penalty
            elif self.battery < self.capacity:
                max_buy = min(3.0, self.capacity - self.battery)
                cost_action_energy = max_buy * price_real
                self.battery += max_buy
        elif action == 1:
            if self.battery > 0:
                revenue = self.battery * price_real
                cost_action_energy = -revenue
                self.battery = 0.0
        elif action == 2:
            cost_action_energy = 0.0

        self.agent_energy_cost += cost_action_energy
        self.agent_penalty_cost += cost_action_penalty

        baseline_cost = consumption_real * price_real
        self.baseline_day_cost += baseline_cost

        if self.battery >= consumption_real:
            consumption_cost = 0.0
            self.battery -= consumption_real
        else:
            shortfall = consumption_real - self.battery
            consumption_cost = shortfall * price_real
            self.battery = 0.0
        self.agent_energy_cost += consumption_cost

        self.current_step += 1
        self.current_hour += 1

        if self.current_hour >= self.hours_per_day:
            self.results_df.loc[len(self.results_df)] = [
                self.current_day + 1, self.baseline_day_cost, self.agent_energy_cost, self.agent_penalty_cost
            ]
            self.agent_energy_cost = 0.0
            self.agent_penalty_cost = 0.0
            self.baseline_day_cost = 0.0
            self.current_hour = 0
            self.current_day += 1

            if self.current_day >= self.n_days:
                self.done = True
            reward = self.results_df.iloc[-1]['baseline'] - self.results_df.iloc[-1]['modelo']
        else:
            reward = 0.0

        state = None if self.done else self._get_state()
        return state, reward, self.done
