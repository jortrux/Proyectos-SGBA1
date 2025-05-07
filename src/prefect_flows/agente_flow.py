
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, Tuple, Dict, Any, List, Union
from prefect import task, flow, get_run_logger


# Variables globales
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DOCKER_DATA_DIR = os.getenv("DOCKER_DATA_DIR")
KUBERNETES_PV_DIR = os.getenv("KUBERNETES_PV_DIR")

DATA_PATH = "dataAgente.csv"
TRENDS_PATH = "tendencias_diarias.csv"
PRICE_COL = "€/kwh"
CONSUMPTION_COL = "consumo_kwh"
MODEL_PATH = "energy_model.pt"
RESULTS_PATH = "energy_results.csv"


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


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


@task
def get_predictions_consumo_precio() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Se obtienen los datos de precio y consumo obtenidos de los otros 2 modelos
    df_precio = pd.read_csv(f"{KUBERNETES_PV_DIR}datos_simulacion_precio/prediccion_precio.csv")
    df_consumo = pd.read_csv(f"{KUBERNETES_PV_DIR}datos_simulacion_consumo/predicciones_consumo.csv")

    df_merged = pd.merge(df_precio, df_consumo, left_on="timestamp", right_on="fecha", how="inner")

    df_real = df_merged[["timestamp", "real_consumo", "€/kwh"]]
    df_real = df_real.rename(columns={"real_consumo": CONSUMPTION_COL, "€/kwh": PRICE_COL})

    df_predicted = df_merged[["timestamp", "pred_consumo", "prediccion_€/kwh"]]
    df_predicted = df_predicted.rename(columns={"pred_consumo": "price_future", "prediccion_€/kwh": "consumo_futuro"})

    return df_real, df_predicted


@task
def train_ppo(env, input_dim, hidden_dim=64, output_dim=3, lr=3e-4, num_epochs=10, eps_clip=0.2, 
              gamma=0.99, num_episodes=10):
    model = ActorCritic(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        states, actions, rewards, log_probs, values = [], [], [], [], []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            if next_state is None:
                break

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(log_probs)
        values_tensor = torch.FloatTensor(values)

        advantages = returns - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(num_epochs):
            logits, value_pred = model(states_tensor)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(value_pred.squeeze(), returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[EPISODIO {episode+1}] Pérdida: {loss.item():.4f} - Retorno total: {returns.sum().item():.2f}")

    return model


@task
def test_agent_and_plot(model, env):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = model(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        next_state, reward, done = env.step(action)
        total_reward += reward
        if next_state is None:
            break
        state = next_state

    print(f"Recompensa total en la prueba: {total_reward:.2f}")


@task
def inicializar_entorno_entrenamiento():
    print("[INFO] Cargando datos...")
    df, trends_df = get_predictions_consumo_precio()

    print("[INFO] Columnas disponibles:")
    print("  - dataAgente.csv:", df.columns.tolist())
    print("  - tendencias_diarias.csv:", trends_df.columns.tolist())

    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print("[INFO] Entorno inicializado correctamente.")
    return df, trends_df


@task
def entrenar_agente(df, trends_df, num_episodios=20):
    env = EnergyTradingEnvDaily(df, trends_df, PRICE_COL, CONSUMPTION_COL, capacity=10.0)
    input_dim = 5
    model = train_ppo(env, input_dim=input_dim, num_episodes=num_episodios)
    return model, env


@task
def evaluar_y_guardar_resultados(model, env):
    env.results_df = pd.DataFrame(columns=['día', 'baseline', 'modelo', 'penalizaciones'])
    env.daily_records = []

    test_agent_and_plot(model, env)

    env.results_df.to_csv(RESULTS_PATH, index=False)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Resultados guardados en {RESULTS_PATH}")
    print(f"[INFO] Modelo guardado en {MODEL_PATH}")


@flow
def flow_agente():

    logger = get_run_logger()

    try:
        df, trends_df = inicializar_entorno_entrenamiento()
        model, env = entrenar_agente(df, trends_df, num_episodios=20)
        evaluar_y_guardar_resultados(model, env)

        return 0
    
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(flow_agente())
