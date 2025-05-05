# energy_env_solar.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ============================
# FUNCION PARA GENERACION SOLAR
# ============================
def generar_energia_solar_realista_desde_timestamp(timestamp, noise_std=0.2):
    hora = timestamp.hour
    mes = timestamp.month
    estacionalidad = [0.6, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6]
    escala_estacion = estacionalidad[mes - 1]
    max_gen = 3.0 * escala_estacion
    if 6 <= hora <= 18:
        base_gen = np.sin(np.pi * (hora - 6) / 12)
    else:
        base_gen = 0.0
    ruido = np.clip(np.random.normal(1.0, noise_std), 0.0, 1.0)
    return round(max_gen * base_gen * ruido, 3)

# ============================
# CARGA DE DATOS
# ============================
df = pd.read_csv('dataAgente.csv', parse_dates=['timestamp'])
trends_df = pd.read_csv('tendencias_diarias.csv')

print("Columnas en dataAgente.csv:", df.columns.tolist())
print("Columnas en tendencias_diarias.csv:", trends_df.columns.tolist())

price_col = "€/kwh"
consumption_col = "consumo_kwh"

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

        self.results_df = pd.DataFrame(columns=['día', 'baseline', 'modelo', 'penalizaciones', 'solar'])
        self.reset()

    def reset(self):
        self.current_day = 0
        self.current_hour = 0
        self.current_step = 0
        self.battery = 0.0
        self.agent_energy_cost = 0.0
        self.agent_penalty_cost = 0.0
        self.baseline_day_cost = 0.0
        self.solar_generation_total = 0.0
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
        return np.array([current_price, current_consumption, forecast_price, forecast_consumption, battery_level], dtype=np.float32)

    def step(self, action):
        row = self.df.loc[self.current_step]
        price_real = self.raw_df.loc[self.current_step, self.price_col]
        consumption_real = self.raw_df.loc[self.current_step, self.consumption_col]
        timestamp = self.raw_df.loc[self.current_step, 'timestamp']

        solar_kwh = generar_energia_solar_realista_desde_timestamp(timestamp)
        self.solar_generation_total += solar_kwh

        # Calcular consumo neto y energía desechada
        if solar_kwh >= consumption_real:
            excedente = solar_kwh - consumption_real
            energia_almacenable = min(excedente, self.capacity - self.battery)
            self.battery += energia_almacenable
            energia_desechada = excedente - energia_almacenable  # energía que se pierde
            consumo_neto = 0.0
        else:
            consumo_neto = consumption_real - solar_kwh
            energia_desechada = 0.0

        penalty = 10.0
        cost_action_energy = 0.0
        cost_action_penalty = 0.0
        cost_total_hora = 0.0

        # Degradación de la batería
        if self.battery > 0:
            lost_energy = self.battery * self.degradation_rate
            self.battery = max(0.0, self.battery - lost_energy)

        # Acciones del agente
        accion_texto = ""
        if action == 0:
            forecast_price_real = self.raw_trends.loc[self.current_day, 'price_future']
            if forecast_price_real <= price_real:
                cost_action_penalty = penalty
                accion_texto = "Comprar (penalizado)"
            elif self.battery < self.capacity:
                max_buy = min(3.0, self.capacity - self.battery)
                cost_action_energy = max_buy * price_real
                self.battery += max_buy
                accion_texto = f"Comprar {max_buy:.2f}kWh"
            else:
                accion_texto = "Intento de compra (batería llena)"
        elif action == 1 and self.battery > 0:
            revenue = self.battery * price_real
            cost_action_energy = -revenue
            accion_texto = f"Vender {self.battery:.2f}kWh"
            self.battery = 0.0
        elif action == 2:
            accion_texto = "Hold"
        else:
            accion_texto = "Acción no válida"

        self.agent_energy_cost += cost_action_energy
        self.agent_penalty_cost += cost_action_penalty

        baseline_cost = consumption_real * price_real
        self.baseline_day_cost += baseline_cost

        # Coste de cubrir el consumo neto con batería y red
        if self.battery >= consumo_neto:
            consumption_cost = 0.0
            self.battery -= consumo_neto
        else:
            shortfall = consumo_neto - self.battery
            consumption_cost = shortfall * price_real
            self.battery = 0.0

        self.agent_energy_cost += consumption_cost
        cost_total_hora = cost_action_energy + cost_action_penalty + consumption_cost

        print(f"[DEBUG] Día {self.current_day} - Fecha: {timestamp.date()} - Hora {self.current_hour:02d} | Acción: {accion_texto} | "
            f"Precio: {price_real:.2f}€/kWh | Consumo: {consumption_real:.2f}kWh | Solar: {solar_kwh:.2f}kWh | "
            f"Consumo neto: {consumo_neto:.2f}kWh | Desechada: {energia_desechada:.2f}kWh | "
            f"Costo total hora: {cost_total_hora:.2f}€ | Batería: {self.battery:.2f}kWh")

        self.current_step += 1
        self.current_hour += 1

        if self.current_hour >= self.hours_per_day:
            fecha_dia = self.raw_df.loc[self.current_step - 1, 'timestamp'].date()
            print(f"[FIN DÍA {self.current_day}] Fecha: {fecha_dia} | Baseline: {self.baseline_day_cost:.2f}€ | Modelo: {self.agent_energy_cost:.2f}€ | "
                f"Penalizaciones: {self.agent_penalty_cost:.2f}€ | Solar total: {self.solar_generation_total:.2f}kWh")

            self.results_df.loc[len(self.results_df)] = [
                f"{self.current_day + 1} - {fecha_dia}",
                self.baseline_day_cost,
                self.agent_energy_cost,
                self.agent_penalty_cost,
                self.solar_generation_total
            ]
            self.agent_energy_cost = 0.0
            self.agent_penalty_cost = 0.0
            self.baseline_day_cost = 0.0
            self.solar_generation_total = 0.0
            self.current_hour = 0
            self.current_day += 1
            if self.current_day >= self.n_days:
                self.done = True
            reward = self.results_df.iloc[-1]['baseline'] - self.results_df.iloc[-1]['modelo']
        else:
            reward = 0.0

        state = None if self.done else self._get_state()
        return state, reward, self.done



# FUNCIONES DE ENTRENAMIENTO Y PRUEBA CON GRAFICA

def train_ppo(env, input_dim, hidden_dim=64, output_dim=3, lr=3e-4, num_epochs=10, eps_clip=0.2, gamma=0.99, num_episodes=10):
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
    return model

def test_agent_and_plot(model, env):
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = model(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        next_state, reward, done = env.step(action)
        if next_state is None:
            break
        state = next_state
    df = env.results_df
    plt.figure(figsize=(10, 6))
    plt.plot(df['día'], df['baseline'], label='Baseline')
    plt.plot(df['día'], df['modelo'], label='Modelo')
    plt.plot(df['día'], df['solar'], label='Solar')
    plt.title("Resultados diarios")
    plt.xlabel("Día")
    plt.ylabel("€ / Día")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Recompensa total: {df['baseline'].sum() - df['modelo'].sum():.2f}")

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

# Entrenamiento final y prueba
env = EnergyTradingEnvDaily(df, trends_df, price_col, consumption_col, capacity=10.0)
input_dim = 5
model = train_ppo(env, input_dim, num_episodes=20)
env.results_df = pd.DataFrame(columns=['día', 'baseline', 'modelo', 'penalizaciones', 'solar'])
env.daily_records = []
test_agent_and_plot(model, env)
torch.save(model.state_dict(), "energy_model.pt")
env.results_df.to_csv("Demo/energy_results.csv", index=False)
