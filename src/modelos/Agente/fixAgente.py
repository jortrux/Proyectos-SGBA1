import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
# import matplotlib.pyplot as plt

df = pd.read_csv('dataAgente.csv')
trends_df = pd.read_csv('tendencias_diarias.csv')

print("Columnas en dataAgente.csv:", df.columns.tolist())
print("Columnas en tendencias_diarias.csv:", trends_df.columns.tolist())

price_col = "€/kwh"
consumption_col = "consumo_kwh"

###  Antes de empezar, no termino de estar convencido con una parte del dataset
###  Hablando con pablo, una solucion para interpretar el tiempo era coger la tendencia del dia y utilizar eso en lugar de la hora siguiente
###  Pero claro, si utilizo la tendencia de este dia, no tendra en cuenta el dia siguiente, y si tuilizo la del dia siguiente, se perderan las horas de este dia
###  Va a haber que pensar esto un poco mas pero ya nos acercamos a la version final (ESPERO)

from sklearn.preprocessing import MinMaxScaler

class EnergyTradingEnvDaily:
    def __init__(self, df, trends_df, price_col, consumption_col, capacity=10.0, degradation_rate=0.005):
        self.raw_df = df.copy()
        self.raw_trends = trends_df.copy()
        self.price_col = price_col
        self.consumption_col = consumption_col
        self.capacity = capacity
        self.degradation_rate = degradation_rate  # 🔋 pérdida por degradación (0.5% por hora)

        # 🧮 Normalización de columnas necesarias
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
        """
        Obtiene el forecast para el día actual. Ya no usamos el día siguiente.
        """
        if self.current_day < len(self.trends_df):
            forecast_price = self.trends_df.loc[self.current_day, 'price_scaled']
            forecast_consumption = self.trends_df.loc[self.current_day, 'consumption_scaled']
        else:
            # Usamos el último valor válido
            forecast_price = self.df['price_scaled'].iloc[-1]
            forecast_consumption = self.df['consumption_scaled'].iloc[-1]
        return forecast_price, forecast_consumption

    def _get_state(self):
        """
        Estado compuesto por:
         [precio actual, consumo actual, forecast precio, forecast consumo, nivel de batería]
        """
        row = self.df.loc[self.current_step]
        current_price = row['price_scaled']
        current_consumption = row['consumption_scaled']
        forecast_price, forecast_consumption = self._get_forecast()
        battery_level = self.battery / self.capacity
        state = np.array([current_price, current_consumption, forecast_price, forecast_consumption, battery_level], dtype=np.float32)
        return state

    def step(self, action):
        """
        Acciones disponibles:
          0: Comprar energía → cargar la batería hasta la capacidad máxima, solo si el forecast indica precio mayor.
          1: Vender energía → vender toda la energía almacenada.
          2: Hold → no operar con la batería en este horario.
        """

        # Datos reales (no escalados) para cálculos
        row = self.df.loc[self.current_step]
        price_real = self.raw_df.loc[self.current_step, self.price_col]
        consumption_real = self.raw_df.loc[self.current_step, self.consumption_col]

        penalty = 10.0
        cost_action_energy = 0.0
        cost_action_penalty = 0.0

        # Simular degradación de batería
        if self.battery > 0:
            lost_energy = self.battery * self.degradation_rate
            self.battery = max(0.0, self.battery - lost_energy)

        # Esto podria ser un switch jejejejejejejejeejejejeje
        if action == 0:
            forecast_price_real = self.raw_trends.loc[self.current_day, 'price_future']
            if forecast_price_real <= price_real:
                cost_action_penalty = penalty
                action_text = "Intento de comprar sin perspectiva de beneficio"
            elif self.battery >= self.capacity:
                # Puedes dejar este bloque comentado o activarlo según lo que decida Pablo
                # cost_action_penalty = penalty
                action_text = "Intento de comprar (batería llena)"
            else:
                max_buy = min(3.0, self.capacity - self.battery)
                cost_action_energy = max_buy * price_real
                self.battery += max_buy
                action_text = f"Comprar {max_buy:.2f} kWh (máx. permitido por hora)"
        elif action == 1:  # Vender
            if self.battery <= 0:
                ## Pablo me ha dicho que pruebe a eliminar esto para ver si mejora (todavia no lo he probado)
                # cost_action_penalty = penalty
                action_text = "Intento de vender (batería vacía)"
            else:
                revenue = self.battery * price_real
                cost_action_energy = -revenue
                self.battery = 0.0
                action_text = "Vender batería completa"
        elif action == 2:  # Hold
            cost_action_energy = 0.0
            action_text = "Hold (sin acción en batería)"
        else:
            action_text = "Acción desconocida"

        # Acumular costos
        self.agent_energy_cost += cost_action_energy
        self.agent_penalty_cost += cost_action_penalty

        # Costo baseline que es el de sin usar batería
        baseline_cost = consumption_real * price_real
        self.baseline_day_cost += baseline_cost

        # COSTO POR CONSUMO:
        if self.battery >= consumption_real:
            consumption_cost = 0.0
            self.battery -= consumption_real
        else:
            shortfall = consumption_real - self.battery
            consumption_cost = shortfall * price_real
            self.battery = 0.0
        self.agent_energy_cost += consumption_cost

        print(f"[DEBUG] Día {self.current_day}, Hora {self.current_hour:02d} | Acción: {action_text} | Precio: {price_real:.2f} | "
              f"Consumo: {consumption_real:.2f} | Costo acción (energía): {cost_action_energy:.2f} | "
              f"Costo acción (penalty): {cost_action_penalty:.2f} | Costo consumo: {consumption_cost:.2f} | Batería: {self.battery:.2f}")

        ###############################################
        # Next dia como se diria en spanglish, perdon tengo sueño
        ###############################################

        self.current_step += 1
        self.current_hour += 1

        if self.current_hour >= self.hours_per_day:
            daily_baseline = self.baseline_day_cost
            daily_agent_energy = self.agent_energy_cost
            daily_agent_penalty = self.agent_penalty_cost
            self.daily_records.append((daily_baseline, daily_agent_energy, daily_agent_penalty))
            print(f"[DEBUG] Fin del día {self.current_day} | Costo Baseline: {daily_baseline:.2f} | "
                  f"Costo Agente (energía): {daily_agent_energy:.2f} | Penalizaciones: {daily_agent_penalty:.2f} | "
                  f"Recompensa (ahorro energía): {(daily_baseline - daily_agent_energy):.2f}")

            self.results_df.loc[len(self.results_df)] = [
                self.current_day + 1, daily_baseline, daily_agent_energy, daily_agent_penalty
            ]

            self.agent_energy_cost = 0.0
            self.agent_penalty_cost = 0.0
            self.baseline_day_cost = 0.0
            self.current_hour = 0
            self.current_day += 1

            if self.current_day >= self.n_days:
                self.done = True
            reward = daily_baseline - daily_agent_energy
        else:
            reward = 0.0

        if not self.done:
            state = self._get_state()
        else:
            state = None

        return state, reward, self.done


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        input_dim: dimensión del estado (en este ejemplo 5)
        hidden_dim: tamaño de la capa oculta
        output_dim: número de acciones (3: comprar, vender, hold)
        """
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


def train_ppo(env, input_dim, hidden_dim=64, output_dim=3, lr=3e-4, num_epochs=10, eps_clip=0.2, 
              gamma=0.99, num_episodes=10):
    model = ActorCritic(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        print(f"\n=== INICIO DEL EPISODIO (iteración) {episode+1} ===")
        state = env.reset()
        done = False
        
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        while not done:
            print(f"[DEBUG TRAIN LOOP] Día: {env.current_day}, Hora: {env.current_hour}, Step: {env.current_step}")
            
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits, value = model(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                next_state, reward, done = env.step(action.item())
                
                if next_state is None:
                    print("[DEBUG] next_state es None. Termino el episodio.")
                    break
                
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                log_probs.append(log_prob.item())
                values.append(value.item())
                
                state = next_state
            except Exception as e:
                print(f"[ERROR FATAL EN train_ppo] {e}")
                break
            
        # Calcular retornos descontados (se acumulan múltiples días por episodio)
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
        
        # Actualización con PPO
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


def test_agent_and_plot(model, env):
    print("\n--- INICIO DE PRUEBA DEL AGENTE ENTRENADO ---")
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
    print("--- FIN DE PRUEBA ---")
    

# =============================================================================
# EJECUCIÓN FINAL
# Recordar mirar el bug de porque no termina la ejecucion
# =============================================================================
env = EnergyTradingEnvDaily(df, trends_df, price_col, consumption_col, capacity=10.0)
input_dim = 5  # [precio actual, consumo actual, forecast precio, forecast consumo, batería]
model = train_ppo(env, input_dim, num_episodes=20)

env.results_df = pd.DataFrame(columns=['día', 'baseline', 'modelo', 'penalizaciones'])
env.daily_records = []
test_agent_and_plot(model, env)
print("fin de ejecucion")

# Guardar resultados y modelo
env.results_df.to_csv("energy_results.csv", index=False)
torch.save(model.state_dict(), "energy_model.pt")