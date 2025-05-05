# pipeline_trading_energia.py
import os
import pandas as pd
import torch
from modelo_entorno import EnergyTradingEnvDaily
from modelo_ppo import ActorCritic, train_ppo, test_agent_and_plot

# Variables globales
DATA_PATH = "dataAgente.csv"
TRENDS_PATH = "tendencias_diarias.csv"
PRICE_COL = "€/kwh"
CONSUMPTION_COL = "consumo_kwh"
MODEL_PATH = "energy_model.pt"
RESULTS_PATH = "energy_results.csv"

def inicializar_entorno_entrenamiento():
    print("[INFO] Cargando datos...")
    df = pd.read_csv(DATA_PATH)
    trends_df = pd.read_csv(TRENDS_PATH)

    print("[INFO] Columnas disponibles:")
    print("  - dataAgente.csv:", df.columns.tolist())
    print("  - tendencias_diarias.csv:", trends_df.columns.tolist())

    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print("[INFO] Entorno inicializado correctamente.")
    return df, trends_df

def entrenar_agente(df, trends_df, num_episodios=20):
    env = EnergyTradingEnvDaily(df, trends_df, PRICE_COL, CONSUMPTION_COL, capacity=10.0)
    input_dim = 5
    model = train_ppo(env, input_dim=input_dim, num_episodes=num_episodios)
    return model, env

def evaluar_y_guardar_resultados(model, env):
    env.results_df = pd.DataFrame(columns=['día', 'baseline', 'modelo', 'penalizaciones'])
    env.daily_records = []

    test_agent_and_plot(model, env)

    env.results_df.to_csv(RESULTS_PATH, index=False)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Resultados guardados en {RESULTS_PATH}")
    print(f"[INFO] Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    df, trends_df = inicializar_entorno_entrenamiento()
    model, env = entrenar_agente(df, trends_df, num_episodios=20)
    evaluar_y_guardar_resultados(model, env)
