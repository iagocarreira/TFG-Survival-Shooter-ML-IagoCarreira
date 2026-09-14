"""
Inferencia del modelo Dueling Double DQN con brújula v5.

Carga el modelo .pth entrenado con train_dqn_brujula_v5.py (3 M steps)
y lo ejecuta en Unity sin entrenar.

Configuración Unity antes de lanzar:
  - Current Mode                = DQN_Brujula
  - Current Run Mode            = Inference   ← NO TrainDQN (TimeScale = 1)
  - Behavior Type               = Default     (este script controla las acciones)
  - Model                       = None        (sin modelo ONNX cargado)
  - RayPerceptionSensor3D       ACTIVADO  (15 rayos, tags Enemy + Collectible)
  - CamaraFija / CameraSensor   DESACTIVADO
  - Vector Observation Space Size = 2         (brújula X,Z)
  - Discrete Actions            = 1 rama de tamaño 6

Uso:
  python play_dqn_brujula_v5.py
  python play_dqn_brujula_v5.py --modelo dqn_brujula_v5_mejor.pth --episodios 30
"""

import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper


# ── Arquitectura Dueling DQN (idéntica a train_dqn_brujula_v5.py) ─────────────

class DuelingDQN(nn.Module):
    def __init__(self, n_obs: int, n_acciones: int):
        super().__init__()
        self.tronco = nn.Sequential(
            nn.Linear(n_obs, 256), nn.ReLU(),
            nn.Linear(256, 256),   nn.ReLU(),
        )
        self.valor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.ventaja = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_acciones),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tronco(x)
        v = self.valor(h)
        a = self.ventaja(h)
        return v + a - a.mean(dim=1, keepdim=True)


N_ACCIONES = 6


# ── Guardado de resultados ────────────────────────────────────────────────────

def guardar_resultados(
    salida: str,
    modelo_path: str,
    n_episodios: int,
    rewards: list,
    monedas: list,
    pasos: list,
    victorias: int,
    epsilon: float,
):
    lineas = []
    lineas.append("=" * 60)
    lineas.append("  RESULTADOS INFERENCIA — DQN Brújula v5")
    lineas.append("=" * 60)
    lineas.append(f"  Fecha:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lineas.append(f"  Modelo:    {modelo_path}")
    lineas.append(f"  Episodios: {n_episodios}")
    lineas.append(f"  Epsilon:   {epsilon}")
    lineas.append("")
    lineas.append(f"{'Ep':>4}  {'Pasos':>6}  {'Reward':>9}  {'Monedas':>8}  {'Victoria':>8}")
    lineas.append("─" * 46)
    for i, (r, m, p) in enumerate(zip(rewards, monedas, pasos), 1):
        victoria = "SI" if m >= 5 else "NO"
        lineas.append(f"{i:>4}  {p:>6}  {r:>9.2f}  {m:>8}  {victoria:>8}")
    lineas.append("─" * 46)
    lineas.append("")
    lineas.append(f"  Victorias (≥5 monedas):   {victorias}/{n_episodios}  ({100 * victorias / n_episodios:.0f}%)")
    lineas.append(f"  Reward medio:             {np.mean(rewards):8.2f}")
    lineas.append(f"  Reward mejor:             {np.max(rewards):8.2f}")
    lineas.append(f"  Reward peor:              {np.min(rewards):8.2f}")
    lineas.append(f"  Desv. típica reward:      {np.std(rewards):8.2f}")
    lineas.append(f"  Media monedas/episodio:   {np.mean(monedas):8.2f}")
    lineas.append(f"  Media pasos/episodio:     {np.mean(pasos):8.2f}")
    lineas.append("=" * 60)

    texto = "\n".join(lineas)
    print(texto)

    with open(salida, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(f"\n  Resultados guardados en: {salida}")


# ── Inferencia ────────────────────────────────────────────────────────────────

def main(modelo_path: str, n_episodios: int, epsilon: float, salida: str):
    print("=" * 60)
    print("  DQN Brújula v5 — Inferencia (3 M steps)")
    print("=" * 60)
    print(f"  Modelo:    {modelo_path}")
    print(f"  Episodios: {n_episodios}")
    print(f"  Epsilon:   {epsilon}  (exploración residual)")
    print(f"  Salida:    {salida}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=0, side_channels=[params_ch])
    env = UnityToGymWrapper(
        unity_env,
        uint8_visual=False,
        flatten_branched=True,
        allow_multiple_obs=False,
    )

    # Condiciones de inferencia: mapa completo + enemigos activos
    params_ch.set_float_parameter("coin_spawn_radius",       -1.0)
    params_ch.set_float_parameter("normal_enemy_spawn_rate",  5.0)
    params_ch.set_float_parameter("big_enemy_spawn_rate",    12.0)

    n_obs = env.observation_space.shape[0]
    print(f"Observaciones: {n_obs} floats  (brújula 2 + rayos {n_obs - 2})")
    print(f"Acciones:      {env.action_space}\n")

    red = DuelingDQN(n_obs, N_ACCIONES).to(device)
    red.load_state_dict(torch.load(modelo_path, map_location=device))
    red.eval()
    print(f"Modelo cargado desde '{modelo_path}'\n")

    rewards_totales = []
    monedas_totales = []
    pasos_totales   = []
    victorias       = 0

    for ep in range(1, n_episodios + 1):
        obs       = env.reset()
        estado    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        reward_ep = 0.0
        pasos     = 0
        terminado = False
        monedas   = 0

        while not terminado:
            if random.random() < epsilon:
                accion = env.action_space.sample()
            else:
                with torch.no_grad():
                    accion = red(estado).argmax().item()

            obs_sig, recompensa, terminado, _ = env.step(accion)
            estado = torch.tensor(obs_sig, dtype=torch.float32).unsqueeze(0).to(device)

            reward_ep += recompensa
            pasos     += 1

            if recompensa > 0.5:
                monedas += 1

        gano = monedas >= 5
        if gano:
            victorias += 1

        rewards_totales.append(reward_ep)
        monedas_totales.append(monedas)
        pasos_totales.append(pasos)

        print(f"Ep {ep:3d} | Pasos: {pasos:5d} | Reward: {reward_ep:8.2f} | "
              f"Monedas: {monedas} {'🏆' if gano else ''}")

    env.close()

    guardar_resultados(
        salida, modelo_path, n_episodios,
        rewards_totales, monedas_totales, pasos_totales, victorias, epsilon,
    )


# ── Argumentos de línea de comandos ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia DQN Brújula v5")
    parser.add_argument(
        "--modelo",
        type=str,
        default="dqn_brujula_v5_mejor.pth",
        help="Ruta al archivo .pth  (default: dqn_brujula_v5_mejor.pth)",
    )
    parser.add_argument(
        "--episodios",
        type=int,
        default=20,
        help="Número de episodios de evaluación  (default: 20)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Exploración residual ε  (default: 0.05, usa 0 para puro greedy)",
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="resultados_dqn_brujula_v5.txt",
        help="Archivo .txt donde guardar los resultados  (default: resultados_dqn_brujula_v5.txt)",
    )
    args = parser.parse_args()
    main(args.modelo, args.episodios, args.epsilon, args.salida)
