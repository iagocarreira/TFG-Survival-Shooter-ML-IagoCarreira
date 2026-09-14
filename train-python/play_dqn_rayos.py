"""
Inferencia del modelo Dueling Double DQN con rayos (sin brújula).

Carga el modelo .pth entrenado y lo ejecuta en Unity sin entrenar.
Útil para visualizar el comportamiento aprendido y evaluar métricas.

Configuración Unity antes de lanzar:
  - Current Mode          = DQN_Rayos
  - Current Run Mode      = Inference   ← importante: NO TrainDQN
  - Behavior Type         = Default     (el script controla las acciones)
  - Model                 = None        (sin modelo ONNX cargado)
  - RayPerceptionSensor3D ACTIVADO
  - CamaraFija / CameraSensor  DESACTIVADO
  - Vector Observation Space Size = 0
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper


# ── Arquitectura (debe ser idéntica a la usada en entrenamiento) ──────────────

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


NOMBRES_ACCION = {0: "Quieto", 1: "Arriba", 2: "Abajo",
                  3: "Izquierda", 4: "Derecha", 5: "Disparar"}
N_ACCIONES = 6


# ── Inferencia ────────────────────────────────────────────────────────────────

def main(modelo_path: str, n_episodios: int, epsilon: float):
    print("=" * 60)
    print("  DQN Rayos — Inferencia")
    print("=" * 60)
    print(f"  Modelo:    {modelo_path}")
    print(f"  Episodios: {n_episodios}")
    print(f"  Epsilon:   {epsilon}  (exploración residual)")
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

    # Sin curriculum en inferencia: spawn normal, enemigos según escenario
    params_ch.set_float_parameter("coin_spawn_radius",       -1.0)
    params_ch.set_float_parameter("normal_enemy_spawn_rate",  5.0)
    params_ch.set_float_parameter("big_enemy_spawn_rate",    12.0)

    n_obs = env.observation_space.shape[0]
    print(f"Observaciones: {n_obs} floats | Acciones: {env.action_space}\n")

    red = DuelingDQN(n_obs, N_ACCIONES).to(device)
    red.load_state_dict(torch.load(modelo_path, map_location=device))
    red.eval()
    print(f"Modelo cargado desde '{modelo_path}'\n")

    rewards_totales  = []
    monedas_totales  = []
    victorias        = 0

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

        print(f"Ep {ep:3d} | Pasos: {pasos:5d} | Reward: {reward_ep:8.2f} | "
              f"Monedas: {monedas} {'🏆' if gano else ''}")

    print("\n" + "─" * 60)
    print("  RESUMEN")
    print("─" * 60)
    print(f"  Episodios jugados:       {n_episodios}")
    print(f"  Victorias (5 monedas):   {victorias}/{n_episodios}  "
          f"({100*victorias/n_episodios:.0f}%)")
    print(f"  Reward medio:            {np.mean(rewards_totales):8.2f}")
    print(f"  Reward mejor:            {np.max(rewards_totales):8.2f}")
    print(f"  Reward peor:             {np.min(rewards_totales):8.2f}")
    print(f"  Media monedas/episodio:  {np.mean(monedas_totales):8.2f}")
    print("─" * 60)

    env.close()


# ── Argumentos de línea de comandos ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia DQN Rayos")
    parser.add_argument("--modelo",    type=str,   default="dqn_rayos_mejor.pth",
                        help="Ruta al archivo .pth (default: dqn_rayos_mejor.pth)")
    parser.add_argument("--episodios", type=int,   default=20,
                        help="Número de episodios (default: 20)")
    parser.add_argument("--epsilon",   type=float, default=0.05,
                        help="Exploración residual (default: 0.05)")
    args = parser.parse_args()
    main(args.modelo, args.episodios, args.epsilon)
