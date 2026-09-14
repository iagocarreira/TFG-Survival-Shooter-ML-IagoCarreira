"""
play_dqn_brujula.py
===================
Ejecuta el modelo DQN_Brujula (Dueling DQN, rayos + dirección a la moneda).
Los resultados los graba Unity a través del SurvivalRecorder.

Configuración Unity antes de lanzar:
  - Current Mode              = DQN_Brujula
  - Current Run Mode          = Inference
  - Behavior Type             = Default   (Python controla las acciones)
  - Model                     = None
  - RayPerceptionSensor3D     ACTIVADO (15 rayos)
  - Vector Observation Space  = 2        (dirección X,Z a la moneda)
  - SurvivalRecorder:
      mlAgentsMode            = true
      episodiosObjetivo       = 100

Uso:
    python play_dqn_brujula.py
    python play_dqn_brujula.py --modelo ../train-python/dqn_brujula_v7_exp1_mejor.pth
    python play_dqn_brujula.py --episodios 50
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


N_ACCIONES = 6


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


def main(modelo_path: str, n_episodios: int, epsilon: float) -> None:
    print("=" * 60)
    print("  DQN Brujula — Inferencia (resultados grabados por Unity)")
    print("=" * 60)
    print(f"  Modelo:    {modelo_path}")
    print(f"  Episodios: {n_episodios}")
    print(f"  Epsilon:   {epsilon}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=0, side_channels=[params_ch])
    env = UnityToGymWrapper(unity_env, uint8_visual=False,
                            flatten_branched=True, allow_multiple_obs=False)

    params_ch.set_float_parameter("coin_spawn_radius",       -1.0)
    params_ch.set_float_parameter("normal_enemy_spawn_rate",  5.0)
    params_ch.set_float_parameter("big_enemy_spawn_rate",    12.0)

    n_obs = env.observation_space.shape[0]
    print(f"Observaciones: {n_obs} | Acciones: {N_ACCIONES}\n")

    red = DuelingDQN(n_obs, N_ACCIONES).to(device)
    red.load_state_dict(torch.load(modelo_path, map_location=device))
    red.eval()
    print(f"Modelo cargado.\n")

    for ep in range(1, n_episodios + 1):
        obs = env.reset()
        estado = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        terminado = False
        pasos = 0

        try:
            while not terminado:
                if random.random() < epsilon:
                    accion = env.action_space.sample()
                else:
                    with torch.no_grad():
                        accion = red(estado).argmax().item()

                obs_sig, _, terminado, _ = env.step(accion)
                estado = torch.tensor(obs_sig, dtype=torch.float32).unsqueeze(0).to(device)
                pasos += 1
        except Exception:
            terminado = True

        print(f"Ep {ep:3d}/{n_episodios} completado ({pasos} pasos)")

    try:
        env.close()
    except Exception:
        pass
    print("\nDone. Resultados grabados por el SurvivalRecorder de Unity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",    type=str,   default="../dqn_brujula/dqn_brujula_v6_mejor.pth")
    parser.add_argument("--episodios", type=int,   default=100)
    parser.add_argument("--epsilon",   type=float, default=0.0)
    args = parser.parse_args()
    main(args.modelo, args.episodios, args.epsilon)
