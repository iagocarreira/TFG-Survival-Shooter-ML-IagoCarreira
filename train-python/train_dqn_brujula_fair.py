"""
train_dqn_brujula_fair.py
=========================
Entrenamiento Dueling Double DQN con brujula, configurado para que
el curriculo sea comparable al de PPO_Oneshot (ppo_oneshot.yaml).

Parametros espejados del ppo_oneshot.yaml:
  - Curriculo basado en REWARD MEDIO (measure: reward), no en pasos
  - 3 fases con los mismos umbrales: 60.0 y 30.0
  - Total de pasos: 4 M
  - Learning rate: 3e-4
  - Gamma: 0.99
  - Checkpoint cada 100 k pasos
  - Red: 256 x 2 capas

Curriculo (espejado del ppo_oneshot.yaml):
  Fase 1 - CoinCerca / SinEnemigos
            coin_radius=3, normal=9999, big=9999
            Avanza cuando reward_medio > 60  (min. 100 episodios)

  Fase 2 - CoinCerca / PocosEnemigos
            coin_radius=3, normal=10, big=9999
            Avanza cuando reward_medio > 30  (min. 150 episodios)

  Fase 3 - CoinLibre / TodosEnemigos
            coin_radius=-1, normal=5, big=12
            Hasta completar 4 M pasos.

Recompensas Unity (deben coincidir con ppo_oneshot.yaml):
  +100   por moneda recogida
  +200   bonus victoria (5 monedas)
  -200   por morir  (one-shot: attackDamage enemigo = 100)
  -0.01  por paso
  (sin penalizacion por disparo fallado)

Configuracion Unity antes de lanzar:
  - Current Mode          = DQN_Brujula
  - Current Run Mode      = TrainDQN
  - attackDamage (Enemy)  = 100  (one-shot kill, igual que PPO)
  - RayPerceptionSensor3D ACTIVADO (15 rayos, Enemy + Collectible + Terrain)
  - Vector Observation Space Size = 2  (brujula: direccion X,Z hacia la moneda)
  - Behavior Type         = Default
  - Discrete Actions      = 1 rama de tamano 6
  - Max Step              = 0  (sin limite por episodio)

Uso:
    python train_dqn_brujula_fair.py
    python train_dqn_brujula_fair.py --run-id run1 --checkpoint dqn_brujula_fair_run1_mejor.pth
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper


# ==============================================================================
# Hiperparametros -- alineados con ppo_oneshot.yaml
# ==============================================================================

TOTAL_STEPS        = 4_000_000   # max_steps PPO oneshot
BUFFER_SIZE        = 100_000
BATCH_SIZE         = 64
LEARNING_RATE      = 3e-4        # learning_rate PPO
GAMMA              = 0.99        # gamma PPO
EPSILON_START      = 1.0
EPSILON_END        = 0.05
EPSILON_DECAY      = 0.995       # decae lento para 4 M pasos
TARGET_UPDATE_FREQ = 500
LEARNING_STARTS    = 2_000
GRAD_CLIP_NORM     = 10.0
CHECKPOINT_FREQ    = 100_000     # checkpoint_interval PPO oneshot

N_ACTIONS          = 6


# ==============================================================================
# Curriculo basado en reward medio (igual que el YAML de PPO)
# ==============================================================================

class CurriculumPasos:
    """
    Avanza de fase al superar un numero fijo de pasos globales.

    Paso          0 - 1 000 000 : CoinCerca / SinEnemigos
    Paso  1 000 000 - 2 000 000 : CoinCerca / PocosEnemigos
    Paso  2 000 000 - 4 000 000 : CoinLibre / TodosEnemigos
    """

    # (paso_inicio, nombre, coin_radius, normal_rate, big_rate)
    _FASES = [
        (        0, "CoinCerca/SinEnemigos",    3.0, 9999.0, 9999.0),
        (1_000_000, "CoinCerca/PocosEnemigos",  3.0,   10.0, 9999.0),
        (2_000_000, "CoinLibre/TodosEnemigos", -1.0,    5.0,   12.0),
    ]

    def __init__(self, ch: EnvironmentParametersChannel):
        self.ch    = ch
        self._fase = 0
        self._aplicar()

    def actualizar(self, paso_global: int) -> None:
        siguiente = self._fase + 1
        if siguiente < len(self._FASES) and paso_global >= self._FASES[siguiente][0]:
            self._fase = siguiente
            self._aplicar()
            print(f"\n  [Curriculum] Paso {paso_global:>9,} -> {self._FASES[self._fase][1]}\n")

    def nombre(self) -> str:
        return self._FASES[self._fase][1]

    def fase(self) -> int:
        return self._fase

    def _aplicar(self) -> None:
        _, nombre, coin_r, norm_r, big_r = self._FASES[self._fase]
        self.ch.set_float_parameter("coin_spawn_radius",       coin_r)
        self.ch.set_float_parameter("normal_enemy_spawn_rate", norm_r)
        self.ch.set_float_parameter("big_enemy_spawn_rate",    big_r)
        print(f"  [Curriculum] Fase {self._fase}: {nombre}")


# ==============================================================================
# Red neuronal -- Dueling Double DQN, 256 x 2 capas (igual que PPO network)
# ==============================================================================

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


# ==============================================================================
# Replay buffer
# ==============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self._buf.append((s, a, r, s2, done))

    def sample(self, n: int):
        batch = random.sample(self._buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
        )

    def __len__(self):
        return len(self._buf)


# ==============================================================================
# Entrenamiento
# ==============================================================================

def main(run_id: str, checkpoint_path: str | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  DQN Brujula Fair -- curriculo identico a ppo_oneshot")
    print("=" * 60)
    print(f"  Dispositivo: {device}")
    print(f"  Run ID:      {run_id}")
    print(f"  Total pasos: {TOTAL_STEPS:,}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    # -- Entorno ---------------------------------------------------------------
    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=42,
                                 side_channels=[params_ch])
    env = UnityToGymWrapper(unity_env, uint8_visual=False,
                            flatten_branched=True, allow_multiple_obs=False)

    curriculum = CurriculumReward(params_ch)

    n_obs = env.observation_space.shape[0]
    print(f"  Observaciones: {n_obs} floats | Acciones: {N_ACTIONS}\n")

    # -- Redes -----------------------------------------------------------------
    red_principal = DuelingDQN(n_obs, N_ACTIONS).to(device)
    red_objetivo  = DuelingDQN(n_obs, N_ACTIONS).to(device)
    red_objetivo.load_state_dict(red_principal.state_dict())
    red_objetivo.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        red_principal.load_state_dict(torch.load(checkpoint_path,
                                                  map_location=device))
        red_objetivo.load_state_dict(red_principal.state_dict())
        print(f"  [OK] Checkpoint cargado: {checkpoint_path}\n")

    optimizer    = optim.Adam(red_principal.parameters(), lr=LEARNING_RATE)
    buffer       = ReplayBuffer(BUFFER_SIZE)
    nombre_mejor = f"dqn_brujula_fair_{run_id}_mejor.pth"

    # -- Variables de seguimiento ----------------------------------------------
    epsilon     = EPSILON_START
    paso_global = 0
    episodio    = 0
    mejor_reward = float("-inf")
    reward_ep   = 0.0
    obs         = env.reset()

    print(f"{'Paso':>10}  {'Ep':>5}  {'Fase':>2}  "
          f"{'Reward':>9}  {'eps':>5}  {'Mejor':>9}")
    print("-" * 55)

    # -- Bucle principal -------------------------------------------------------
    while paso_global < TOTAL_STEPS:

        # Accion epsilon-greedy
        if random.random() < epsilon or paso_global < LEARNING_STARTS:
            accion = env.action_space.sample()
        else:
            s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                accion = red_principal(s_t).argmax().item()

        obs_sig, recompensa, terminado, _ = env.step(accion)
        buffer.push(obs, accion, recompensa, obs_sig, float(terminado))

        obs         = obs_sig
        reward_ep  += recompensa
        paso_global += 1

        # -- Actualizacion de la red -------------------------------------------
        if paso_global >= LEARNING_STARTS and len(buffer) >= BATCH_SIZE:
            s, a, r, s2, d = [t.to(device) for t in buffer.sample(BATCH_SIZE)]

            with torch.no_grad():
                a_next = red_principal(s2).argmax(1)
                q_next = red_objetivo(s2).gather(1, a_next.unsqueeze(1)).squeeze(1)
                q_obj  = r + GAMMA * q_next * (1 - d)

            q_actual = red_principal(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss     = nn.functional.smooth_l1_loss(q_actual, q_obj)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(red_principal.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        # -- Red objetivo ------------------------------------------------------
        if paso_global % TARGET_UPDATE_FREQ == 0:
            red_objetivo.load_state_dict(red_principal.state_dict())

        # -- Checkpoint periodico ----------------------------------------------
        if paso_global % CHECKPOINT_FREQ == 0:
            ckpt = f"dqn_brujula_fair_{run_id}_step{paso_global}.pth"
            torch.save(red_principal.state_dict(), ckpt)
            print(f"  [checkpoint] {ckpt}")

        # -- Fin de episodio ---------------------------------------------------
        if terminado:
            curriculum.registrar_episodio(reward_ep)
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            if reward_ep > mejor_reward:
                mejor_reward = reward_ep
                torch.save(red_principal.state_dict(), nombre_mejor)

            print(f"{paso_global:>10,}  {episodio:>5}  {curriculum.fase():>2}  "
                  f"{reward_ep:>9.2f}  {epsilon:>5.3f}  {mejor_reward:>9.2f}"
                  f"  [{curriculum.nombre()}]")

            episodio  += 1
            reward_ep  = 0.0
            obs        = env.reset()

    # -- Guardado final --------------------------------------------------------
    nombre_final = f"dqn_brujula_fair_{run_id}_final.pth"
    torch.save(red_principal.state_dict(), nombre_final)
    print(f"\n[OK] Entrenamiento completado ({TOTAL_STEPS:,} pasos)")
    print(f"     Mejor modelo: {nombre_mejor}")
    print(f"     Modelo final: {nombre_final}")

    env.close()


# ==============================================================================
# Argumentos
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DQN Brujula -- curriculo identico a ppo_oneshot"
    )
    parser.add_argument(
        "--run-id", type=str, default="run1",
        help="Identificador de la ejecucion (se usa en el nombre de checkpoints)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Ruta a un .pth para reanudar el entrenamiento",
    )
    args = parser.parse_args()
    main(args.run_id, args.checkpoint)
