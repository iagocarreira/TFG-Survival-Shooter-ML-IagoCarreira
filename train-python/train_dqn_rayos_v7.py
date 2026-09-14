"""
train_dqn_rayos_v7.py
=====================
Dueling Double DQN solo con rayos — configuracion espejada EXACTAMENTE de
ppo_oneshot.yaml para que la comparativa de resultados sea justa.

Diferencia con train_dqn_brujula_v7.py:
  - Current Mode = DQN_Rayos  (sin brujula, solo RayPerceptionSensor3D)
  - Vector Observation Space Size = 0 en Unity  (los rayos son toda la obs)

Mapping ppo_oneshot.yaml → este script:
  max_steps: 4000000           → TOTAL_STEPS = 4_000_000
  learning_rate: 3.0e-4        → LEARNING_RATE = 3e-4
  gamma: 0.99                  → GAMMA = 0.99
  hidden_units: 256            → tronco 256-256 (2 capas)
  num_layers: 2                → tronco 256-256 (2 capas)
  checkpoint_interval: 100000  → CHECKPOINT_FREQ = 100_000
  (beta, epsilon PPO, lambd)   → no aplica en DQN
  (curiosity strength: 0.02)   → no aplica en DQN

Curriculo — mismas 3 fases y mismos valores que ppo_oneshot.yaml,
avance POR PASOS (en vez de por reward umbral):
  Fase 0 — CoinCerca / SinEnemigos      paso       0 → 1 999 999
             coin_radius=3, normal=9999, big=9999
  Fase 1 — CoinCerca / PocosEnemigos    paso 2 000 000 → 2 999 999
             coin_radius=3, normal=10,   big=9999
  Fase 2 — CoinLibre / TodosEnemigos    paso 3 000 000 → 4 000 000
             coin_radius=-1, normal=5,   big=12

Recompensas Unity (deben coincidir con las del PPO Oneshot):
  +100   por moneda recogida
  +200   bonus victoria (5 monedas)
  -200   por morir
  -0.01  por paso

Configuracion Unity antes de lanzar:
  - Current Mode           = DQN_Rayos
  - Current Run Mode       = TrainDQN
  - attackDamage (Enemy)   = 100  (one-shot kill, igual que PPO)
  - RayPerceptionSensor3D  ACTIVADO (15 rayos, Enemy + Collectible + Terrain)
  - Vector Observation Space Size = 0  (sin brujula; solo rayos)
  - Behavior Type          = Default
  - Discrete Actions       = 1 rama de tamano 6
  - Max Step               = 0  (sin limite, igual que ppo_oneshot)
  - Max Steps Sin Enemigos = 5000

Uso:
    python train_dqn_rayos_v7.py
    python train_dqn_rayos_v7.py --run-id exp1 --seed 123
    tensorboard --logdir runs/
"""

import argparse
import csv
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper


# ==============================================================================
# Hiperparametros -- alineados con ppo_oneshot.yaml
# ==============================================================================

TOTAL_STEPS         = 4_000_000   # max_steps:           ppo_oneshot.yaml
BUFFER_SIZE         = 300_000     # replay buffer grande para retener memorias
BATCH_SIZE          = 64
LEARNING_RATE       = 3e-4        # learning_rate:        ppo_oneshot.yaml
GAMMA               = 0.99        # gamma extrinsico:     ppo_oneshot.yaml
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 3_000_000   # decae hasta 0.05 antes de que lleguen enemigos (paso 3M)
TARGET_UPDATE_FREQ  = 500
LEARNING_STARTS     = 2_000
GRAD_CLIP_NORM      = 10.0
CHECKPOINT_FREQ     = 100_000     # checkpoint_interval:  ppo_oneshot.yaml
LOG_FREQ            = 10          # escribir TensorBoard cada N episodios

N_ACTIONS           = 6


# ==============================================================================
# Curriculo por pasos — mismas 3 fases que ppo_oneshot.yaml
# ==============================================================================

class Curriculum:
    """
    Avanza de fase por numero de pasos globales.
    Las fases replican las del curriculo de ppo_oneshot.yaml.

    Paso           0 – 1 999 999 : CoinCerca / SinEnemigos
    Paso   2 000 000 – 2 999 999 : CoinCerca / PocosEnemigos
    Paso   3 000 000 – 4 000 000 : CoinLibre / TodosEnemigos
    """

    # (paso_inicio, nombre_display, coin_radius, normal_rate, big_rate)
    _FASES = [
        (        0, "CoinCerca/SinEnemigos",    3.0, 9999.0, 9999.0),
        (2_000_000, "CoinCerca/PocosEnemigos",  3.0,   10.0, 9999.0),
        (3_000_000, "CoinLibre/TodosEnemigos", -1.0,    5.0,   12.0),
    ]

    def __init__(self, ch: EnvironmentParametersChannel) -> None:
        self.ch    = ch
        self._fase = 0
        self._aplicar()

    def actualizar(self, paso_global: int) -> bool:
        """Devuelve True si hubo cambio de fase."""
        siguiente = self._fase + 1
        if siguiente < len(self._FASES) and paso_global >= self._FASES[siguiente][0]:
            self._fase = siguiente
            self._aplicar()
            print(f"\n  [Curriculum] Paso {paso_global:>9,} -> {self._FASES[self._fase][1]}\n")
            return True
        return False

    def fase(self) -> int:
        return self._fase

    def nombre(self) -> str:
        return self._FASES[self._fase][1]

    def estado(self, paso_global: int) -> str:
        nombre = self._FASES[self._fase][1]
        siguiente = self._fase + 1
        if siguiente < len(self._FASES):
            faltan = self._FASES[siguiente][0] - paso_global
            return f"Fase={nombre} | Prox.cambio en {faltan:>7,} pasos"
        return f"Fase={nombre} | Fase final"

    def _aplicar(self) -> None:
        _, nombre, coin_r, norm_r, big_r = self._FASES[self._fase]
        self.ch.set_float_parameter("coin_spawn_radius",       coin_r)
        self.ch.set_float_parameter("normal_enemy_spawn_rate", norm_r)
        self.ch.set_float_parameter("big_enemy_spawn_rate",    big_r)
        print(f"  [Curriculum] Fase {self._fase}: {nombre}  "
              f"(coin_r={coin_r}, normal={norm_r}, big={big_r})")


# ==============================================================================
# Red neuronal — Dueling Double DQN, 256 x 2 capas (= hidden_units/num_layers PPO)
# ==============================================================================

class DuelingDQN(nn.Module):
    def __init__(self, n_obs: int, n_acciones: int) -> None:
        super().__init__()
        self.tronco = nn.Sequential(
            nn.Linear(n_obs, 256), nn.ReLU(),
            nn.Linear(256,   256), nn.ReLU(),
        )
        self.valor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.ventaja = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_acciones),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tronco(x)
        v = self.valor(h)
        a = self.ventaja(h)
        return v + a - a.mean(dim=1, keepdim=True)


# ==============================================================================
# Replay buffer
# ==============================================================================

class ReplayBuffer:
    def __init__(self, capacidad: int) -> None:
        self._buf = deque(maxlen=capacidad)

    def guardar(self, s, a, r, s2, done) -> None:
        self._buf.append((s, a, r, s2, done))

    def muestrear(self, n: int):
        muestra = random.sample(self._buf, n)
        s, a, r, s2, d = zip(*muestra)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ==============================================================================
# Entrenamiento
# ==============================================================================

def main(run_id: str, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"DQN_Rayos_v7_{run_id}_{timestamp}"

    print("=" * 65)
    print("  Dueling Double DQN Rayos v7 — config espejada ppo_oneshot")
    print("=" * 65)
    print(f"  Run name:    {run_name}")
    print(f"  Seed:        {seed}")
    print(f"  Total pasos: {TOTAL_STEPS:,}")
    print(f"  LR={LEARNING_RATE}  Gamma={GAMMA}  Buffer={BUFFER_SIZE:,}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    # -- TensorBoard y CSV -----------------------------------------------------
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/{run_name}.csv"
    csv_fh   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w    = csv.writer(csv_fh)
    csv_w.writerow(["episodio", "paso_global", "reward", "monedas",
                    "pasos_ep", "epsilon", "fase_curriculum", "victoria"])

    print(f"  TensorBoard: tensorboard --logdir runs/")
    print(f"  CSV log:     {csv_path}\n")

    # -- Entorno ---------------------------------------------------------------
    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=seed,
                                 side_channels=[params_ch])
    env = UnityToGymWrapper(unity_env, uint8_visual=False,
                            flatten_branched=True, allow_multiple_obs=False)

    curriculum = Curriculum(params_ch)

    n_obs = env.observation_space.shape[0]
    print(f"  Observaciones: {n_obs} floats (solo rayos)")
    print(f"  Acciones:      {env.action_space}")
    print(f"  {curriculum.estado(0)}\n")

    # -- Redes -----------------------------------------------------------------
    red_principal = DuelingDQN(n_obs, N_ACTIONS).to(device)
    red_objetivo  = DuelingDQN(n_obs, N_ACTIONS).to(device)
    red_objetivo.load_state_dict(red_principal.state_dict())
    red_objetivo.eval()

    optimizador = optim.Adam(red_principal.parameters(), lr=LEARNING_RATE)
    buffer      = ReplayBuffer(BUFFER_SIZE)

    # -- Estado de entrenamiento -----------------------------------------------
    epsilon      = EPSILON_START
    paso_global  = 0
    episodio     = 0
    mejor_reward = float("-inf")

    ventana_reward  = deque(maxlen=50)
    ventana_monedas = deque(maxlen=50)
    perdida_acum    = 0.0
    pasos_train     = 0

    nombre_mejor = f"dqn_rayos_v7_{run_id}_mejor.pth"
    nombre_final = f"dqn_rayos_v7_{run_id}_final.pth"

    # -- Bucle principal -------------------------------------------------------
    while paso_global < TOTAL_STEPS:
        obs       = env.reset()
        reward_ep = 0.0
        terminado = False
        episodio += 1
        monedas_ep = 0
        pasos_ep   = 0

        while not terminado:
            # Accion epsilon-greedy
            if random.random() < epsilon or paso_global < LEARNING_STARTS:
                accion = env.action_space.sample()
            else:
                s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    accion = red_principal(s_t).argmax().item()

            obs_sig, recompensa, terminado, _ = env.step(accion)

            buffer.guardar(obs, accion, recompensa, obs_sig, float(terminado))

            obs        = obs_sig
            reward_ep += recompensa
            paso_global += 1
            pasos_ep   += 1

            if recompensa > 50.0:
                monedas_ep += 1

            # Actualizacion de red
            if paso_global >= LEARNING_STARTS and len(buffer) >= BATCH_SIZE:
                s, a, r, s2, d = [t.to(device) for t in buffer.muestrear(BATCH_SIZE)]

                with torch.no_grad():
                    a_next = red_principal(s2).argmax(1)
                    q_next = red_objetivo(s2).gather(1, a_next.unsqueeze(1)).squeeze(1)
                    q_obj  = r + GAMMA * q_next * (1 - d)

                q_actual = red_principal(s).gather(1, a.unsqueeze(1)).squeeze(1)
                perdida  = nn.functional.smooth_l1_loss(q_actual, q_obj)

                optimizador.zero_grad()
                perdida.backward()
                nn.utils.clip_grad_norm_(red_principal.parameters(), GRAD_CLIP_NORM)
                optimizador.step()

                perdida_acum += perdida.item()
                pasos_train  += 1

            # Red objetivo
            if paso_global % TARGET_UPDATE_FREQ == 0:
                red_objetivo.load_state_dict(red_principal.state_dict())

            # Checkpoint periodico
            if paso_global % CHECKPOINT_FREQ == 0:
                ckpt = f"dqn_rayos_v7_{run_id}_step{paso_global}.pth"
                torch.save(red_principal.state_dict(), ckpt)
                print(f"  [checkpoint] {ckpt}")

        # -- Fin de episodio ---------------------------------------------------
        epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * paso_global / EPSILON_DECAY_STEPS
        )
        curriculum.actualizar(paso_global)

        victoria = monedas_ep >= 5
        ventana_reward.append(reward_ep)
        ventana_monedas.append(monedas_ep)

        # Solo guardar mejor en fase final (con todos los enemigos)
        nuevo_mejor = curriculum.fase() == 2 and reward_ep > mejor_reward
        if nuevo_mejor:
            mejor_reward = reward_ep
            torch.save(red_principal.state_dict(), nombre_mejor)

        # CSV
        csv_w.writerow([episodio, paso_global, f"{reward_ep:.2f}", monedas_ep,
                        pasos_ep, f"{epsilon:.4f}", curriculum.fase(), int(victoria)])
        csv_fh.flush()

        # TensorBoard
        if episodio % LOG_FREQ == 0:
            media_reward  = np.mean(ventana_reward)
            media_monedas = np.mean(ventana_monedas)
            media_perdida = perdida_acum / max(pasos_train, 1)
            perdida_acum  = 0.0
            pasos_train   = 0

            writer.add_scalar("Reward/episodio",        reward_ep,     episodio)
            writer.add_scalar("Reward/media_50ep",      media_reward,  episodio)
            writer.add_scalar("Monedas/media_50ep",     media_monedas, episodio)
            writer.add_scalar("Entrenamiento/epsilon",  epsilon,       episodio)
            writer.add_scalar("Entrenamiento/perdida",  media_perdida, episodio)
            writer.add_scalar("Entrenamiento/pasos_ep", pasos_ep,      episodio)
            writer.add_scalar("Curriculum/fase",        curriculum.fase(), episodio)

            mejor_tag = " ** NUEVO MEJOR" if nuevo_mejor else ""
            print(
                f"Ep {episodio:4d} | Paso {paso_global:>7,} | "
                f"Reward: {reward_ep:8.2f} | Media50: {media_reward:7.2f} | "
                f"Monedas: {monedas_ep} | e={epsilon:.3f} | "
                f"Loss: {media_perdida:.4f} | "
                f"{curriculum.estado(paso_global)}{mejor_tag}"
            )

    # -- Fin de entrenamiento --------------------------------------------------
    torch.save(red_principal.state_dict(), nombre_final)
    writer.close()
    csv_fh.close()

    print("\n¡Entrenamiento finalizado!")
    print(f"  Modelo final:  {nombre_final}")
    print(f"  Mejor modelo:  {nombre_mejor}")
    print(f"  CSV log:       {csv_path}")
    print(f"  TensorBoard:   tensorboard --logdir runs/")
    env.close()


# ==============================================================================
# Argumentos
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DQN Rayos v7 — config espejada de ppo_oneshot.yaml"
    )
    parser.add_argument(
        "--run-id", type=str, default="run1",
        help="Identificador de la ejecucion (usado en nombres de ficheros)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria (default: 42)",
    )
    args = parser.parse_args()
    main(args.run_id, args.seed)
