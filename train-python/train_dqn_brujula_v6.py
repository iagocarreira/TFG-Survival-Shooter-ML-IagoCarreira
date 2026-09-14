"""
Dueling Double DQN con brújula y rayos — v6 con TensorBoard.

Igual que v5 (3 M steps, curriculum escalado) pero añade:
  - TensorBoard: reward, monedas, pérdida, epsilon, pasos/episodio
  - CSV de log por episodio (por si cierras sin querer)
  - Media móvil de 50 episodios para ver tendencia real

Ejecutar TensorBoard mientras entrena:
  tensorboard --logdir runs/

Configuración Unity antes de lanzar:
  - Current Mode          = DQN_Brujula
  - Current Run Mode      = TrainDQN
  - RayPerceptionSensor3D ACTIVADO  (15 rayos, tags Enemy + Collectible)
  - CamaraFija / CameraSensor  DESACTIVADO
  - Vector Observation Space Size = 2  (brújula; los rayos se añaden aparte)
  - Behavior Type         = Default
  - Discrete Actions      = 1 rama de tamaño 6
"""

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


# ── Hiperparámetros ───────────────────────────────────────────────────────────

TOTAL_STEPS        = 3_000_000
BUFFER_SIZE        = 300_000   # más grande: retiene memorias de fases anteriores (evita olvido)
BATCH_SIZE         = 64
LEARNING_RATE      = 5e-4
GAMMA              = 0.99
EPSILON_START      = 1.0
EPSILON_END        = 0.05
# Epsilon decae por PASOS, no por episodios → garantiza que sea bajo
# antes de que aparezcan los enemigos (paso 2 000 000)
EPSILON_DECAY_STEPS = 1_800_000   # llega a 0.05 en el paso 1.8M
TARGET_UPDATE_FREQ  = 500
LEARNING_STARTS     = 2_000
GRAD_CLIP_NORM      = 10.0
LOG_FREQ            = 10          # escribir a TensorBoard cada N episodios

N_ACTIONS = 6


# ── Curriculum basado en pasos ────────────────────────────────────────────────

class Curriculum:
    """
    Paso           0 – 1 000 000 : moneda muy cerca (radio 2), sin enemigos
    Paso   1 000 000 – 1 500 000 : moneda un poco más lejos (radio 5), sin enemigos
    Paso   1 500 000 – 2 000 000 : spawn normal (radio -1), sin enemigos
    Paso   2 000 000 – 3 000 000 : spawn normal, monstruos activos
    """

    _FASES = [
        (        0, "CoinCerca/SinEnemigos",  2.0, 9999.0, 9999.0),
        (1_000_000, "CoinMedia/SinEnemigos",  5.0, 9999.0, 9999.0),
        (1_500_000, "CoinLibre/SinEnemigos", -1.0, 9999.0, 9999.0),
        (2_000_000, "CoinLibre/Monstruos",   -1.0,    5.0,   12.0),
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
            print(f"\n  [Curriculum] ▶ Paso {paso_global:>7d} → {self._FASES[self._fase][1]}\n")

    def fase_actual(self) -> int:
        return self._fase

    def estado(self, paso_global: int) -> str:
        nombre = self._FASES[self._fase][1]
        siguiente = self._fase + 1
        if siguiente < len(self._FASES):
            faltan = self._FASES[siguiente][0] - paso_global
            return f"Fase={nombre} | Próximo cambio en {faltan:>7d} pasos"
        return f"Fase={nombre} | Fase final"

    def _aplicar(self):
        _, _, coin_r, norm_r, big_r = self._FASES[self._fase]
        self.ch.set_float_parameter("coin_spawn_radius",       coin_r)
        self.ch.set_float_parameter("normal_enemy_spawn_rate", norm_r)
        self.ch.set_float_parameter("big_enemy_spawn_rate",    big_r)


# ── Dueling DQN ───────────────────────────────────────────────────────────────

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tronco(x)
        v = self.valor(h)
        a = self.ventaja(h)
        return v + a - a.mean(dim=1, keepdim=True)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacidad: int):
        self.memoria = deque(maxlen=capacidad)

    def guardar(self, s, a, r, s2, done):
        self.memoria.append((s, a, r, s2, done))

    def muestrear(self, n: int):
        return random.sample(self.memoria, n)

    def __len__(self) -> int:
        return len(self.memoria)


# ── Bucle de entrenamiento ────────────────────────────────────────────────────

def main():
    run_name = f"DQN_Brujula_v6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print(f"  Dueling Double DQN con Brújula v6 — PyTorch")
    print(f"  Run: {run_name}")
    print("=" * 60)
    print("Esperando a Unity... Dale al PLAY en el editor.\n")

    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # CSV de log por episodio
    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/{run_name}.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episodio", "paso_global", "reward", "monedas", "pasos_ep",
                         "epsilon", "fase_curriculum", "victoria"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"TensorBoard: tensorboard --logdir runs/")
    print(f"CSV log:     {csv_path}\n")

    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=42, side_channels=[params_ch])
    env = UnityToGymWrapper(
        unity_env,
        uint8_visual=False,
        flatten_branched=True,
        allow_multiple_obs=False,
    )

    curriculum = Curriculum(params_ch)

    n_obs = env.observation_space.shape[0]
    print(f"Observaciones: {n_obs} floats  (brújula 2 + rayos {n_obs - 2})")
    print(f"Acciones:      {env.action_space}")
    print(f"Curriculum:    {curriculum.estado(0)}\n")

    red_principal = DuelingDQN(n_obs, N_ACTIONS).to(device)
    red_objetivo  = DuelingDQN(n_obs, N_ACTIONS).to(device)
    red_objetivo.load_state_dict(red_principal.state_dict())
    red_objetivo.eval()

    optimizador = optim.Adam(red_principal.parameters(), lr=LEARNING_RATE)
    buffer      = ReplayBuffer(BUFFER_SIZE)
    epsilon     = EPSILON_START

    paso_global  = 0
    episodio     = 0
    mejor_reward = float("-inf")

    # Media móvil de los últimos 50 episodios
    ventana_reward  = deque(maxlen=50)
    ventana_monedas = deque(maxlen=50)
    perdida_acum    = 0.0
    pasos_train     = 0

    while paso_global < TOTAL_STEPS:
        obs        = env.reset()
        estado     = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        reward_ep  = 0.0
        terminado  = False
        episodio  += 1
        monedas_ep = 0
        pasos_ep   = 0

        while not terminado:
            if random.random() < epsilon or paso_global < LEARNING_STARTS:
                accion = env.action_space.sample()
            else:
                with torch.no_grad():
                    accion = red_principal(estado).argmax().item()

            obs_sig, recompensa, terminado, _ = env.step(accion)
            sig_estado = torch.tensor(obs_sig, dtype=torch.float32).unsqueeze(0)

            buffer.guardar(estado.cpu(), accion, recompensa, sig_estado, terminado)

            estado      = sig_estado.to(device)
            reward_ep  += recompensa
            paso_global += 1
            pasos_ep    += 1

            if recompensa > 0.5:
                monedas_ep += 1

            if paso_global >= LEARNING_STARTS and len(buffer) >= BATCH_SIZE:
                muestra = buffer.muestrear(BATCH_SIZE)
                estados, acciones, recompensas, sig_estados, terminados = zip(*muestra)

                estados     = torch.cat(estados).to(device)
                sig_estados = torch.cat(sig_estados).to(device)
                acciones    = torch.tensor(acciones,    dtype=torch.long).to(device)
                recompensas = torch.tensor(recompensas, dtype=torch.float32).to(device)
                terminados  = torch.tensor(terminados,  dtype=torch.float32).to(device)

                q_actual = red_principal(estados).gather(1, acciones.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = red_principal(sig_estados).argmax(1).unsqueeze(1)
                    q_sig        = red_objetivo(sig_estados).gather(1, next_actions).squeeze(1)
                    q_objetivo   = recompensas + GAMMA * q_sig * (1 - terminados)

                perdida = nn.functional.smooth_l1_loss(q_actual, q_objetivo)

                optimizador.zero_grad()
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(red_principal.parameters(), GRAD_CLIP_NORM)
                optimizador.step()

                perdida_acum += perdida.item()
                pasos_train  += 1

            if paso_global % TARGET_UPDATE_FREQ == 0:
                red_objetivo.load_state_dict(red_principal.state_dict())

        # Epsilon lineal por pasos: garantiza que sea bajo antes del paso 2M
        epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * paso_global / EPSILON_DECAY_STEPS
        )
        curriculum.actualizar(paso_global)

        victoria = monedas_ep >= 5
        ventana_reward.append(reward_ep)
        ventana_monedas.append(monedas_ep)

        # Solo guardar el mejor modelo en la fase con enemigos (fase índice 3)
        # Así el "mejor" siempre refleja rendimiento bajo presión real
        nuevo_mejor = curriculum.fase_actual() == 3 and reward_ep > mejor_reward
        if nuevo_mejor:
            mejor_reward = reward_ep
            torch.save(red_principal.state_dict(), "dqn_brujula_v6_mejor.pth")

        # ── CSV: una fila por episodio ─────────────────────────────────────
        csv_writer.writerow([
            episodio, paso_global, f"{reward_ep:.2f}", monedas_ep,
            pasos_ep, f"{epsilon:.4f}", curriculum.fase_actual(),
            int(victoria)
        ])
        csv_file.flush()

        # ── TensorBoard: cada LOG_FREQ episodios ───────────────────────────
        if episodio % LOG_FREQ == 0:
            media_reward  = np.mean(ventana_reward)
            media_monedas = np.mean(ventana_monedas)
            media_perdida = perdida_acum / max(pasos_train, 1)
            perdida_acum  = 0.0
            pasos_train   = 0

            writer.add_scalar("Reward/episodio",           reward_ep,    episodio)
            writer.add_scalar("Reward/media_50ep",         media_reward, episodio)
            writer.add_scalar("Monedas/media_50ep",        media_monedas, episodio)
            writer.add_scalar("Entrenamiento/epsilon",     epsilon,      episodio)
            writer.add_scalar("Entrenamiento/perdida",     media_perdida, episodio)
            writer.add_scalar("Entrenamiento/pasos_ep",    pasos_ep,     episodio)
            writer.add_scalar("Curriculum/fase",           curriculum.fase_actual(), episodio)

            tag = "🏆" if victoria else ""
            mejor_tag = " ★ NUEVO MEJOR" if nuevo_mejor else ""
            print(
                f"Ep {episodio:4d} | Paso {paso_global:7d} | "
                f"Reward: {reward_ep:8.2f} | Media50: {media_reward:7.2f} | "
                f"Monedas: {monedas_ep} | ε={epsilon:.3f} | "
                f"Loss: {media_perdida:.4f} | "
                f"{curriculum.estado(paso_global)} {tag}{mejor_tag}"
            )

    torch.save(red_principal.state_dict(), "dqn_brujula_v6_final.pth")
    writer.close()
    csv_file.close()

    print("\n¡Entrenamiento finalizado!")
    print(f"Modelo final:  dqn_brujula_v6_final.pth")
    print(f"Mejor modelo:  dqn_brujula_v6_mejor.pth")
    print(f"CSV log:       {csv_path}")
    print(f"TensorBoard:   tensorboard --logdir runs/")
    env.close()


if __name__ == "__main__":
    main()
