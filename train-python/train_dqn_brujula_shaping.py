"""
Dueling Double DQN con brújula y rayos — Escenario 2 (v2).

Variante de comparación frente a train_dqn_rayos.py.
La única diferencia es que el agente conoce la dirección hacia la moneda.

Observaciones vectoriales (sin cámara):
  - Brújula:           2 floats  (dirección X,Z normalizada hacia la moneda)
  - RayPerceptionSensor3D:  ~45 floats (15 rayos × 3 valores: hit_enemy, hit_coin, dist)
  ──────────────────────────────────────────────────────
  Total:              ~47 floats

Acciones discretas (6):
  0=quieto  1=arriba  2=abajo  3=izquierda  4=derecha  5=disparar

Condición de victoria:
  Si el agente recoge 5 monedas en un episodio, Unity llama EndEpisode()
  con un bonus de +200 adicional a los +100 por moneda.

Configuración Unity antes de lanzar:
  - Current Mode          = DQN_Brujula
  - Current Run Mode      = TrainDQN
  - RayPerceptionSensor3D ACTIVADO  (15 rayos, tags Enemy + Collectible)
  - CamaraFija / CameraSensor  DESACTIVADO
  - Vector Observation Space Size = 2  (brújula; los rayos se añaden aparte)
  - Behavior Type         = Default
  - Discrete Actions      = 1 rama de tamaño 6
"""

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


# ── Hiperparámetros ───────────────────────────────────────────────────────────

TOTAL_STEPS        = 1_500_000
BUFFER_SIZE        = 100_000
BATCH_SIZE         = 64
LEARNING_RATE      = 5e-4
GAMMA              = 0.99
EPSILON_START      = 1.0
EPSILON_END        = 0.05
EPSILON_DECAY      = 0.990
TARGET_UPDATE_FREQ = 500
LEARNING_STARTS    = 2_000
GRAD_CLIP_NORM     = 10.0

N_ACTIONS = 6


# ── Curriculum basado en pasos ────────────────────────────────────────────────

class Curriculum:
    """
    Progresión de dificultad fija por número de pasos globales.

    Paso         0 – 500 000 : moneda muy cerca (radio 2), sin enemigos
    Paso   500 000 – 750 000 : moneda un poco más lejos (radio 5), sin enemigos
    Paso   750 000 – 1 000 000: spawn normal (radio -1), sin enemigos
    Paso 1 000 000 – 1 500 000: spawn normal, monstruos activos
    """

    _FASES = [
        (        0, "CoinCerca/SinEnemigos",  2.0, 9999.0, 9999.0),
        (  500_000, "CoinMedia/SinEnemigos",  5.0, 9999.0, 9999.0),
        (  750_000, "CoinLibre/SinEnemigos", -1.0, 9999.0, 9999.0),
        (1_000_000, "CoinLibre/Monstruos",   -1.0,    5.0,   12.0),
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
    """
    Red MLP con arquitectura Dueling (Wang et al., 2016).
    Entrada: brújula (2 floats) + vector de rayos (~45 floats) = ~47 floats.
    """

    def __init__(self, n_obs: int, n_acciones: int):
        super().__init__()

        self.tronco = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.valor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.ventaja = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
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
    print("=" * 60)
    print("  Dueling Double DQN con Brújula + Shaping — PyTorch")
    print("=" * 60)
    print("Esperando a Unity... Dale al PLAY en el editor.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

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

    while paso_global < TOTAL_STEPS:
        obs       = env.reset()
        estado    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        reward_ep = 0.0
        terminado = False
        episodio += 1
        monedas_ep = 0

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

            if paso_global % TARGET_UPDATE_FREQ == 0:
                red_objetivo.load_state_dict(red_principal.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        curriculum.actualizar(paso_global)

        if reward_ep > mejor_reward:
            mejor_reward = reward_ep
            torch.save(red_principal.state_dict(), "dqn_brujula_shaping_mejor.pth")

        if episodio % 10 == 0:
            victoria = "🏆 VICTORIA" if monedas_ep >= 5 else ""
            print(
                f"Ep {episodio:4d} | Paso {paso_global:7d} | "
                f"Reward: {reward_ep:8.2f} | Mejor: {mejor_reward:8.2f} | "
                f"Monedas: {monedas_ep} | ε={epsilon:.3f} | "
                f"{curriculum.estado(paso_global)} {victoria}"
            )

    torch.save(red_principal.state_dict(), "dqn_brujula_shaping_final.pth")
    print("\n¡Entrenamiento finalizado!")
    print("Modelo final:  dqn_brujula_shaping_final.pth")
    print("Mejor modelo:  dqn_brujula_shaping_mejor.pth")
    env.close()


if __name__ == "__main__":
    main()
