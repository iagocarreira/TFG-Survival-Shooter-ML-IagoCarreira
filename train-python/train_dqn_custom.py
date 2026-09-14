"""
DQN propio implementado desde cero con PyTorch.
Entrenamiento visual con CamaraFija — Escenario 2 (recoger moneda).

A diferencia de stable-baselines3, aquí implementamos nosotros mismos:
  - La red neuronal convolucional (CNN)
  - El replay buffer
  - El algoritmo DQN completo (epsilon-greedy, target network, etc.)
  - Un curriculum manual que replica la progresión del YAML de PPO

Configuración Unity antes de lanzar:
  - Current Mode     = DQN_Discrete
  - Current Run Mode = TrainDQN
  - CamaraFija activa con ZonaCamara
  - CameraSensor en Player_Agent apuntando a CamaraFija (128x128, sin compresión)
  - RayPerceptionSensor3D DESACTIVADO
  - Vector Observation Space Size = 0
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper

# ── Hiperparámetros ───────────────────────────────────────────────────────────

TOTAL_STEPS        = 1_000_000  # pasos totales de entrenamiento
BUFFER_SIZE        = 50_000     # capacidad del replay buffer
BATCH_SIZE         = 32         # muestras por actualización
LEARNING_RATE      = 1e-4       # tasa de aprendizaje
GAMMA              = 0.99       # factor de descuento de recompensas futuras
EPSILON_START      = 1.0        # exploración inicial (100% aleatoria)
EPSILON_END        = 0.05       # exploración mínima (5% aleatoria)
EPSILON_DECAY      = 0.995      # reducción de epsilon por episodio
TARGET_UPDATE_FREQ = 1_000      # cada cuántos pasos se copia la target network
LEARNING_STARTS    = 5_000      # pasos de exploración pura antes de entrenar
IMAGE_SAVE_FREQ    = 100_000    # cada cuántos pasos guarda un frame de verificación

N_ACTIONS = 6   # 0=quieto, 1=arriba, 2=abajo, 3=izq, 4=der, 5=disparar


# ── Curriculum manual ─────────────────────────────────────────────────────────

class Curriculum:
    """
    Replica en Python la misma progresión de dificultad del ppo_escenario2.yaml.

    Fases de moneda (coin_spawn_radius):
      CoinCerca  → radio 3 u   (moneda a ≤3 u del jugador)
      CoinMedia  → radio 8 u
      CoinLibre  → -1          (spawn aleatorio normal)

    Fases de enemigos (se activan cuando la moneda ya está en spawn libre):
      SinEnemigos   → spawn_rate 9999 / 9999
      PocosEnemigos → 8 / 20
      NormalEnemigos→ 5 / 12

    El avance de fase se basa en la media de recompensa de los últimos
    `ventana` episodios superando un umbral, después de un mínimo de
    `min_episodios` en esa fase.
    """

    def __init__(self, params_channel: EnvironmentParametersChannel):
        self.ch = params_channel

        # Definición de fases: (nombre, coin_radius, umbral_avance, min_episodios)
        self._fases_moneda = [
            ("CoinCerca",  3.0,  60.0, 100),
            ("CoinMedia",  8.0,  40.0, 100),
            ("CoinLibre", -1.0,  None,   0),   # None = fase final, sin avance
        ]

        # (nombre, normal_rate, big_rate, umbral_avance, min_episodios)
        self._fases_enemigos = [
            ("SinEnemigos",    9999.0, 9999.0, 50.0, 200),
            ("PocosEnemigos",     8.0,   20.0, 20.0, 100),
            ("NormalEnemigos",    5.0,   12.0, None,   0),
        ]

        self._fase_moneda   = 0
        self._fase_enemigos = 0
        self._ep_en_fase_moneda   = 0
        self._ep_en_fase_enemigos = 0

        self._historial = deque(maxlen=50)  # últimos 50 episodios para media móvil

        self._aplicar_params()

    # ── API pública ───────────────────────────────────────────────────────────

    def registrar_episodio(self, reward: float) -> bool:
        """
        Registra el reward de un episodio y comprueba si hay que avanzar de fase.
        Devuelve True si alguna fase cambió (para loguear).
        """
        self._historial.append(reward)
        self._ep_en_fase_moneda   += 1
        self._ep_en_fase_enemigos += 1

        cambio = False
        cambio |= self._intentar_avanzar_moneda()
        cambio |= self._intentar_avanzar_enemigos()
        return cambio

    def estado(self) -> str:
        fm = self._fases_moneda[self._fase_moneda]
        fe = self._fases_enemigos[self._fase_enemigos]
        media = np.mean(self._historial) if self._historial else 0.0
        return (f"Moneda={fm[0]}(radio={fm[1]}) | "
                f"Enemigos={fe[0]} | "
                f"MediaReward={media:.1f}")

    # ── Privado ───────────────────────────────────────────────────────────────

    def _media(self) -> float:
        return float(np.mean(self._historial)) if self._historial else float("-inf")

    def _intentar_avanzar_moneda(self) -> bool:
        idx = self._fase_moneda
        if idx >= len(self._fases_moneda) - 1:
            return False                       # ya en la última fase

        _, _, umbral, min_ep = self._fases_moneda[idx]
        if self._ep_en_fase_moneda >= min_ep and self._media() >= umbral:
            self._fase_moneda += 1
            self._ep_en_fase_moneda = 0
            self._historial.clear()
            self._aplicar_params()
            nombre = self._fases_moneda[self._fase_moneda][0]
            print(f"\n  [Curriculum] ▶ Fase moneda → {nombre}\n")
            return True
        return False

    def _intentar_avanzar_enemigos(self) -> bool:
        idx = self._fase_enemigos
        if idx >= len(self._fases_enemigos) - 1:
            return False

        # Los enemigos solo se activan cuando la moneda ya está en spawn libre
        if self._fase_moneda < len(self._fases_moneda) - 1:
            return False

        _, _, _, umbral, min_ep = self._fases_enemigos[idx]
        if self._ep_en_fase_enemigos >= min_ep and self._media() >= umbral:
            self._fase_enemigos += 1
            self._ep_en_fase_enemigos = 0
            self._historial.clear()
            self._aplicar_params()
            nombre = self._fases_enemigos[self._fase_enemigos][0]
            print(f"\n  [Curriculum] ▶ Fase enemigos → {nombre}\n")
            return True
        return False

    def _aplicar_params(self):
        """Envía los parámetros actuales a Unity vía side channel."""
        _, coin_radius, _, _ = self._fases_moneda[self._fase_moneda]
        _, normal_rate, big_rate, _, _ = self._fases_enemigos[self._fase_enemigos]

        self.ch.set_float_parameter("coin_spawn_radius",       coin_radius)
        self.ch.set_float_parameter("normal_enemy_spawn_rate", normal_rate)
        self.ch.set_float_parameter("big_enemy_spawn_rate",    big_rate)


# ── Red neuronal convolucional ────────────────────────────────────────────────

class DQNRed(nn.Module):
    """
    Red CNN que recibe una imagen 128x128 RGB y devuelve un Q-value por acción.

    Arquitectura:
      3 capas convolucionales para extracción de características visuales +
      2 capas fully-connected para calcular los Q-values.
    """

    def __init__(self, n_acciones: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # (3,128,128) → (32,31,31)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (32,31,31)  → (64,14,14)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (64,14,14)  → (64,12,12)
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 512),
            nn.ReLU(),
            nn.Linear(512, n_acciones),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, C, H, W) normalizado en [0, 1]"""
        return self.fc(self.conv(x))


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Memoria de experiencias (s, a, r, s', done).
    El muestreo aleatorio rompe la correlación temporal que haría inestable el DQN.
    """

    def __init__(self, capacidad: int):
        self.memoria = deque(maxlen=capacidad)

    def guardar(self, estado, accion, recompensa, siguiente_estado, terminado):
        self.memoria.append((estado, accion, recompensa, siguiente_estado, terminado))

    def muestrear(self, n: int):
        return random.sample(self.memoria, n)

    def __len__(self) -> int:
        return len(self.memoria)


# ── Preprocesado de imagen ────────────────────────────────────────────────────

def preprocesar(obs: np.ndarray) -> torch.Tensor:
    """
    gym_unity devuelve (C, H, W) uint8.
    Normaliza a float32 en [0,1] y añade dimensión de batch → (1, C, H, W).
    """
    return torch.tensor(obs.astype(np.float32) / 255.0).unsqueeze(0)


def guardar_imagen(obs_raw: np.ndarray, nombre: str = "frame_camara.png"):
    """Guarda un frame como PNG para verificar lo que ve la red."""
    img = Image.fromarray(np.transpose(obs_raw, (1, 2, 0)).astype(np.uint8))
    img.save(nombre)
    print(f"  → Frame guardado: '{nombre}'")


# ── Bucle de entrenamiento ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DQN Visual con Curriculum — Implementación propia PyTorch")
    print("=" * 60)
    print("Esperando a Unity... Dale al PLAY en el editor.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # ── Conexión con Unity ────────────────────────────────────────────────────
    params_channel = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=42, side_channels=[params_channel])
    env = UnityToGymWrapper(
        unity_env,
        uint8_visual=True,
        flatten_branched=True,
        allow_multiple_obs=False,
    )

    # Curriculum: arranca con moneda cerca y sin enemigos
    curriculum = Curriculum(params_channel)
    print(f"Curriculum inicial: {curriculum.estado()}\n")

    print(f"Observaciones: {env.observation_space}")
    print(f"Acciones:      {env.action_space}\n")

    # ── Redes y optimizador ───────────────────────────────────────────────────
    red_principal = DQNRed(N_ACTIONS).to(device)
    red_objetivo  = DQNRed(N_ACTIONS).to(device)
    red_objetivo.load_state_dict(red_principal.state_dict())
    red_objetivo.eval()

    optimizador = optim.Adam(red_principal.parameters(), lr=LEARNING_RATE)
    buffer      = ReplayBuffer(BUFFER_SIZE)
    epsilon     = EPSILON_START

    paso_global  = 0
    episodio     = 0
    mejor_reward = float("-inf")

    # Guardar primer frame para verificar que la cámara está bien
    obs_inicial = env.reset()
    print(f"Forma observación: {obs_inicial.shape}  (C, H, W)")
    guardar_imagen(obs_inicial, "frame_inicial.png")

    # ── Bucle principal ───────────────────────────────────────────────────────
    while paso_global < TOTAL_STEPS:
        obs       = env.reset()
        estado    = preprocesar(obs).to(device)
        reward_ep = 0.0
        terminado = False
        episodio += 1

        while not terminado:
            # Política epsilon-greedy
            if random.random() < epsilon or paso_global < LEARNING_STARTS:
                accion = env.action_space.sample()
            else:
                with torch.no_grad():
                    accion = red_principal(estado).argmax().item()

            # Paso en Unity
            obs_sig, recompensa, terminado, _ = env.step(accion)
            siguiente_estado = preprocesar(obs_sig).to(device)

            buffer.guardar(estado.cpu(), accion, recompensa,
                           siguiente_estado.cpu(), terminado)

            estado      = siguiente_estado
            reward_ep  += recompensa
            paso_global += 1

            # Frame de verificación periódico
            if paso_global % IMAGE_SAVE_FREQ == 0:
                guardar_imagen(obs_sig, f"frame_paso_{paso_global}.png")

            # Actualizar red principal
            if paso_global >= LEARNING_STARTS and len(buffer) >= BATCH_SIZE:
                muestra = buffer.muestrear(BATCH_SIZE)
                estados, acciones, recompensas, sig_estados, terminados = zip(*muestra)

                estados     = torch.cat(estados).to(device)
                sig_estados = torch.cat(sig_estados).to(device)
                acciones    = torch.tensor(acciones,    dtype=torch.long).to(device)
                recompensas = torch.tensor(recompensas, dtype=torch.float32).to(device)
                terminados  = torch.tensor(terminados,  dtype=torch.float32).to(device)

                q_actual   = red_principal(estados).gather(1, acciones.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_objetivo = recompensas + GAMMA * red_objetivo(sig_estados).max(1).values * (1 - terminados)

                perdida = nn.functional.mse_loss(q_actual, q_objetivo)
                optimizador.zero_grad()
                perdida.backward()
                optimizador.step()

            # Sincronizar target network
            if paso_global % TARGET_UPDATE_FREQ == 0:
                red_objetivo.load_state_dict(red_principal.state_dict())

        # ── Fin de episodio ───────────────────────────────────────────────────
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Registrar en curriculum y comprobar avance de fase
        curriculum.registrar_episodio(reward_ep)

        # Guardar mejor modelo
        if reward_ep > mejor_reward:
            mejor_reward = reward_ep
            torch.save(red_principal.state_dict(), "dqn_mejor.pth")

        # Log cada 10 episodios
        if episodio % 10 == 0:
            print(
                f"Ep {episodio:4d} | Paso {paso_global:7d} | "
                f"Reward: {reward_ep:7.2f} | Mejor: {mejor_reward:7.2f} | "
                f"ε={epsilon:.3f} | {curriculum.estado()}"
            )

    # ── Guardar modelo final ──────────────────────────────────────────────────
    torch.save(red_principal.state_dict(), "dqn_final.pth")
    print("\n¡Entrenamiento finalizado!")
    print("Modelo final:  dqn_final.pth")
    print("Mejor modelo:  dqn_mejor.pth")
    env.close()


if __name__ == "__main__":
    main()
