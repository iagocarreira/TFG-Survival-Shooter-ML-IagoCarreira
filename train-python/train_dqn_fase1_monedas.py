"""
FASE 1 — Experto en monedas (sin enemigos).

El agente aprende ÚNICAMENTE a recoger monedas. Sin enemigos, sin presión.
Curriculum de radio progresivo: 2 → 5 → 10.
NO hay fase "Libre": la moneda siempre dentro de radio 10.
La Fase 2 ya se encarga del mapa completo con enemigos.

Correcciones respecto a versiones anteriores:
  - Recompensas NORMALIZADAS ÷ 100 antes de entrar al buffer
    (moneda = +1, penalización = −0.0001/paso → Q-valores acotados)
  - Fase "Libre" eliminada: saltarla aquí evita el colapso de Q-valores
    que se producía al cambiar bruscamente la distribución de estados.
  - Curriculum termina en Radio10 → se guardan los pesos como base de Fase 2.

Configuración Unity:
  - Current Mode          = DQN_Brujula
  - Current Run Mode      = TrainDQN
  - Behavior Type         = Default
  - Model                 = None
  - RayPerceptionSensor3D ACTIVADO
  - Vector Obs Size       = 2
  - Discrete Actions      = 1 rama de 6

Salida:
  dqn_fase1_mejor.pth   ← mejor episodio en Radio10 (usar como entrada de Fase 2)
  dqn_fase1_final.pth   ← último checkpoint
"""

import csv, os, random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from gym_unity.envs import UnityToGymWrapper


# ── Hiperparámetros ───────────────────────────────────────────────────────────

TOTAL_STEPS         = 1_200_000   # termina en Radio10; sin fase Libre
BUFFER_SIZE         = 200_000
BATCH_SIZE          = 64
LEARNING_RATE       = 3e-4
GAMMA               = 0.99
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 1_000_000   # llega a 0.05 cerca del final
TARGET_UPDATE_FREQ  = 1_000
LEARNING_STARTS     = 2_000
GRAD_CLIP_NORM      = 5.0
REWARD_SCALE        = 100.0       # dividir recompensa antes del buffer
                                  # → moneda = +1, victoria = +2, penalty ≈ 0
LOG_FREQ            = 10
N_ACTIONS           = 6


# ── Curriculum sin enemigos ───────────────────────────────────────────────────

class CurriculumFase1:
    """
    Radio de moneda creciente. Sin enemigos en toda la fase.
    NO hay fase Libre; el colapso de Q-valores ocurre con spawns alejados.

    Paso       0 –  400 000 : radio 2  (moneda muy cerca)
    Paso 400 000 –  800 000 : radio 5  (distancia media)
    Paso 800 000 – 1 200 000: radio 10 (distancia larga) ← fase final
    """
    _FASES = [
        (      0, "Radio2",  2.0),
        (400_000, "Radio5",  5.0),
        (800_000, "Radio10", 10.0),
    ]

    def __init__(self, ch):
        self.ch = ch
        self._fase = 0
        self._aplicar()

    def actualizar(self, paso):
        siguiente = self._fase + 1
        if siguiente < len(self._FASES) and paso >= self._FASES[siguiente][0]:
            self._fase = siguiente
            self._aplicar()
            print(f"\n  [Curriculum F1] ▶ Paso {paso:>7d} → {self._FASES[self._fase][1]}\n")

    def fase_actual(self): return self._fase
    def nombre(self):      return self._FASES[self._fase][1]

    def estado(self, paso):
        nombre = self._FASES[self._fase][1]
        siguiente = self._fase + 1
        if siguiente < len(self._FASES):
            return f"{nombre} | Próximo en {self._FASES[siguiente][0] - paso:>7d} pasos"
        return f"{nombre} | Fase final"

    def _aplicar(self):
        _, _, coin_r = self._FASES[self._fase]
        self.ch.set_float_parameter("coin_spawn_radius",       coin_r)
        self.ch.set_float_parameter("normal_enemy_spawn_rate", 9999.0)  # sin enemigos
        self.ch.set_float_parameter("big_enemy_spawn_rate",    9999.0)


# ── Red y Buffer ──────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    def __init__(self, n_obs, n_acc):
        super().__init__()
        self.tronco = nn.Sequential(
            nn.Linear(n_obs, 256), nn.ReLU(),
            nn.Linear(256, 256),   nn.ReLU(),
        )
        self.valor   = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.ventaja = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_acc))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.tronco(x)
        v = self.valor(h)
        a = self.ventaja(h)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, cap):
        self.mem = deque(maxlen=cap)
    def guardar(self, s, a, r, s2, d): self.mem.append((s, a, r, s2, d))
    def muestrear(self, n): return random.sample(self.mem, n)
    def __len__(self): return len(self.mem)


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def main():
    run = f"DQN_Fase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("=" * 60)
    print("  FASE 1 — Experto en monedas (sin enemigos, sin Libre)")
    print(f"  Run: {run}")
    print("=" * 60)
    print("Dale al PLAY en Unity...\n")

    writer = SummaryWriter(f"runs/{run}")
    os.makedirs("logs", exist_ok=True)
    csv_f  = open(f"logs/{run}.csv", "w", newline="", encoding="utf-8")
    csv_w  = csv.writer(csv_f)
    csv_w.writerow(["ep","paso","reward","monedas","pasos_ep","eps","fase","victoria"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}  |  TensorBoard: tensorboard --logdir runs/\n")

    params_ch = EnvironmentParametersChannel()
    env = UnityToGymWrapper(
        UnityEnvironment(file_name=None, seed=42, side_channels=[params_ch]),
        uint8_visual=False, flatten_branched=True, allow_multiple_obs=False,
    )
    curriculum = CurriculumFase1(params_ch)

    n_obs = env.observation_space.shape[0]
    print(f"Obs: {n_obs}  |  Acciones: {N_ACTIONS}  |  Curriculum: {curriculum.estado(0)}")
    print(f"Normalización de recompensa: ÷ {REWARD_SCALE}  (moneda=+1, victoria=+2)\n")

    red   = DuelingDQN(n_obs, N_ACTIONS).to(device)
    target = DuelingDQN(n_obs, N_ACTIONS).to(device)
    target.load_state_dict(red.state_dict())
    target.eval()

    opt    = optim.Adam(red.parameters(), lr=LEARNING_RATE)
    buf    = ReplayBuffer(BUFFER_SIZE)
    eps    = EPSILON_START

    paso_global  = 0
    episodio     = 0
    mejor_reward = float("-inf")
    ventana_r    = deque(maxlen=50)
    ventana_m    = deque(maxlen=50)
    loss_acum    = 0.0
    pasos_train  = 0

    while paso_global < TOTAL_STEPS:
        obs      = env.reset()
        estado   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        rew_ep   = 0.0
        done     = False
        episodio += 1
        monedas  = 0
        pasos_ep = 0

        while not done:
            if random.random() < eps or paso_global < LEARNING_STARTS:
                accion = env.action_space.sample()
            else:
                with torch.no_grad():
                    accion = red(estado).argmax().item()

            obs2, r, done, _ = env.step(accion)
            # Normalizar recompensa antes del buffer → Q-valores acotados
            r_norm = r / REWARD_SCALE

            sig = torch.tensor(obs2, dtype=torch.float32).unsqueeze(0)
            buf.guardar(estado.cpu(), accion, r_norm, sig, done)

            estado    = sig.to(device)
            rew_ep   += r          # reward real (sin normalizar) para logging
            paso_global += 1
            pasos_ep  += 1
            if r > 0.5: monedas += 1

            if paso_global >= LEARNING_STARTS and len(buf) >= BATCH_SIZE:
                s, a, rr, s2, dd = zip(*buf.muestrear(BATCH_SIZE))
                s  = torch.cat(s).to(device)
                s2 = torch.cat(s2).to(device)
                a  = torch.tensor(a,  dtype=torch.long).to(device)
                rr = torch.tensor(rr, dtype=torch.float32).to(device)
                dd = torch.tensor(dd, dtype=torch.float32).to(device)

                q = red(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    a2 = red(s2).argmax(1).unsqueeze(1)
                    qt = target(s2).gather(1, a2).squeeze(1)
                    q_obj = rr + GAMMA * qt * (1 - dd)

                loss = nn.functional.smooth_l1_loss(q, q_obj)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(red.parameters(), GRAD_CLIP_NORM)
                opt.step()
                loss_acum   += loss.item()
                pasos_train += 1

            if paso_global % TARGET_UPDATE_FREQ == 0:
                target.load_state_dict(red.state_dict())

        eps = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * paso_global / EPSILON_DECAY_STEPS)
        curriculum.actualizar(paso_global)

        victoria    = monedas >= 5
        nuevo_mejor = rew_ep > mejor_reward
        if nuevo_mejor:
            mejor_reward = rew_ep
            torch.save(red.state_dict(), "dqn_fase1_mejor.pth")

        ventana_r.append(rew_ep)
        ventana_m.append(monedas)

        csv_w.writerow([episodio, paso_global, f"{rew_ep:.2f}", monedas,
                        pasos_ep, f"{eps:.4f}", curriculum.fase_actual(), int(victoria)])
        csv_f.flush()

        if episodio % LOG_FREQ == 0:
            media_r = np.mean(ventana_r)
            media_m = np.mean(ventana_m)
            media_l = loss_acum / max(pasos_train, 1)
            loss_acum = pasos_train = 0

            writer.add_scalar("Reward/ep",       rew_ep,   episodio)
            writer.add_scalar("Reward/media50",  media_r,  episodio)
            writer.add_scalar("Monedas/media50", media_m,  episodio)
            writer.add_scalar("Epsilon",         eps,      episodio)
            writer.add_scalar("Loss",            media_l,  episodio)
            writer.add_scalar("Curriculum/fase", curriculum.fase_actual(), episodio)

            tag = "🏆" if victoria else ""
            best_tag = " ★" if nuevo_mejor else ""
            print(f"Ep {episodio:4d} | Paso {paso_global:7d} | "
                  f"Rew: {rew_ep:7.1f} | M50: {media_r:7.1f} | "
                  f"Monedas: {monedas} | ε={eps:.3f} | Loss: {media_l:.4f} | "
                  f"{curriculum.estado(paso_global)} {tag}{best_tag}")

    torch.save(red.state_dict(), "dqn_fase1_final.pth")
    writer.close()
    csv_f.close()
    env.close()
    print("\n✅ FASE 1 completada")
    print("  Mejor modelo → dqn_fase1_mejor.pth  (usar como entrada de Fase 2)")
    print("  Final modelo → dqn_fase1_final.pth")


if __name__ == "__main__":
    main()
