"""
FASE 2 — Supervivencia con monedas (con enemigos).

Puede ejecutarse de dos formas:
  A) Con modelo de Fase 1: carga dqn_fase1_mejor.pth → epsilon=0.4
  B) Sin Fase 1 (standalone): arranca desde cero → epsilon=1.0

El script detecta automáticamente cuál aplica.

Curriculum con buffer limpio en cada transición:
  Paso         0 –   400 000 : radio 5,  enemigos escasos   (base)
  Paso   400 000 –   800 000 : radio 10, enemigos escasos   (+ distancia)
  Paso   800 000 – 1 200 000 : radio 10, todos los enemigos (+ presión)
  Paso 1 200 000 – 2 000 000 : libre,    todos los enemigos (fase final)

  En cada transición se vacía el replay buffer para evitar que experiencias
  de la distribución anterior envenenen el aprendizaje de la nueva.

Configuración Unity:
  - Current Mode          = DQN_Brujula
  - Current Run Mode      = TrainDQN
  - Behavior Type         = Default
  - Model                 = None
  - RayPerceptionSensor3D ACTIVADO
  - Vector Obs Size       = 2
  - Discrete Actions      = 1 rama de 6

Salida:
  dqn_fase2_mejor.pth   ← mejor episodio fase final (usar en inferencia)
  dqn_fase2_final.pth   ← último checkpoint
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

TOTAL_STEPS         = 2_000_000
BUFFER_SIZE         = 300_000
BATCH_SIZE          = 64
LEARNING_RATE       = 2e-4
GAMMA               = 0.99
# Epsilon se fija en main() según si hay modelo de Fase 1 o no
EPSILON_CON_FASE1   = 0.40   # ya sabe coger monedas → menos exploración necesaria
EPSILON_STANDALONE  = 1.00   # sin preentrenamiento → exploración completa
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 1_600_000
TARGET_UPDATE_FREQ  = 1_000
LEARNING_STARTS     = 2_000
GRAD_CLIP_NORM      = 5.0
REWARD_CLIP         = 10.0
LOG_FREQ            = 10
N_ACTIONS           = 6

FASE1_MODEL = "dqn_fase1_mejor.pth"


# ── Curriculum con enemigos ───────────────────────────────────────────────────

class CurriculumFase2:
    """
    Paso         0 –   400 000 : radio 5,  enemigos escasos
    Paso   400 000 –   800 000 : radio 10, enemigos escasos
    Paso   800 000 – 1 200 000 : radio 10, todos los enemigos
    Paso 1 200 000 – 2 000 000 : libre,    todos los enemigos (fase final)

    El buffer se vacía en cada transición para evitar interferencias
    entre distribuciones de estados muy distintas.
    """
    _FASES = [
        (        0, "Radio5/EnemigosEscasos",   5.0, 15.0, 9999.0),
        (  400_000, "Radio10/EnemigosEscasos", 10.0, 15.0, 9999.0),
        (  800_000, "Radio10/TodosEnemigos",   10.0,  5.0,    5.0),
        (1_200_000, "Libre/TodosEnemigos",     -1.0,  5.0,    5.0),
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
            print(f"\n  [Curriculum F2] ▶ Paso {paso:>7d} → {self._FASES[self._fase][1]}\n")

    def fase_actual(self): return self._fase
    def nombre(self):      return self._FASES[self._fase][1]

    def estado(self, paso):
        nombre = self._FASES[self._fase][1]
        siguiente = self._fase + 1
        if siguiente < len(self._FASES):
            return f"{nombre} | Próximo en {self._FASES[siguiente][0] - paso:>7d} pasos"
        return f"{nombre} | Fase final"

    def tiene_enemigos(self):
        """True desde la primera fase con enemigos reales (fase 2 en adelante)."""
        return self._fase >= 2

    def es_nueva_fase(self, fase_anterior):
        """True si acaba de avanzar de fase (para limpiar el buffer)."""
        return self._fase != fase_anterior

    def _aplicar(self):
        _, _, coin_r, norm_r, big_r = self._FASES[self._fase]
        self.ch.set_float_parameter("coin_spawn_radius",       coin_r)
        self.ch.set_float_parameter("normal_enemy_spawn_rate", norm_r)
        self.ch.set_float_parameter("big_enemy_spawn_rate",    big_r)


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
    run = f"DQN_Fase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    tiene_fase1 = os.path.exists(FASE1_MODEL)
    eps_inicio  = EPSILON_CON_FASE1 if tiene_fase1 else EPSILON_STANDALONE
    lr          = 2e-4 if tiene_fase1 else 3e-4   # más conservador si hay preentrenamiento

    print("=" * 60)
    print("  FASE 2 — Supervivencia con monedas (con enemigos)")
    print(f"  Run: {run}")
    if tiene_fase1:
        print(f"  Modo: Transfer desde {FASE1_MODEL}  (ε₀={eps_inicio})")
    else:
        print(f"  Modo: Standalone sin Fase 1  (ε₀={eps_inicio})")
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
    curriculum = CurriculumFase2(params_ch)

    n_obs = env.observation_space.shape[0]
    print(f"Obs: {n_obs}  |  Acciones: {N_ACTIONS}  |  Curriculum: {curriculum.estado(0)}\n")

    red    = DuelingDQN(n_obs, N_ACTIONS).to(device)
    target = DuelingDQN(n_obs, N_ACTIONS).to(device)

    if tiene_fase1:
        red.load_state_dict(torch.load(FASE1_MODEL, map_location=device))
        print(f"✅ Pesos cargados desde {FASE1_MODEL}\n")

    target.load_state_dict(red.state_dict())
    target.eval()

    opt   = optim.Adam(red.parameters(), lr=lr)
    buf   = ReplayBuffer(BUFFER_SIZE)
    eps   = eps_inicio

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
            r_clip = np.clip(r, -REWARD_CLIP, REWARD_CLIP)

            sig = torch.tensor(obs2, dtype=torch.float32).unsqueeze(0)
            buf.guardar(estado.cpu(), accion, r_clip, sig, done)

            estado    = sig.to(device)
            rew_ep   += r
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

        # Epsilon: decay lineal desde epsilon inicial hasta EPSILON_END
        progreso = paso_global / EPSILON_DECAY_STEPS
        eps = max(EPSILON_END, eps_inicio - (eps_inicio - EPSILON_END) * progreso)

        fase_antes = curriculum.fase_actual()
        curriculum.actualizar(paso_global)
        if curriculum.es_nueva_fase(fase_antes):
            buf = ReplayBuffer(BUFFER_SIZE)
            print(f"  [Buffer] Limpiado al entrar en fase {curriculum.nombre()}\n")

        victoria    = monedas >= 5
        # Guardar mejor modelo en cualquier fase con enemigos activos
        nuevo_mejor = curriculum.tiene_enemigos() and rew_ep > mejor_reward
        if nuevo_mejor:
            mejor_reward = rew_ep
            torch.save(red.state_dict(), "dqn_fase2_mejor.pth")

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
                  f"Monedas: {monedas} | ε={eps:.3f} | Loss: {media_l:.1f} | "
                  f"{curriculum.estado(paso_global)} {tag}{best_tag}")

    torch.save(red.state_dict(), "dqn_fase2_final.pth")
    writer.close()
    csv_f.close()
    env.close()
    print("\n✅ FASE 2 completada")
    print("  Mejor modelo → dqn_fase2_mejor.pth  (usar en inferencia)")
    print("  Final modelo → dqn_fase2_final.pth")


if __name__ == "__main__":
    main()
