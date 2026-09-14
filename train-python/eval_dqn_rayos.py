"""
eval_dqn_rayos.py
=================
Evaluación del modelo DQN entrenado con solo rayos (sin brújula).
Carga el .pth con PyTorch y ejecuta N episodios, guardando los
resultados en un .txt con el formato estándar del proyecto.

Configuración Unity antes de lanzar:
  - Current Mode          = DQN_Rayos
  - Current Run Mode      = Inference
  - Behavior Type         = Default       ← Python controla las acciones
  - Model                 = None
  - RayPerceptionSensor3D ACTIVADO (15 rayos, Enemy + Collectible + Terrain)
  - Vector Observation Space Size = 0     (sin brújula)

Uso:
    python eval_dqn_rayos.py
    python eval_dqn_rayos.py --modelo dqn_rayos_mejor.pth --episodios 100
"""

import argparse
import datetime
import random
import time

import numpy as np
import torch
import torch.nn as nn

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper


# ── Constantes de formato ─────────────────────────────────────────────────────

SEP  = "=" * 66
SEP2 = "─" * 66
COIN_REWARD_THRESHOLD = 50.0
MONEDAS_VICTORIA      = 5
N_ACCIONES            = 6


# ── Arquitectura (idéntica al entrenamiento) ──────────────────────────────────

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


# ── Escritura del .txt ────────────────────────────────────────────────────────

def guardar_txt(nombre: str, checkpoint: str,
                episodios: list, ruta: str) -> None:
    n         = len(episodios)
    victorias = sum(1 for e in episodios if e["victoria"])
    rewards   = [e["reward"]  for e in episodios]
    monedas   = [e["monedas"] for e in episodios]
    pasos_l   = [e["pasos"]   for e in episodios]
    durs      = [e["dur"]     for e in episodios]

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(SEP + "\n")
        f.write(f"  RESULTADOS — {nombre}\n")
        f.write(SEP + "\n")
        f.write(f"  Fecha:       {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"  Experimento: {nombre}\n")
        f.write(f"  Episodios:   {n}/{n}  (COMPLETADO)\n\n")
        f.write(f"{'Ep':>4}  {'Pasos':>6}  {'Dur(s)':>8}  {'Reward':>10}  "
                f"{'Monedas':>7}  {'Victoria':>8}\n")
        f.write(SEP2 + "\n")
        for e in episodios:
            v = "SI" if e["victoria"] else "NO"
            f.write(f"{e['ep']:>4}  {e['pasos']:>6}  {e['dur']:>8.1f}  "
                    f"{e['reward']:>+10.2f}  {e['monedas']:>7}  {v:>8}\n")
        f.write(SEP2 + "\n\n")
        f.write(f"  Victorias:                {victorias}/{n}"
                f"  ({100*victorias/n:.0f}%)\n")
        f.write(f"  Reward medio:             {np.mean(rewards):>10.2f}\n")
        f.write(f"  Reward mejor:             {np.max(rewards):>10.2f}\n")
        f.write(f"  Reward peor:              {np.min(rewards):>10.2f}\n")
        f.write(f"  Desv. tipica reward:      {np.std(rewards):>10.2f}\n")
        f.write(f"  Duracion media (s):       {np.mean(durs):>10.2f}\n")
        f.write(f"  Desv. tipica dur. (s):    {np.std(durs):>10.2f}\n")
        f.write(f"  Media monedas/episodio:   {np.mean(monedas):>10.2f}\n")
        f.write(f"  Media pasos/episodio:     {np.mean(pasos_l):>10.2f}\n")
        f.write(SEP + "\n")


# ── Bucle de evaluación ───────────────────────────────────────────────────────

MIN_PASOS = 20   # episodios con menos pasos son fantasmas del gym wrapper y se descartan
DELAY_EP  = 2.0  # segundos de espera entre episodios para que Unity se estabilice

def main(modelo_path: str, n_episodios: int,
         nombre: str, salida: str, epsilon: float) -> None:
    print(SEP)
    print("  DQN_Rayos — Evaluación")
    print(SEP)
    print(f"  Modelo:    {modelo_path}")
    print(f"  Episodios: {n_episodios}")
    print(f"  Epsilon:   {epsilon}  (exploración residual)")
    print(f"  Salida:    {salida}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=42,
                                 side_channels=[params_ch])
    env = UnityToGymWrapper(unity_env, uint8_visual=False,
                            flatten_branched=True, allow_multiple_obs=False)

    # Sin curriculum en evaluación
    params_ch.set_float_parameter("coin_spawn_radius",        -1.0)
    params_ch.set_float_parameter("normal_enemy_spawn_rate",   5.0)
    params_ch.set_float_parameter("big_enemy_spawn_rate",     12.0)

    n_obs = env.observation_space.shape[0]
    red   = DuelingDQN(n_obs, N_ACCIONES).to(device)
    red.load_state_dict(torch.load(modelo_path, map_location=device))
    red.eval()
    print(f"Modelo cargado: {n_obs} observaciones | {N_ACCIONES} acciones\n")

    episodios_data: list = []
    ep = 0

    while ep < n_episodios:
        obs       = env.reset()
        estado    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        reward_ep = 0.0
        pasos     = 0
        monedas   = 0
        terminado = False
        t_inicio  = time.time()

        while not terminado:
            if random.random() < epsilon:
                accion = env.action_space.sample()
            else:
                with torch.no_grad():
                    accion = red(estado).argmax().item()

            obs_sig, r, terminado, _ = env.step(accion)
            estado = torch.tensor(obs_sig, dtype=torch.float32).unsqueeze(0).to(device)

            reward_ep += r
            pasos     += 1
            if r > COIN_REWARD_THRESHOLD:
                monedas += 1

        dur = time.time() - t_inicio

        # Descartar episodios fantasma (gym wrapper genera uno brevísimo tras cada muerte)
        if pasos < MIN_PASOS:
            print(f"  [skip] episodio fantasma descartado ({pasos} pasos)")
            time.sleep(DELAY_EP)
            continue

        ep      += 1
        victoria = monedas >= MONEDAS_VICTORIA
        tag      = "🏆" if victoria else ""
        print(f"Ep {ep:3d} | Pasos: {pasos:5d} | Dur: {dur:6.1f}s | "
              f"Reward: {reward_ep:+8.2f} | Monedas: {monedas} {tag}")

        episodios_data.append({
            "ep": ep, "pasos": pasos, "dur": dur,
            "reward": reward_ep, "monedas": monedas, "victoria": victoria,
        })

        time.sleep(DELAY_EP)

    env.close()
    guardar_txt(nombre, modelo_path, episodios_data, salida)

    victorias = sum(1 for e in episodios_data if e["victoria"])
    rewards   = [e["reward"]  for e in episodios_data]
    monedas_l = [e["monedas"] for e in episodios_data]
    pasos_l   = [e["pasos"]   for e in episodios_data]
    durs      = [e["dur"]     for e in episodios_data]

    print("\n" + SEP2)
    print(f"  Victorias:                {victorias}/{n_episodios}"
          f"  ({100*victorias/n_episodios:.0f}%)")
    print(f"  Reward medio:             {np.mean(rewards):>10.2f}")
    print(f"  Reward mejor:             {np.max(rewards):>10.2f}")
    print(f"  Reward peor:              {np.min(rewards):>10.2f}")
    print(f"  Desv. tipica reward:      {np.std(rewards):>10.2f}")
    print(f"  Duracion media (s):       {np.mean(durs):>10.2f}")
    print(f"  Desv. tipica dur. (s):    {np.std(durs):>10.2f}")
    print(f"  Media monedas/episodio:   {np.mean(monedas_l):>10.2f}")
    print(f"  Media pasos/episodio:     {np.mean(pasos_l):>10.2f}")
    print(SEP2)
    print(f"\n[✓] Resultados guardados en: {salida}")


# ── Argumentos ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación DQN_Rayos")
    parser.add_argument("--modelo",    type=str, default="dqn_rayos_mejor.pth")
    parser.add_argument("--episodios", type=int, default=100)
    parser.add_argument("--nombre",    type=str, default="DQN_Rayos")
    parser.add_argument("--salida",    type=str, default="resultados_DQN_Rayos.txt")
    parser.add_argument("--epsilon",   type=float, default=0.0,
                        help="Exploración residual (0 = greedy puro)")
    args = parser.parse_args()
    main(args.modelo, args.episodios, args.nombre, args.salida, args.epsilon)
