"""
Inferencia del modelo PPO con brújula — Python ejecuta el .onnx.

Python carga el modelo .onnx con onnxruntime, se conecta a Unity y
envía las acciones del modelo en cada paso. Los resultados se guardan
en un archivo .txt para comparación con DQN.

Configuración Unity antes de lanzar:
  - Current Mode                = PPO_Continuous
  - Current Run Mode            = Inference        (TimeScale = 1)
  - Behavior Type               = Default          ← Python controla las acciones
  - Model                       = None             ← Python carga el .onnx
  - RayPerceptionSensor3D       ACTIVADO
  - CamaraFija / CameraSensor   DESACTIVADO
  - Vector Observation Space Size = 2

Requisitos Python:
  pip install onnxruntime

Uso:
  python play_ppo_brujula_v2.py
  python play_ppo_brujula_v2.py --modelo results/PPO_brujula_v3/Shooter_brujula.onnx --episodios 30
  python play_ppo_brujula_v2.py --salida resultados_ppo.txt
"""

import argparse
import os
from datetime import datetime

import numpy as np
import onnxruntime as ort

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)


# ── Carga del modelo .onnx ────────────────────────────────────────────────────

def cargar_modelo(modelo_path: str) -> ort.InferenceSession:
    session = ort.InferenceSession(modelo_path)
    print("Entradas del modelo:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    print("Salidas del modelo:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")
    print()
    return session


def obtener_accion(session: ort.InferenceSession, obs_list):
    """
    Construye el feed dict con las observaciones y ejecuta el modelo onnx.
    Devuelve (continuous_actions, discrete_actions), cualquiera puede ser None.
    El tamaño de action_masks se lee del propio modelo para evitar errores.
    """
    feed = {}
    for i, obs in enumerate(obs_list):
        feed[f"obs_{i}"] = obs.astype(np.float32)

    # Tamaño de masks leído del modelo (no hardcodeado)
    input_names = {inp.name for inp in session.get_inputs()}
    if "action_masks" in input_names:
        masks_info = next(i for i in session.get_inputs() if i.name == "action_masks")
        n_disc = int(masks_info.shape[1])   # e.g. 2 para PPO, 6 para DQN
        feed["action_masks"] = np.ones(
            (obs_list[0].shape[0], n_disc), dtype=np.float32
        )

    outputs = session.run(None, feed)
    output_names = [o.name for o in session.get_outputs()]

    continuous_actions = None
    discrete_actions   = None
    for name, val in zip(output_names, outputs):
        if "continuous_actions" in name and "deterministic" not in name:
            continuous_actions = val
        elif "discrete_actions" in name and "deterministic" not in name:
            discrete_actions = val

    return continuous_actions, discrete_actions


# ── Guardado de resultados ────────────────────────────────────────────────────

def guardar_resultados(
    salida: str,
    modelo_path: str,
    n_episodios: int,
    rewards: list,
    monedas: list,
    pasos: list,
    victorias: int,
):
    lineas = []
    lineas.append("=" * 60)
    lineas.append("  RESULTADOS INFERENCIA — PPO Brújula")
    lineas.append("=" * 60)
    lineas.append(f"  Fecha:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lineas.append(f"  Modelo:    {modelo_path}")
    lineas.append(f"  Episodios: {n_episodios}")
    lineas.append("")
    lineas.append(f"{'Ep':>4}  {'Pasos':>6}  {'Reward':>9}  {'Monedas':>8}  {'Victoria':>8}")
    lineas.append("─" * 46)
    for i, (r, m, p) in enumerate(zip(rewards, monedas, pasos), 1):
        victoria = "SI" if m >= 5 else "NO"
        lineas.append(f"{i:>4}  {p:>6}  {r:>9.2f}  {m:>8}  {victoria:>8}")
    lineas.append("─" * 46)
    lineas.append("")
    lineas.append(f"  Victorias (≥5 monedas):   {victorias}/{n_episodios}  ({100 * victorias / n_episodios:.0f}%)")
    lineas.append(f"  Reward medio:             {np.mean(rewards):8.2f}")
    lineas.append(f"  Reward mejor:             {np.max(rewards):8.2f}")
    lineas.append(f"  Reward peor:              {np.min(rewards):8.2f}")
    lineas.append(f"  Desv. típica reward:      {np.std(rewards):8.2f}")
    lineas.append(f"  Media monedas/episodio:   {np.mean(monedas):8.2f}")
    lineas.append(f"  Media pasos/episodio:     {np.mean(pasos):8.2f}")
    lineas.append("=" * 60)

    texto = "\n".join(lineas)
    print(texto)

    with open(salida, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(f"\n  Resultados guardados en: {salida}")


# ── Bucle de inferencia ───────────────────────────────────────────────────────

def main(modelo_path: str, n_episodios: int, salida: str):
    print("=" * 60)
    print("  PPO Brújula — Inferencia v2  (Python + onnxruntime)")
    print("=" * 60)
    print(f"  Modelo:    {modelo_path}")
    print(f"  Episodios: {n_episodios}")
    print(f"  Salida:    {salida}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    session = cargar_modelo(modelo_path)

    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=0, side_channels=[params_ch])

    # Condiciones de inferencia: mapa completo + enemigos activos
    params_ch.set_float_parameter("coin_spawn_radius",       -1.0)
    params_ch.set_float_parameter("normal_enemy_spawn_rate",  5.0)
    params_ch.set_float_parameter("big_enemy_spawn_rate",    12.0)

    unity_env.reset()
    behavior_name = list(unity_env.behavior_specs)[0]
    spec          = unity_env.behavior_specs[behavior_name]
    print(f"Behavior:    {behavior_name}")
    print(f"Obs shapes:  {[o.shape for o in spec.observation_specs]}")
    print(f"Action spec: {spec.action_spec}\n")

    rewards_totales = []
    monedas_totales = []
    pasos_totales   = []
    victorias       = 0

    ep        = 0
    reward_ep = 0.0
    pasos     = 0
    monedas   = 0

    while ep < n_episodios:
        unity_env.step()
        decision_steps, terminal_steps = unity_env.get_steps(behavior_name)

        # ── Agente vivo: enviar acción ─────────────────────────────────────
        if len(decision_steps) > 0:
            obs_list = [o for o in decision_steps.obs]
            continuous_actions, discrete_actions = obtener_accion(session, obs_list)

            if continuous_actions is not None and discrete_actions is not None:
                action_tuple = ActionTuple(
                    continuous=continuous_actions,
                    discrete=discrete_actions.astype(np.int32),
                )
            elif continuous_actions is not None:
                action_tuple = ActionTuple(continuous=continuous_actions)
            else:
                action_tuple = ActionTuple(
                    discrete=discrete_actions.astype(np.int32)
                )

            unity_env.set_actions(behavior_name, action_tuple)

            r = float(decision_steps.reward[0])
            reward_ep += r
            pasos     += 1
            if r > 0.5:
                monedas += 1

        # ── Fin de episodio ────────────────────────────────────────────────
        if len(terminal_steps) > 0:
            r = float(terminal_steps.reward[0])
            reward_ep += r
            pasos     += 1
            if r > 0.5:
                monedas += 1

            gano = monedas >= 5
            if gano:
                victorias += 1

            rewards_totales.append(reward_ep)
            monedas_totales.append(monedas)
            pasos_totales.append(pasos)

            print(f"Ep {ep + 1:3d} | Pasos: {pasos:5d} | Reward: {reward_ep:8.2f} | "
                  f"Monedas: {monedas} {'🏆' if gano else ''}")

            ep        += 1
            reward_ep  = 0.0
            pasos      = 0
            monedas    = 0

    unity_env.close()

    guardar_resultados(
        salida, modelo_path, n_episodios,
        rewards_totales, monedas_totales, pasos_totales, victorias,
    )


# ── Argumentos de línea de comandos ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia PPO Brújula v2")
    parser.add_argument(
        "--modelo",
        type=str,
        default="results/PPO_brujula_v3/Shooter_brujula.onnx",
        help="Ruta al .onnx exportado por mlagents-learn",
    )
    parser.add_argument(
        "--episodios",
        type=int,
        default=20,
        help="Número de episodios de evaluación  (default: 20)",
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="resultados_ppo_brujula.txt",
        help="Archivo .txt donde guardar los resultados  (default: resultados_ppo_brujula.txt)",
    )
    args = parser.parse_args()
    main(args.modelo, args.episodios, args.salida)
