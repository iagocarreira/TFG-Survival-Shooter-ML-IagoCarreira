"""
eval_ppo_aimbot.py
==================
Evaluación del modelo PPO_Aimbot entrenado.
Carga el .onnx exportado por mlagents-learn y ejecuta N episodios,
guardando los resultados en un .txt con el formato estándar del proyecto.

Configuración Unity antes de lanzar:
  - Current Mode          = PPO_Aimbot
  - Current Run Mode      = Inference
  - Behavior Type         = Default       ← Python controla las acciones
  - Model                 = None          ← Python carga el .onnx
  - RayPerceptionSensor3D ACTIVADO (15 rayos, Enemy + Collectible + Terrain)
  - Vector Observation Space Size = 2     (posición moneda X,Z)

Uso:
    python eval_ppo_aimbot.py
    python eval_ppo_aimbot.py --modelo results/PPO_aimbot/Shooter.onnx --episodios 100
"""

import argparse
import datetime
import os

import numpy as np
import onnxruntime as ort

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)


# ── Constantes de formato ─────────────────────────────────────────────────────

SEP  = "=" * 66
SEP2 = "─" * 66
COIN_REWARD_THRESHOLD = 50.0   # recompensa > 50 → moneda recogida (+100 nominal)
MONEDAS_VICTORIA      = 5


# ── Carga del modelo ONNX ─────────────────────────────────────────────────────

def cargar_modelo(path: str) -> ort.InferenceSession:
    session = ort.InferenceSession(path)
    return session


def obtener_accion(session: ort.InferenceSession, obs_list: list):
    """Construye el feed-dict y ejecuta el modelo ONNX."""
    feed = {f"obs_{i}": o.astype(np.float32) for i, o in enumerate(obs_list)}
    input_names = {inp.name for inp in session.get_inputs()}
    if "action_masks" in input_names:
        n_disc = 1
        feed["action_masks"] = np.ones((obs_list[0].shape[0], n_disc),
                                       dtype=np.float32)
    outputs = session.run(None, feed)
    output_names = [o.name for o in session.get_outputs()]
    continuous_actions = None
    discrete_actions   = None
    for name, val in zip(output_names, outputs):
        if "continuous_actions" in name:
            continuous_actions = val
        elif "discrete_actions" in name:
            discrete_actions = val
    return continuous_actions, discrete_actions


# ── Escritura del .txt ────────────────────────────────────────────────────────

def guardar_txt(nombre: str, checkpoint: str,
                episodios: list, ruta: str) -> None:
    n         = len(episodios)
    victorias = sum(1 for e in episodios if e["victoria"])
    rewards   = [e["reward"]  for e in episodios]
    monedas   = [e["monedas"] for e in episodios]
    pasos_l   = [e["pasos"]   for e in episodios]

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(SEP + "\n")
        f.write(f"  RESULTADOS — {nombre}\n")
        f.write(SEP + "\n")
        f.write(f"  Fecha:        {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"  Checkpoint:   {checkpoint}\n")
        f.write(f"  Episodios:    {n}/{n}  (COMPLETADO)\n\n")
        f.write(f"{'Ep':>4}  {'Pasos':>6}  {'R_Shooter':>10}  "
                f"{'Monedas':>8}  {'Victoria':>8}\n")
        f.write(SEP2 + "\n")
        for e in episodios:
            v = "SI" if e["victoria"] else "NO"
            f.write(f"{e['ep']:>4}  {e['pasos']:>6}  {e['reward']:>+10.2f}  "
                    f"{e['monedas']:>8}  {v:>8}\n")
        f.write(SEP2 + "\n\n")
        f.write(f"  Victorias:              {victorias}/{n}"
                f"  ({100*victorias/n:.0f}%)\n")
        f.write(f"  R_Shooter — media: {np.mean(rewards):>8.2f}"
                f"  mejor: {np.max(rewards):>8.2f}"
                f"  peor: {np.min(rewards):>8.2f}\n")
        f.write(f"  Media monedas/episodio: {np.mean(monedas):.2f}\n")
        f.write(f"  Media pasos/episodio:   {np.mean(pasos_l):.0f}\n")
        f.write(SEP + "\n")


# ── Bucle de evaluación ───────────────────────────────────────────────────────

def main(modelo_path: str, n_episodios: int, nombre: str, salida: str) -> None:
    print(SEP)
    print("  PPO_Aimbot — Evaluación")
    print(SEP)
    print(f"  Modelo:    {modelo_path}")
    print(f"  Episodios: {n_episodios}")
    print(f"  Salida:    {salida}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    session   = cargar_modelo(modelo_path)
    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=42,
                                 side_channels=[params_ch])

    # Sin curriculum en evaluación: spawn normal, enemigos activos
    params_ch.set_float_parameter("coin_spawn_radius",        -1.0)
    params_ch.set_float_parameter("normal_enemy_spawn_rate",   5.0)
    params_ch.set_float_parameter("big_enemy_spawn_rate",     12.0)

    unity_env.reset()
    behavior_name = list(unity_env.behavior_specs)[0]

    episodios_data: list = []
    ep        = 0
    reward_ep = 0.0
    pasos     = 0
    monedas   = 0

    while ep < n_episodios:
        unity_env.step()
        decision_steps, terminal_steps = unity_env.get_steps(behavior_name)

        # ── Paso normal: el agente necesita una acción ────────────────────────
        if len(decision_steps) > 0:
            obs_list = list(decision_steps.obs)
            cont, disc = obtener_accion(session, obs_list)

            if cont is not None and disc is not None:
                action = ActionTuple(continuous=cont,
                                     discrete=disc.astype(np.int32))
            elif cont is not None:
                action = ActionTuple(continuous=cont)
            else:
                action = ActionTuple(discrete=disc.astype(np.int32))

            unity_env.set_actions(behavior_name, action)

            r = float(decision_steps.reward[0])
            reward_ep += r
            pasos     += 1
            if r > COIN_REWARD_THRESHOLD:
                monedas += 1

        # ── Fin de episodio ───────────────────────────────────────────────────
        if len(terminal_steps) > 0:
            r = float(terminal_steps.reward[0])
            reward_ep += r
            pasos     += 1

            victoria = monedas >= MONEDAS_VICTORIA
            if victoria:
                monedas = max(monedas, MONEDAS_VICTORIA)

            tag = "🏆" if victoria else ""
            print(f"Ep {ep+1:3d} | Pasos: {pasos:5d} | "
                  f"Reward: {reward_ep:+8.2f} | Monedas: {monedas} {tag}")

            episodios_data.append({
                "ep": ep + 1, "pasos": pasos,
                "reward": reward_ep, "monedas": monedas, "victoria": victoria,
            })
            ep       += 1
            reward_ep = 0.0
            pasos     = 0
            monedas   = 0

    unity_env.close()
    guardar_txt(nombre, modelo_path, episodios_data, salida)

    # ── Resumen en consola ────────────────────────────────────────────────────
    victorias = sum(1 for e in episodios_data if e["victoria"])
    rewards   = [e["reward"]  for e in episodios_data]
    monedas_l = [e["monedas"] for e in episodios_data]
    pasos_l   = [e["pasos"]   for e in episodios_data]

    print("\n" + SEP2)
    print(f"  Victorias:              {victorias}/{n_episodios}"
          f"  ({100*victorias/n_episodios:.0f}%)")
    print(f"  R_Shooter — media: {np.mean(rewards):>8.2f}"
          f"  mejor: {np.max(rewards):>8.2f}"
          f"  peor: {np.min(rewards):>8.2f}")
    print(f"  Media monedas/episodio: {np.mean(monedas_l):.2f}")
    print(f"  Media pasos/episodio:   {np.mean(pasos_l):.0f}")
    print(SEP2)
    print(f"\n[✓] Resultados guardados en: {salida}")


# ── Argumentos ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación PPO_Aimbot")
    parser.add_argument(
        "--modelo",
        type=str,
        default="results/PPO_aimbot/Shooter.onnx",
        help="Ruta al .onnx exportado por mlagents-learn",
    )
    parser.add_argument(
        "--episodios", type=int, default=100,
        help="Número de episodios de evaluación (default: 100)",
    )
    parser.add_argument(
        "--nombre", type=str, default="PPO_Aimbot",
        help="Nombre que aparecerá en el encabezado del .txt",
    )
    parser.add_argument(
        "--salida", type=str, default="resultados_PPO_Aimbot.txt",
        help="Ruta del archivo de salida (default: resultados_PPO_Aimbot.txt)",
    )
    args = parser.parse_args()
    main(args.modelo, args.episodios, args.nombre, args.salida)
