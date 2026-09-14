"""
Inferencia del modelo PPO con brújula — Python ejecuta el .onnx.

Python carga el modelo .onnx con onnxruntime, se conecta a Unity y
manda las acciones del modelo en cada paso. Así obtenemos métricas
comparables a los scripts de DQN.

Configuración Unity antes de lanzar:
  - Current Mode          = PPO_Continuous
  - Current Run Mode      = Inference    (TimeScale = 1)
  - Behavior Type         = Default      ← Python controla las acciones
  - Model                 = None         ← Python carga el .onnx
  - RayPerceptionSensor3D ACTIVADO
  - Vector Observation Space Size = 2

Requisitos Python:
  pip install onnxruntime
"""

import argparse
import numpy as np
import onnxruntime as ort

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)


def cargar_modelo(modelo_path: str):
    session = ort.InferenceSession(modelo_path)
    print("Entradas del modelo:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    print("Salidas del modelo:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")
    print()
    return session


def obtener_accion(session, obs_list):
    """Construye el feed dict y ejecuta el modelo onnx."""
    feed = {}
    for i, obs in enumerate(obs_list):
        feed[f"obs_{i}"] = obs.astype(np.float32)

    # Algunos modelos de ML-Agents requieren action_masks
    input_names = {inp.name for inp in session.get_inputs()}
    if "action_masks" in input_names:
        # Sin máscaras: todas las acciones permitidas
        n_disc = 6  # tamaño de la rama discreta
        feed["action_masks"] = np.ones((obs_list[0].shape[0], n_disc), dtype=np.float32)

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


def main(modelo_path: str, n_episodios: int):
    print("=" * 60)
    print("  PPO Brújula — Inferencia (Python + onnxruntime)")
    print("=" * 60)
    print(f"  Modelo:    {modelo_path}")
    print(f"  Episodios: {n_episodios}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    session = cargar_modelo(modelo_path)

    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=0, side_channels=[params_ch])

    # Sin curriculum en inferencia
    params_ch.set_float_parameter("coin_spawn_radius",       -1.0)
    params_ch.set_float_parameter("normal_enemy_spawn_rate",  5.0)
    params_ch.set_float_parameter("big_enemy_spawn_rate",    12.0)

    unity_env.reset()
    behavior_name = list(unity_env.behavior_specs)[0]
    spec          = unity_env.behavior_specs[behavior_name]
    print(f"Behavior: {behavior_name}")
    print(f"Obs shapes: {[o.shape for o in spec.observation_specs]}")
    print(f"Action spec: {spec.action_spec}\n")

    rewards_totales = []
    monedas_totales = []
    victorias       = 0

    ep        = 0
    reward_ep = 0.0
    pasos     = 0
    monedas   = 0

    while ep < n_episodios:
        unity_env.step()
        decision_steps, terminal_steps = unity_env.get_steps(behavior_name)

        # Agente vivo — necesita acción
        if len(decision_steps) > 0:
            obs_list = [o for o in decision_steps.obs]
            continuous_actions, discrete_actions = obtener_accion(session, obs_list)

            # Construir ActionTuple según lo que el modelo devuelva
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

        # Fin de episodio
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

            print(f"Ep {ep+1:3d} | Pasos: {pasos:5d} | Reward: {reward_ep:8.2f} | "
                  f"Monedas: {monedas} {'🏆' if gano else ''}")

            ep       += 1
            reward_ep = 0.0
            pasos     = 0
            monedas   = 0

    print("\n" + "─" * 60)
    print("  RESUMEN")
    print("─" * 60)
    print(f"  Episodios jugados:       {n_episodios}")
    print(f"  Victorias (5 monedas):   {victorias}/{n_episodios}  "
          f"({100*victorias/n_episodios:.0f}%)")
    print(f"  Reward medio:            {np.mean(rewards_totales):8.2f}")
    print(f"  Reward mejor:            {np.max(rewards_totales):8.2f}")
    print(f"  Reward peor:             {np.min(rewards_totales):8.2f}")
    print(f"  Media monedas/episodio:  {np.mean(monedas_totales):8.2f}")
    print("─" * 60)

    unity_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia PPO Brújula")
    parser.add_argument(
        "--modelo",
        type=str,
        default="results/PPO_brujula_v3/Shooter.onnx",
        help="Ruta al .onnx exportado por mlagents-learn",
    )
    parser.add_argument("--episodios", type=int, default=20)
    args = parser.parse_args()
    main(args.modelo, args.episodios)
