"""
play_dqn_escenario0.py
======================
Inferencia del DQN de supervivencia pura (Escenario 0).
Usa stable-baselines3 DQN (no DuelingDQN custom).
Los resultados los graba Unity a través del SurvivalRecorder.

Configuración Unity antes de lanzar:
  - Current Mode              = DQN_Rayos   (ajusta si hay mismatch de obs)
  - Current Run Mode          = Inference
  - Behavior Type             = Default     (Python controla las acciones)
  - Model                     = None
  - RayPerceptionSensor3D     ACTIVADO
  - Vector Observation Space  = 0
  - SurvivalRecorder:
      mlAgentsMode            = true
      episodiosObjetivo       = 100
      monedasParaVictoria     = 9999        (escenario sin monedas)

Uso:
    python play_dqn_escenario0.py
    python play_dqn_escenario0.py --modelo ../dqn/dqn_survival_shooter_final_newrewards.zip
    python play_dqn_escenario0.py --episodios 50
"""

import argparse

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN


def main(modelo_path: str, n_episodios: int) -> None:
    print("=" * 60)
    print("  DQN Escenario 0 — Supervivencia pura")
    print("=" * 60)
    print(f"  Modelo:    {modelo_path}")
    print(f"  Episodios: {n_episodios}")
    print("\nEsperando a Unity... Dale al PLAY en el editor.\n")

    params_ch = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=0, side_channels=[params_ch])
    env = UnityToGymWrapper(unity_env, uint8_visual=False,
                            flatten_branched=True, allow_multiple_obs=False)

    # Escenario 0: sin monedas, enemigos a dificultad máxima
    params_ch.set_float_parameter("coin_spawn_radius",       0.0)   # sin monedas
    params_ch.set_float_parameter("normal_enemy_spawn_rate", 5.0)
    params_ch.set_float_parameter("big_enemy_spawn_rate",   12.0)

    n_obs = env.observation_space.shape[0]
    print(f"Observaciones recibidas: {n_obs}")
    print(f"Acciones:                {env.action_space}\n")

    model = DQN.load(modelo_path)
    print("Modelo cargado.\n")

    for ep in range(1, n_episodios + 1):
        obs = env.reset()
        terminado = False
        pasos = 0

        try:
            while not terminado:
                accion, _ = model.predict(obs, deterministic=True)
                obs, _, terminado, _ = env.step(accion)
                pasos += 1
        except Exception:
            # Unity se detiene al completar todos los episodios
            terminado = True

        print(f"Ep {ep:3d}/{n_episodios} completado ({pasos} pasos)")

    try:
        env.close()
    except Exception:
        pass

    print("\nDone. Resultados grabados por el SurvivalRecorder de Unity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",    type=str, default="../train-python/dqn_supervivencia/dqn_supervivencia_final.zip")
    # Alternativa: mejor checkpoint por reward → dqn_supervivencia_2000000_steps.zip
    parser.add_argument("--episodios", type=int, default=100)
    args = parser.parse_args()
    main(args.modelo, args.episodios)
