"""
Entrenamiento DQN — Escenario 0: supervivencia pura (sin monedas).

El agente aprende a sobrevivir el mayor tiempo posible eliminando enemigos.
No hay monedas ni condicion de victoria; el episodio termina por muerte.

Recompensas (configuradas en Unity / PlayerAgent):
  +0.01 / numero_enemigos  por frame (incentivo de supervivencia)
  -1.0                     al recibir dano
  -5.0                     al morir (EndEpisode)

Configuracion Unity antes de lanzar:
  - Current Mode              = DQN_Rayos
  - Current Run Mode          = TrainDQN
  - RayPerceptionSensor3D     ACTIVADO
  - Vector Observation Space  = 0  (sin brujula, sin monedas)
  - Behavior Type             = Default
  - Escena sin monedas (escenario 0)

Uso:
    python train_dqn_supervivencia.py
"""

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


TOTAL_STEPS = 2_000_000


def main():
    print("=" * 50)
    print("  DQN Escenario 0 — Supervivencia pura")
    print("=" * 50)
    print("Esperando a Unity... Dale al PLAY en el editor.")

    params_channel = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(file_name=None, seed=1,
                                 side_channels=[params_channel])
    env = UnityToGymWrapper(
        unity_env, uint8_visual=False, flatten_branched=True,
        allow_multiple_obs=False,
    )

    # Sin monedas, enemigos a dificultad maxima
    params_channel.set_float_parameter("coin_spawn_radius",       0.0)
    params_channel.set_float_parameter("normal_enemy_spawn_rate", 5.0)
    params_channel.set_float_parameter("big_enemy_spawn_rate",   12.0)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./dqn_supervivencia/",
        name_prefix="dqn_supervivencia",
    )

    print("Iniciando DQN supervivencia...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=200_000,
        learning_rate=2.5e-4,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        learning_starts=10_000,
        train_freq=4,
        batch_size=64,
        tensorboard_log="./dqn_supervivencia_tensorboard/",
    )

    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando modelo...")
        model.save("./dqn_supervivencia/dqn_supervivencia_interrupted")
        env.close()
        return

    model.save("./dqn_supervivencia/dqn_supervivencia_final")
    print("\n¡Entrenamiento finalizado!")
    print("Modelo: ./dqn_supervivencia/dqn_supervivencia_final.zip")
    env.close()


if __name__ == "__main__":
    main()
