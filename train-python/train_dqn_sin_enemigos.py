from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Entrenamiento 1: Sin enemigos
# El agente aprende exclusivamente a buscar y recoger el objeto.
# Sin presión de combate, la señal de recompensa es limpia:
#   +2000 por recoger el objeto
#   -0.01 por frame (incentiva rapidez)
#   -9999 por morir (no aplica en este entrenamiento)

TOTAL_STEPS = 1_000_000


def main():
    print("=" * 50)
    print("Entrenamiento 1: Sin enemigos - Solo recoger objeto")
    print("=" * 50)
    print("Esperando a Unity... Dale al PLAY en el editor.")

    params_channel = EnvironmentParametersChannel()

    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[params_channel])

    env = UnityToGymWrapper(
        unity_env, uint8_visual=False, flatten_branched=True, allow_multiple_obs=False
    )

    # Sin enemigos: spawn rate altísimo para que no aparezca ninguno
    params_channel.set_float_parameter("normal_enemy_spawn_rate", 9999.0)
    params_channel.set_float_parameter("big_enemy_spawn_rate", 9999.0)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=200_000,
        learning_rate=2.5e-4,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=10_000,
        train_freq=4,
        batch_size=64,
        tensorboard_log="./dqn_tensorboard_sin_enemigos/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models_sin_enemigos/",
        name_prefix="dqn_sin_enemigos",
    )

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=checkpoint_callback,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando modelo actual...")
        model.save("dqn_sin_enemigos_interrupted")
        env.close()
        return

    model.save("dqn_sin_enemigos_final")
    print("\n¡Entrenamiento sin enemigos finalizado y modelo guardado!")
    env.close()


if __name__ == "__main__":
    main()
