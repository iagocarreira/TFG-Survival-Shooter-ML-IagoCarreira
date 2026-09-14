from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Entrenamiento DQN_13: Sin enemigos - Observaciones normalizadas
# Cambios respecto a DQN_12:
#   - Observaciones normalizadas (dirección relativa + distancia norm + velocidad norm)
#   - buffer_size aumentado a 200_000 para reducir catastrophic forgetting
#   - exploration_fraction=0.6 para que epsilon decaiga más lento
#   - 2M steps para dar más tiempo de aprendizaje estable
# Recompensas:
#   +1 por recoger la moneda
#   -1 por morir (no aplica sin enemigos)

TOTAL_STEPS = 2_000_000


def main():
    print("=" * 50)
    print("Entrenamiento DQN_13: Sin enemigos - Obs normalizadas v2")
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
        buffer_size=200_000,         # aumentado para reducir catastrophic forgetting
        learning_rate=2.5e-4,
        exploration_fraction=0.6,    # epsilon decae más lento → más exploración sostenida
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
        name_prefix="dqn_sin_enemigos_v2",
    )

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=checkpoint_callback,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando modelo actual...")
        model.save("dqn_sin_enemigos_v2_interrupted")
        env.close()
        return

    model.save("dqn_sin_enemigos_v2_final")
    print("\n¡Entrenamiento DQN_13 finalizado y modelo guardado!")
    env.close()


if __name__ == "__main__":
    main()
