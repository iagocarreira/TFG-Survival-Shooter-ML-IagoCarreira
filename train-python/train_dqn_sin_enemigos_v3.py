from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

# Entrenamiento DQN_14: Sin enemigos - Anti catastrophic forgetting
# Cambios respecto a DQN_13:
#   - buffer_size=500_000        → buffer más grande, conserva más experiencias buenas
#   - learning_rate=1e-4         → aprende más despacio, olvida menos
#   - exploration_fraction=0.7   → epsilon baja aún más lento
#   - target_update_interval=50_000 → red objetivo se actualiza menos frecuentemente
#   - EvalCallback               → guarda automáticamente el mejor modelo encontrado
# Recompensas:
#   +1 por recoger la moneda
#   -1 por morir (no aplica sin enemigos)

TOTAL_STEPS = 2_000_000


def main():
    print("=" * 50)
    print("Entrenamiento DQN_14: Sin enemigos - Anti catastrophic forgetting")
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
        buffer_size=500_000,             # buffer grande → conserva experiencias buenas más tiempo
        learning_rate=1e-4,              # LR más bajo → aprende despacio pero olvida menos
        exploration_fraction=0.7,        # epsilon baja muy despacio → más exploración sostenida
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=10_000,
        train_freq=4,
        batch_size=64,
        target_update_interval=50_000,   # red objetivo se actualiza poco → más estabilidad
        tensorboard_log="./dqn_tensorboard_sin_enemigos/",
    )

    # Guarda checkpoint cada 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models_sin_enemigos/",
        name_prefix="dqn_sin_enemigos_v3",
    )

    # Evalúa cada 50k steps y guarda automáticamente el MEJOR modelo encontrado
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./models_sin_enemigos/best_v3/",
        log_path="./models_sin_enemigos/logs_v3/",
        eval_freq=50_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=callbacks,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando modelo actual...")
        model.save("dqn_sin_enemigos_v3_interrupted")
        env.close()
        return

    model.save("dqn_sin_enemigos_v3_final")
    print("\n¡Entrenamiento DQN_14 finalizado!")
    print("Mejor modelo guardado en: ./models_sin_enemigos/best_v3/best_model.zip")
    env.close()


if __name__ == "__main__":
    main()
