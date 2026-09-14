from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


def main():
    print("Esperando a Unity... Dale al PLAY en el editor.")

    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[])

    env = UnityToGymWrapper(
        unity_env, uint8_visual=False, flatten_branched=True, allow_multiple_obs=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000, save_path="./models_backup_v3/", name_prefix="dqn_shooter_v3"
    )

    print("Iniciando DQN v3...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=200000,
        learning_rate=2.5e-4,  # bajado de 5e-4 para más estabilidad
        exploration_fraction=0.5,  # subido de 0.4, más exploración inicial
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,  # subido de 0.05, más exploración residual
        learning_starts=10000,
        train_freq=4,
        batch_size=64,
        tensorboard_log="./dqn_tensorboard_v3/",
    )

    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando modelo actual...")
        model.save("dqn_shooter_v3_interrupted")
        env.close()
        return

    model.save("dqn_shooter_v3_final")
    print("¡Entrenamiento v3 finalizado y modelo guardado!")
    env.close()


if __name__ == "__main__":
    main()
