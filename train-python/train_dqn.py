import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


def main():
    print("Esperando a Unity... Dale al PLAY en el editor.")

    # 1. ACELERAR EL TIEMPO:
    # time_scale=20 acelera Unity 20 veces.
    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[])

    # 2. ENVOLTORIO:
    env = UnityToGymWrapper(
        unity_env, uint8_visual=False, flatten_branched=True, allow_multiple_obs=False
    )

    # 3. CALLBACK PARA GUARDAR:

    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path="./models_backup/", name_prefix="dqn_shooter"
    )

    # 4. INICIAR DQN CON TENSORBOARD:
    print("Iniciando DQN...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=100000,
        learning_rate=1e-3,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log="./dqn_tensorboard_recogible/",
    )

    # 5. ENTRENAR:
    try:
        model.learn(total_timesteps=500000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario. Guardando modelo actual...")
        model.save("dqn_survival_shooter_interrupted_newvision")
        env.close()
        return

    # 6. GUARDAR MODELO FINAL:
    model.save("dqn_survival_shooter_final_newvision")
    print("¡Entrenamiento finalizado y modelo guardado!")

    env.close()


if __name__ == "__main__":
    main()
