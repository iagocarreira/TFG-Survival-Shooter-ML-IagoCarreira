from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
import gym
import numpy as np


class TransposeImageWrapper(gym.ObservationWrapper):
    """
    gym_unity devuelve la imagen en formato (H, C, W).
    La convertimos a (H, W, C) para que stable-baselines3
    aplique VecTransposeImage y obtenga (C, H, W) correctamente.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape  # (H, C, W)
        # Reordenamos a (H, W, C) = channels-last
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[2], obs_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return np.transpose(obs, (0, 2, 1))  # (H, C, W) → (H, W, C)


# Entrenamiento DQN Visual: Solo recoger moneda con imagen de cámara fija
# El agente recibe como observación la imagen 84x84 de la CamaraFija
# en lugar de vectores numéricos. Usa CnnPolicy para procesar píxeles.
#
# Fase 1 de aprendizaje visual:
#   - La moneda spawna solo dentro del área visible de la CamaraFija
#   - El agente solo puede moverse dentro de esa zona (paredes LevelExtent)
#   - Sin enemigos
#   - Recompensa: +1 por recoger moneda, -1 por morir
#
# Una vez que aprenda, se vuelve a la configuración normal.

TOTAL_STEPS = 2_000_000


def main():
    print("=" * 50)
    print("Entrenamiento DQN Visual - Camara fija")
    print("=" * 50)
    print("Esperando a Unity... Dale al PLAY en el editor.")
    print("Asegurate de que:")
    print("  - CamaraFija activa con ZonaCamara")
    print("  - CameraSensor en Player_Agent apuntando a CamaraFija")
    print("  - Vector Observation Space Size = 0")
    print("  - Current Mode = DQN_Discrete")
    print("  - Sin enemigos (spawn rate 9999)")

    params_channel = EnvironmentParametersChannel()

    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[params_channel])

    env = UnityToGymWrapper(
        unity_env,
        uint8_visual=True,  # imagen en formato uint8 (0-255)
        flatten_branched=True,
        allow_multiple_obs=False,
    )
    env = TransposeImageWrapper(env)  # (H,W,C) → (C,H,W) para CnnPolicy

    # Sin enemigos
    params_channel.set_float_parameter("normal_enemy_spawn_rate", 9999.0)
    params_channel.set_float_parameter("big_enemy_spawn_rate", 9999.0)

    print(f"\nEspacio de observacion: {env.observation_space}")
    print(f"Espacio de acciones:    {env.action_space}")

    model = DQN(
        "CnnPolicy",  # red neuronal convolucional para procesar imagen
        env,
        verbose=1,
        buffer_size=100_000,
        learning_rate=1e-4,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=10_000,
        train_freq=4,
        batch_size=32,
        optimize_memory_usage=False,
        tensorboard_log="./dqn_tensorboard_visual/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models_visual/",
        name_prefix="dqn_visual",
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./models_visual/best/",
        log_path="./models_visual/logs/",
        eval_freq=50_000,
        n_eval_episodes=5,
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
        print("\nInterrumpido. Guardando modelo...")
        model.save("dqn_visual_interrupted")
        env.close()
        return

    model.save("dqn_visual_final")
    print("\n¡Entrenamiento visual finalizado!")
    print("Mejor modelo en: ./models_visual/best/best_model.zip")
    env.close()


if __name__ == "__main__":
    main()
