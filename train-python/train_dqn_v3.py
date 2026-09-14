from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Curriculum de entrenamiento:
#   Fase 1 — Sin enemigos:   el agente aprende a buscar y recoger el objeto sin presión
#   Fase 2 — Pocos enemigos: el agente aprende a navegar con amenaza moderada
#   Fase 3 — Normal:         dificultad completa, igual que el juego real

PHASES = [
    {
        "name": "Fase 1 - Sin enemigos",
        "steps": 300_000,
        "normal_spawn_rate": 9999.0,  # spawn tan lento que no aparecen enemigos
        "big_spawn_rate": 9999.0,
        "checkpoint_prefix": "curriculum_phase1",
    },
    {
        "name": "Fase 2 - Pocos enemigos",
        "steps": 300_000,
        "normal_spawn_rate": 8.0,  # un enemigo cada 8 segundos
        "big_spawn_rate": 15.0,
        "checkpoint_prefix": "curriculum_phase2",
    },
    {
        "name": "Fase 3 - Dificultad normal",
        "steps": 400_000,
        "normal_spawn_rate": 3.0,  # ritmo normal del juego
        "big_spawn_rate": 8.0,
        "checkpoint_prefix": "curriculum_phase3",
    },
]


def main():
    print("Esperando a Unity... Dale al PLAY en el editor.")

    params_channel = EnvironmentParametersChannel()

    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[params_channel])

    env = UnityToGymWrapper(
        unity_env, uint8_visual=False, flatten_branched=True, allow_multiple_obs=False
    )

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=200_000,
        learning_rate=2.5e-4,
        exploration_fraction=0.4,  # por fase: epsilon decae el 40% de los steps de cada fase
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        learning_starts=10_000,
        train_freq=4,
        batch_size=64,
        tensorboard_log="./dqn_tensorboard_curriculum/",
    )

    try:
        for i, phase in enumerate(PHASES):
            print(f"\n{'=' * 50}")
            print(f"Iniciando {phase['name']} ({phase['steps']:,} steps)...")
            print(f"  normal_enemy_spawn_rate = {phase['normal_spawn_rate']}s")
            print(f"  big_enemy_spawn_rate    = {phase['big_spawn_rate']}s")
            print(f"{'=' * 50}")

            # Aplicar parámetros de spawn para esta fase
            params_channel.set_float_parameter(
                "normal_enemy_spawn_rate", phase["normal_spawn_rate"]
            )
            params_channel.set_float_parameter(
                "big_enemy_spawn_rate", phase["big_spawn_rate"]
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=50_000,
                save_path="./models_curriculum/",
                name_prefix=phase["checkpoint_prefix"],
            )

            model.learn(
                total_timesteps=phase["steps"],
                callback=checkpoint_callback,
                reset_num_timesteps=(
                    i == 0
                ),  # solo resetea el contador en la primera fase
            )

            model.save(f"dqn_curriculum_{phase['checkpoint_prefix']}_done")
            print(f"{phase['name']} completada. Modelo guardado.")

    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando modelo actual...")
        model.save("dqn_curriculum_interrupted")
        env.close()
        return

    model.save("dqn_shooter_final_v5_new")
    print("\n¡Entrenamiento curriculum finalizado y modelo guardado!")
    env.close()


if __name__ == "__main__":
    main()
