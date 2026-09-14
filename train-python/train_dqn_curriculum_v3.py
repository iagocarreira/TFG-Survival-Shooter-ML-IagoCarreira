from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

# Entrenamiento Curriculum v3: Con enemigos
# Carga el mejor modelo de DQN_15 (best_model.zip) — pico ~20 monedas/episodio
# y añade presión progresiva de combate en tres fases.
#
# Cambios respecto a curriculum_v2:
#   - Carga best_model.zip de DQN_15 en vez del modelo final (que había olvidado todo)
#   - EvalCallback para guardar automáticamente el mejor modelo de cada fase
#   - Fase 1 más larga (400k) para consolidar antes de añadir enemigos
#   - Hiperparámetros heredados de DQN_15 (lr=1e-4, buffer=500k)
#
# Recompensas:
#   +1 por recoger la moneda
#   -1 por morir

MODELO_PREENTRENADO = "./models_sin_enemigos/best_v3/best_model"

PHASES = [
    {
        "name": "Fase 1 - Consolidacion sin enemigos",
        "steps": 400_000,
        "normal_spawn_rate": 9999.0,
        "big_spawn_rate": 9999.0,
        "checkpoint_prefix": "curriculum_v3_phase1",
    },
    {
        "name": "Fase 2 - Pocos enemigos",
        "steps": 500_000,
        "normal_spawn_rate": 8.0,
        "big_spawn_rate": 15.0,
        "checkpoint_prefix": "curriculum_v3_phase2",
    },
    {
        "name": "Fase 3 - Dificultad normal",
        "steps": 500_000,
        "normal_spawn_rate": 3.0,
        "big_spawn_rate": 8.0,
        "checkpoint_prefix": "curriculum_v3_phase3",
    },
]


def main():
    print("=" * 50)
    print("Entrenamiento Curriculum v3: Con enemigos")
    print(f"Cargando mejor modelo de DQN_15: {MODELO_PREENTRENADO}")
    print("=" * 50)
    print("Esperando a Unity... Dale al PLAY en el editor.")

    params_channel = EnvironmentParametersChannel()

    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[params_channel])

    env = UnityToGymWrapper(
        unity_env, uint8_visual=False, flatten_branched=True, allow_multiple_obs=False
    )

    # Cargar el MEJOR modelo de DQN_15 (pico ~20 monedas/episodio)
    model = DQN.load(
        MODELO_PREENTRENADO,
        env=env,
        tensorboard_log="./dqn_tensorboard_curriculum_v3/",
    )

    # Hiperparámetros para fine-tuning con enemigos:
    # LR bajo para no destruir lo aprendido, epsilon bajo porque ya sabe recoger monedas
    model.learning_rate = 1e-4
    model.exploration_fraction = 0.3
    model.exploration_initial_eps = 0.2
    model.exploration_final_eps = 0.05

    try:
        for i, phase in enumerate(PHASES):
            print(f"\n{'=' * 50}")
            print(f"Iniciando {phase['name']} ({phase['steps']:,} steps)...")
            print(f"  normal_enemy_spawn_rate = {phase['normal_spawn_rate']}s")
            print(f"  big_enemy_spawn_rate    = {phase['big_spawn_rate']}s")
            print(f"{'=' * 50}")

            params_channel.set_float_parameter(
                "normal_enemy_spawn_rate", phase["normal_spawn_rate"]
            )
            params_channel.set_float_parameter(
                "big_enemy_spawn_rate", phase["big_spawn_rate"]
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=50_000,
                save_path="./models_curriculum_v3/",
                name_prefix=phase["checkpoint_prefix"],
            )

            eval_callback = EvalCallback(
                env,
                best_model_save_path=f"./models_curriculum_v3/best_{phase['checkpoint_prefix']}/",
                log_path=f"./models_curriculum_v3/logs_{phase['checkpoint_prefix']}/",
                eval_freq=50_000,
                n_eval_episodes=10,
                deterministic=True,
                verbose=1,
            )

            callbacks = CallbackList([checkpoint_callback, eval_callback])

            model.learn(
                total_timesteps=phase["steps"],
                callback=callbacks,
                reset_num_timesteps=(i == 0),
            )

            model.save(f"./models_curriculum_v3/dqn_curriculum_v3_{phase['checkpoint_prefix']}_done")
            print(f"{phase['name']} completada. Modelo guardado.")

    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando modelo actual...")
        model.save("dqn_curriculum_v3_interrupted")
        env.close()
        return

    model.save("dqn_shooter_escenario2_final_v3")
    print("\n¡Curriculum v3 finalizado!")
    print("Mejor modelo de cada fase guardado en ./models_curriculum_v3/best_*/")
    env.close()


if __name__ == "__main__":
    main()
