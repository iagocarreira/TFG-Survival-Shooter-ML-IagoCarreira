from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Entrenamiento 2: Curriculum con enemigos
# Parte del modelo entrenado en "Entrenamiento 1" (sin enemigos) y
# añade presión progresiva de combate en tres fases.
#
# Recompensas:
#   +2000 por recoger el objeto
#   -0.01 por frame (incentiva rapidez)
#   -9999 por morir
#
# Fases:
#   Fase 1 — Sin enemigos (consolidación):  el agente refuerza lo aprendido
#   Fase 2 — Pocos enemigos:               introduce presión de combate suave
#   Fase 3 — Dificultad normal:            condiciones reales del juego

MODELO_PREENTRENADO = "dqn_sin_enemigos_final"  # modelo del Entrenamiento 1

PHASES = [
    {
        "name": "Fase 1 - Consolidacion sin enemigos",
        "steps": 200_000,
        "normal_spawn_rate": 9999.0,
        "big_spawn_rate": 9999.0,
        "checkpoint_prefix": "curriculum_v2_phase1",
    },
    {
        "name": "Fase 2 - Pocos enemigos",
        "steps": 400_000,
        "normal_spawn_rate": 8.0,
        "big_spawn_rate": 15.0,
        "checkpoint_prefix": "curriculum_v2_phase2",
    },
    {
        "name": "Fase 3 - Dificultad normal",
        "steps": 400_000,
        "normal_spawn_rate": 3.0,
        "big_spawn_rate": 8.0,
        "checkpoint_prefix": "curriculum_v2_phase3",
    },
]


def main():
    print("=" * 50)
    print("Entrenamiento 2: Curriculum con enemigos")
    print(f"Cargando modelo base: {MODELO_PREENTRENADO}")
    print("=" * 50)
    print("Esperando a Unity... Dale al PLAY en el editor.")

    params_channel = EnvironmentParametersChannel()

    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[params_channel])

    env = UnityToGymWrapper(
        unity_env, uint8_visual=False, flatten_branched=True, allow_multiple_obs=False
    )

    # Cargar el modelo preentrenado del Entrenamiento 1
    model = DQN.load(
        MODELO_PREENTRENADO,
        env=env,
        tensorboard_log="./dqn_tensorboard_curriculum_v2/",
    )

    # Ajustar hiperparámetros para fine-tuning:
    # epsilon bajo porque ya sabe buscar el objeto, ahora refina con enemigos
    model.exploration_fraction = 0.3
    model.exploration_initial_eps = 0.3
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
                save_path="./models_curriculum_v2/",
                name_prefix=phase["checkpoint_prefix"],
            )

            model.learn(
                total_timesteps=phase["steps"],
                callback=checkpoint_callback,
                reset_num_timesteps=(i == 0),
            )

            model.save(f"dqn_curriculum_v2_{phase['checkpoint_prefix']}_done")
            print(f"{phase['name']} completada. Modelo guardado.")

    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando modelo actual...")
        model.save("dqn_curriculum_v2_interrupted")
        env.close()
        return

    model.save("dqn_shooter_escenario2_final")
    print("\n¡Entrenamiento curriculum v2 finalizado y modelo guardado!")
    env.close()


if __name__ == "__main__":
    main()
