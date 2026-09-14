import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3 import DQN


def main():
    print("Esperando a Unity... Dale al PLAY en el editor para ver jugar a la IA.")

    # Conectamos con Unity a velocidad normal
    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[])

    # El entorno debe configurarse EXACTAMENTE igual que en el entrenamiento
    env = UnityToGymWrapper(
        unity_env, uint8_visual=False, flatten_branched=True, allow_multiple_obs=False
    )

    print("Cargando el cerebro entrenado (best_model curriculum v3)...")
    model = DQN.load("./models_curriculum_v3/best_curriculum_v3_phase3/best_model")

    print("¡Modelo cargado con éxito! Empieza la partida.")

    obs = env.reset()
    pasos_sobrevividos = 0
    puntuacion_acumulada = 0.0

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            pasos_sobrevividos += 1
            puntuacion_acumulada += reward

            if done:
                print("\n" + "=" * 40)
                print(" ¡EL AGENTE HA MUERTO! Fin de la partida.")
                print(f"  Pasos sobrevividos: {pasos_sobrevividos}")
                print(f" Recompensa total: {puntuacion_acumulada:.2f}")
                print("=" * 40 + "\n")
                break

    except KeyboardInterrupt:
        print("\nPartida detenida manualmente por el usuario.")
        print("\n" + "=" * 40)
        print(" ¡EL AGENTE HA MUERTO! Fin de la partida.")
        print(f"  Pasos sobrevividos: {pasos_sobrevividos}")
        print(f" Recompensa total: {puntuacion_acumulada:.2f}")
        print("=" * 40 + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
