"""
ver_observaciones.py
--------------------
Conecta al editor de Unity en Play Mode y muestra en detalle
todas las observaciones que recibe la red neuronal:
  - Observacion 0: RayPerceptionSensor3D (rayos)
  - Observacion 1: Vector (brujula / compass)

USO:
  1. Abre Unity y dale a Play (Behavior Type = Default)
  2. Ejecuta este script: python ver_observaciones.py
"""

import numpy as np
from mlagents_envs.environment import UnityEnvironment

# Tags configurados en el RayPerceptionSensor3D (mismo orden que en Unity)
TAGS = ["Enemy", "Terrain", "Collectible"]  # + 1 slot para "nada"
NUM_TAGS = len(TAGS) + 1                    # 4 valores por rayo (one-hot)
NUM_RAYS = 15
VALUES_PER_RAY = NUM_TAGS + 1              # 4 tags + 1 distancia = 5


def decode_rays(ray_obs):
    """Decodifica el vector plano de rayos en info legible."""
    print("\n=== OBSERVACION 0: RayPerceptionSensor3D ===")
    print(f"Tamano vector: {len(ray_obs)}  ({NUM_RAYS} rayos x {VALUES_PER_RAY} valores)")
    print("-" * 55)

    for i in range(NUM_RAYS):
        base = i * VALUES_PER_RAY
        one_hot = ray_obs[base: base + NUM_TAGS]      # 4 valores one-hot
        distancia = ray_obs[base + NUM_TAGS]           # distancia normalizada

        hit_idx = int(np.argmax(one_hot))
        hit_val = one_hot[hit_idx]

        if hit_val > 0.5:
            if hit_idx < len(TAGS):
                tag_name = TAGS[hit_idx]
            else:
                tag_name = "sin_tag"
            print(f"  Ray[{i:02d}]: HIT   tag={tag_name:<12}  dist={distancia:.3f}  one_hot={np.round(one_hot,1)}")
        else:
            print(f"  Ray[{i:02d}]: MISS                    dist={distancia:.3f}  one_hot={np.round(one_hot,1)}")


def decode_vector(vec_obs):
    """Decodifica el vector de observaciones de la brujula."""
    print("\n=== OBSERVACION 1: Vector (brujula) ===")
    print(f"Tamano vector: {len(vec_obs)}")
    print("-" * 55)
    if len(vec_obs) >= 8:
        print(f"  Moneda mas cercana  x={vec_obs[0]:.2f}  y={vec_obs[1]:.2f}  z={vec_obs[2]:.2f}")
        print(f"  Posicion agente     x={vec_obs[3]:.2f}  y={vec_obs[4]:.2f}  z={vec_obs[5]:.2f}")
        print(f"  Velocidad           vx={vec_obs[6]:.2f}  vz={vec_obs[7]:.2f}")
    else:
        print(f"  Valores: {np.round(vec_obs, 3)}")


def main():
    print("Conectando al editor de Unity... (asegurate de estar en Play Mode)")
    env = UnityEnvironment(file_name=None, seed=42, side_channels=[], timeout_wait=120)
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    print(f"\nBehavior: {behavior_name}")
    print(f"Numero de observaciones: {len(spec.observation_specs)}")
    for i, obs_spec in enumerate(spec.observation_specs):
        print(f"  [{i}] nombre='{obs_spec.name}'  shape={obs_spec.shape}")

    input("\nPresiona ENTER para capturar un frame de observaciones...")

    decision_steps, _ = env.get_steps(behavior_name)

    if len(decision_steps) == 0:
        print("No hay agentes en decision_steps. Intenta mover el agente en Unity.")
        env.close()
        return

    obs_list = decision_steps.obs
    print(f"\nAgentes activos: {len(decision_steps)}")

    # Obs 0: rayos
    if len(obs_list) > 0:
        decode_rays(obs_list[0][0])  # primer agente

    # Obs 1: vector (brujula)
    if len(obs_list) > 1:
        decode_vector(obs_list[1][0])

    print("\n[Listo] Cierra Unity o ejecuta de nuevo para otro frame.")
    env.close()


if __name__ == "__main__":
    main()
