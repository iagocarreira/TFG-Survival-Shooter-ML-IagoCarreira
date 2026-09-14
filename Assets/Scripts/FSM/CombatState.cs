using UnityEngine;

[CreateAssetMenu(menuName = "FSM/Player/Estados/Combate", fileName = "CombatState")]
public class CombatState : FSMState
{
    [Header("Configuración de Combate")]
    public float safeDistance = 5f;        // Distancia de seguridad a mantener

    public override void EnterState(FSMAgent controller)
    {
        Debug.Log(" Estado: Combatiendo - Manteniendo distancia!");
    }

    public override void UpdateState(FSMAgent controller)
    {
        Vector3? enemyPos = controller.GetClosestEnemyPosition();

        if (enemyPos.HasValue)
        {
            Vector3 enemyPosition = enemyPos.Value;
            Vector3 directionToEnemy = (enemyPosition - controller.transform.position).normalized;
            float distanceToEnemy = Vector3.Distance(controller.transform.position, enemyPosition);

            // 1. Apuntar al enemigo (para visual / rotación del personaje)
            if (controller.PlayerMovement != null)
                controller.PlayerMovement.Turning(directionToEnemy.x, directionToEnemy.z);

            // 2. Disparar directamente hacia el objetivo (sin depender de transform.forward,
            //    que MoveRotation no actualiza hasta el siguiente physics step)
            if (controller.PlayerShooting != null)
                controller.PlayerShooting.ShootToward(enemyPosition);
            else
                Debug.LogWarning("[CombatState] PlayerShooting es null — GunBarrelEnd debe ser hijo de Player_Agent.");

            // 3. Movimiento dinámico (kiting)
            float h, v;
            if (distanceToEnemy < safeDistance)
            {
                // Retroceder: buscar punto válido en NavMesh para no atravesar límites del mapa
                Vector3 retreatPos = controller.GetFleePoint(enemyPosition, safeDistance);
                Vector3 retreatDir = retreatPos - controller.transform.position;
                if (retreatDir.magnitude > 0.5f)
                {
                    retreatDir = retreatDir.normalized;
                    h = retreatDir.x;
                    v = retreatDir.z;
                }
                else
                {
                    // Sin espacio de retirada: quedarse quieto en lugar de empujar el límite
                    h = 0f;
                    v = 0f;
                }
            }
            else if (distanceToEnemy > safeDistance * 1.5f)
            {
                h = directionToEnemy.x;
                v = directionToEnemy.z;
            }
            else
            {
                h = 0f;
                v = 0f;
            }
            controller.PlayerMovement.Move(h, v);
            controller.PlayerMovement.Animating(h, v);
        }
    }

    public override void CheckTransitions(FSMAgent controller) { }
}