using UnityEngine;

[CreateAssetMenu(menuName = "FSM/Player/Estados/Huida", fileName = "FleeState")]
public class FleeState : FSMState
{
    [Header("Configuración de Huida")]
    public float fleeSpeedMultiplier = 2f; 
    public float fleeDistance = 15f;
    public bool useNavMesh = true;
    
    private float enterTime;
    private Vector3 fleeTarget;

    public override void EnterState(FSMAgent controller)
    {
        Debug.Log($" Estado: ¡Huyendo a toda velocidad! (Salud: {controller.PlayerHealth.currentHealth})");
        enterTime = Time.time;
        SetFleeTarget(controller);
    }

    public override void UpdateState(FSMAgent controller)
    {
        Vector3? enemyPos = controller.GetClosestEnemyPosition();
        
        if (enemyPos.HasValue)
        {
            // Si el enemigo se mueve, actualizar punto de huida cada cierto tiempo
            if (Time.time - enterTime > 3f)
            {
                SetFleeTarget(controller);
                enterTime = Time.time;
            }
            
            Vector3 directionToTarget = fleeTarget - controller.transform.position;
            
            if (directionToTarget.magnitude > 1f)
            {
                Vector3 moveDirection = directionToTarget.normalized;
                
                
                controller.PlayerMovement.Move(moveDirection.x, moveDirection.z);
                controller.PlayerMovement.Animating(moveDirection.x, moveDirection.z);
                controller.PlayerMovement.Turning(moveDirection.x, moveDirection.z);
            }
        }
        else
        {
            // Si no hay enemigos, movimiento aleatorio (sin sprint)
            float angle = Time.time * 0.5f;
            Vector3 randomDirection = new Vector3(Mathf.Sin(angle), 0, Mathf.Cos(angle));
            float rx = randomDirection.x * 0.5f, rz = randomDirection.z * 0.5f;
            controller.PlayerMovement.Move(rx, rz);
            controller.PlayerMovement.Animating(rx, rz);
            controller.PlayerMovement.Turning(randomDirection.x, randomDirection.z);
        }
        
        Debug.DrawLine(controller.transform.position, fleeTarget, Color.blue, 2f);
    }

    private void SetFleeTarget(FSMAgent controller)
    {
        Vector3? enemyPos = controller.GetClosestEnemyPosition();
        
        if (enemyPos.HasValue && useNavMesh)
        {
            fleeTarget = controller.GetFleePoint(enemyPos.Value, fleeDistance);
        }
        else if (enemyPos.HasValue)
        {
            Vector3 fleeDirection = (controller.transform.position - enemyPos.Value).normalized;
            fleeTarget = controller.transform.position + fleeDirection * fleeDistance;
        }
        else
        {
            // Sin enemigo visible: usar GetRandomNavMeshPoint para no salir del mapa
            fleeTarget = controller.GetRandomNavMeshPoint(fleeDistance);
        }
        
        fleeTarget.y = controller.transform.position.y;
    }

    public override void CheckTransitions(FSMAgent controller) { }
}