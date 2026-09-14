using UnityEngine;

[CreateAssetMenu(menuName = "FSM/Player/Estados/Busqueda", fileName = "BusquedaState")]
public class BusquedaState : FSMState
{
    [Header("Configuración de Búsqueda")]
    public float wanderRadius = 20f;
    public float wanderTimer = 5f;
    public bool useNavMesh = true;
    
    private Vector3 wanderTarget;
    private float timer;

    public override void EnterState(FSMAgent controller)
    {
        Debug.Log(" Estado: Buscando enemigos de forma fluida...");
        timer = 0f;
        SetNewWanderTarget(controller);
    }

    public override void UpdateState(FSMAgent controller)
    {
        timer += Time.deltaTime;

        // 1. Calculamos la dirección pero IGNORAMOS la altura (Y = 0)
        Vector3 directionToTarget = wanderTarget - controller.transform.position;
        directionToTarget.y = 0f;
        float distanceToTarget = directionToTarget.magnitude;
        
        // 2. ¿Hemos llegado al destino O se ha acabado el tiempo límite?
        if (distanceToTarget < 1f || timer >= wanderTimer)
        {
            // En lugar de quedarnos quietos, elegimos un nuevo destino inmediatamente
            SetNewWanderTarget(controller);
            timer = 0f;
            return; // Salimos de este frame, empezará a caminar en el siguiente
        }
        
        // 3. Nos movemos y rotamos fluidamente hacia el objetivo
        Vector3 moveDirection = directionToTarget.normalized;
        controller.PlayerMovement.Move(moveDirection.x, moveDirection.z);
        controller.PlayerMovement.Animating(moveDirection.x, moveDirection.z);
        controller.PlayerMovement.Turning(moveDirection.x, moveDirection.z);
    }
    
    public override void CheckTransitions(FSMAgent controller) { }

    private void SetNewWanderTarget(FSMAgent controller)
    {
        if (useNavMesh)
        {
            wanderTarget = controller.GetRandomNavMeshPoint(wanderRadius);
        }
        else
        {
            Vector2 randomCircle = Random.insideUnitCircle * wanderRadius;
            wanderTarget = controller.transform.position + new Vector3(randomCircle.x, 0, randomCircle.y);
        }
        
        // Aseguramos que la altura visual del Gizmo coincida con el jugador
        wanderTarget.y = controller.transform.position.y;
        
        // La línea amarilla te ayudará a ver hacia dónde se dirige en la pestaña "Scene"
        Debug.DrawLine(controller.transform.position, wanderTarget, Color.yellow, 2f);
    }
}