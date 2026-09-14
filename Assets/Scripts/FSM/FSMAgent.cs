using UnityEngine;
using UnityEngine.AI;

public class FSMAgent : MonoBehaviour
{
    private PlayerMovement _playerMovement;
    private PlayerShooting _playerShooting;
    private PlayerHealth _playerHealth;
    private EnemySpawnerManager _enemyManager;
    private NavMeshAgent _navMeshAgent;
    
    [Header("Referencias")]
    [SerializeField] private FSMBrain _brain;

    [Header("NavMesh")]
    public bool useNavMesh = true;
    public float navMeshSampleDistance = 10f;

    private void Awake()
    {
        _playerMovement = GetComponent<PlayerMovement>();
        _playerShooting = GetComponentInChildren<PlayerShooting>();
        _playerHealth = GetComponent<PlayerHealth>();
        _enemyManager = FindFirstObjectByType<EnemySpawnerManager>();
        _navMeshAgent = GetComponent<NavMeshAgent>();
        
        if (_brain == null)
            _brain = GetComponent<FSMBrain>();

        if (_navMeshAgent != null)
        {
            _navMeshAgent.enabled = false;
        }
    }

    // Propiedades
    public PlayerMovement PlayerMovement => _playerMovement;
    public PlayerShooting PlayerShooting => _playerShooting;
    public PlayerHealth PlayerHealth => _playerHealth;
    public EnemySpawnerManager EnemyManager => _enemyManager;
    public NavMeshAgent NavMeshAgent => _navMeshAgent;

    // Métodos de NavMesh y Huida
    public Vector3 GetRandomNavMeshPoint(float radius)
    {
        Vector3 randomDirection = Random.insideUnitSphere * radius;
        randomDirection += transform.position;
        
        if (NavMesh.SamplePosition(randomDirection, out NavMeshHit hit, navMeshSampleDistance, NavMesh.AllAreas))
        {
            return hit.position;
        }
        return transform.position;
    }
    
    public Vector3 GetNavMeshPointTowardsEnemy(Vector3 enemyPosition, float distance)
    {
        Vector3 direction = (enemyPosition - transform.position).normalized;
        Vector3 targetPoint = transform.position + direction * distance;
        
        if (NavMesh.SamplePosition(targetPoint, out NavMeshHit hit, navMeshSampleDistance, NavMesh.AllAreas))
        {
            return hit.position;
        }
        return transform.position;
    }
    
    public Vector3 GetFleePoint(Vector3 enemyPosition, float fleeDistance)
    {
        Vector3 baseDir = (transform.position - enemyPosition).normalized;

        // Probar la dirección base y luego rotaciones de 45° hasta encontrar un punto en NavMesh
        for (int i = 0; i < 8; i++)
        {
            float angle = i * 45f;
            Vector3 dir = Quaternion.Euler(0, angle, 0) * baseDir;
            Vector3 candidate = transform.position + dir * fleeDistance;

            if (NavMesh.SamplePosition(candidate, out NavMeshHit hit, navMeshSampleDistance, NavMesh.AllAreas))
                return hit.position;
        }

        // Último recurso: quedarse quieto (mejor que salir del mapa)
        return transform.position;
    }
    
    // Método de ayuda para visión
    public Vector3? GetClosestEnemyPosition()
    {
        if (_enemyManager != null && _enemyManager.EnemyCount > 0)
            return _enemyManager.GetClosestEnemyPositionFromPlayer();

        // Fallback: búsqueda directa por físicas si el manager no reporta enemigos
        int shootable = LayerMask.GetMask("Shootable");
        Collider[] hits = Physics.OverlapSphere(transform.position, 30f, shootable);
        foreach (var hit in hits)
        {
            EnemyHealth eh = hit.GetComponent<EnemyHealth>();
            if (eh != null && eh.currentHealth > 0)
                return hit.transform.position;
        }
        return null;
    }
}