using Unity.MLAgents;
using UnityEngine;
using UnityEngine.AI;

public class CollectibleItem : MonoBehaviour
{
    /// <summary>
    /// Se dispara cada vez que el jugador recoge una moneda.
    /// Lo usan SurvivalRecorder (FSM/Subsunción) y cualquier sistema externo.
    /// </summary>
    public static event System.Action OnCoinCollected;

    private const float collectReward         = 100f;
    private const float minDistanceFromSpawner = 8f;
    private const float obstacleCheckRadius   = 1.2f;
    private const float MAP_HALF_SIZE         = 40f;   // radio de búsqueda para spawn libre
    private LayerMask   obstacleLayerMask;

    private Transform _playerTransform;
    private PlayerAgent _playerAgent;
    private float _lastDistance;
    private EnemySpawnerManager _enemySpawnerManager;

    private void Awake()
    {
        obstacleLayerMask = LayerMask.GetMask("Default");
    }

    private void Start()
    {
        GameObject player = GameObject.FindWithTag("Player");
        if (player != null)
        {
            _playerTransform = player.transform;
            _playerAgent     = player.GetComponent<PlayerAgent>();
            _lastDistance    = Vector3.Distance(_playerTransform.position, transform.position);
        }

        _enemySpawnerManager = FindFirstObjectByType<EnemySpawnerManager>();

        RespawnInNewLocation();
    }

    private void Update()
    {
        // Giro continuo estilo moneda de Mario
        transform.Rotate(Vector3.up, 180f * Time.deltaTime, Space.World);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            Debug.Log("¡Objeto recogido!");
            _playerAgent?.NotificarMonedaRecogida();
            OnCoinCollected?.Invoke();          // notifica a SurvivalRecorder y similares
            _lastDistance = float.MaxValue;
            RespawnInNewLocation();
        }
    }

    public void RespawnInNewLocation()
    {
        float coinSpawnRadius = -1f;
        if (Academy.IsInitialized)
            coinSpawnRadius = Academy.Instance.EnvironmentParameters
                                     .GetWithDefault("coin_spawn_radius", -1f);

        // ── Fase curriculum: moneda cerca del jugador ─────────────────────────
        if (coinSpawnRadius > 0f && _playerTransform != null)
        {
            for (int i = 0; i < 30; i++)
            {
                Vector2 circle    = Random.insideUnitCircle.normalized
                                    * Random.Range(0.5f, coinSpawnRadius);
                Vector3 candidate = _playerTransform.position
                                    + new Vector3(circle.x, 0f, circle.y);
                NavMeshHit nmHit;
                if (!NavMesh.SamplePosition(candidate, out nmHit, 2f, NavMesh.AllAreas))
                    continue;

                Vector3 spawnPos = nmHit.position + Vector3.up * 0.5f;
                if (Physics.CheckSphere(spawnPos, obstacleCheckRadius, obstacleLayerMask))
                    continue;

                SetPosition(spawnPos);
                return;
            }
            // Si no encuentra punto cerca, cae al spawn libre
        }

        // ── Spawn libre: puntos aleatorios validados con SamplePosition ───────
        System.Collections.Generic.List<Vector3> spawnerPositions =
            _enemySpawnerManager != null
                ? _enemySpawnerManager.GetAllSpawnPointPositions()
                : new System.Collections.Generic.List<Vector3>();

        for (int i = 0; i < 50; i++)
        {
            // Punto aleatorio dentro del mapa
            Vector3 randomPoint = new Vector3(
                Random.Range(-MAP_HALF_SIZE, MAP_HALF_SIZE),
                0f,
                Random.Range(-MAP_HALF_SIZE, MAP_HALF_SIZE)
            );

            NavMeshHit hit;
            if (!NavMesh.SamplePosition(randomPoint, out hit, 5f, NavMesh.AllAreas))
                continue;

            // Distancia mínima a spawners de enemigos
            bool demasiadoCerca = false;
            foreach (Vector3 sp in spawnerPositions)
            {
                if (Vector3.Distance(hit.position, sp) < minDistanceFromSpawner)
                {
                    demasiadoCerca = true;
                    break;
                }
            }
            if (demasiadoCerca) continue;

            Vector3 spawnPos = hit.position + Vector3.up * 0.5f;
            if (Physics.CheckSphere(spawnPos, obstacleCheckRadius, obstacleLayerMask))
                continue;

            SetPosition(spawnPos);
            return;
        }

        // ── Fallback: spawn cerca del jugador sin restricciones ───────────────
        if (_playerTransform != null)
        {
            NavMeshHit hit;
            if (NavMesh.SamplePosition(_playerTransform.position, out hit, 10f, NavMesh.AllAreas))
            {
                SetPosition(hit.position + Vector3.up * 0.5f);
                return;
            }
        }

        Debug.LogWarning("CollectibleItem: no se encontró ningún punto de spawn válido.");
    }

    private void SetPosition(Vector3 pos)
    {
        transform.position = pos;
        if (_playerTransform != null)
            _lastDistance = Vector3.Distance(_playerTransform.position, pos);
    }
}
