using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// Arquitectura de Subsunción — Escenario 2 (monedas + enemigos).
///
/// La detección replica exactamente el RayPerceptionSensor3D:
///   - 15 rayos distribuidos en 180°
///   - Rango máximo igual al del sensor (10 unidades)
///   - Detecta los tags "Enemy" y "Collectible"
///
/// Jerarquía de capas (de menor a mayor prioridad):
///   Layer 0 - Explorar       → ningún rayo detecta nada relevante
///   Layer 1 - Matar enemigo  → algún rayo detecta un enemigo
///   Layer 2 - Recoger moneda → algún rayo detecta la moneda (suprime Layer 1)
///   Layer 3 - Huir           → varios rayos detectan enemigos muy cerca (prioridad máxima)
/// </summary>
[RequireComponent(typeof(NavMeshAgent))]
public class SubsumptionBrain : MonoBehaviour
{
    [Header("Referencias")]
    [SerializeField] private PlayerShooting _playerShooting;
    [SerializeField] private PlayerHealth   _playerHealth;
    [SerializeField] private PlayerMovement _playerMovement;

    [Header("Configuración de rayos (debe coincidir con RayPerceptionSensor3D)")]
    [SerializeField] private int   _numRayos    = 15;
    [SerializeField] private float _anguloTotal  = 180f;
    [SerializeField] private float _rangoRayo    = 10f;
    [SerializeField] private float _alturaRayo   = 0.5f;   // altura desde la que se lanzan

    [Header("Layer 3 — Huir")]
    [Tooltip("Número mínimo de rayos que detectan enemigo para activar la huida")]
    [SerializeField] private int   _rayosParaHuir    = 4;
    [Tooltip("Distancia máxima del enemigo para que cuente en la huida")]
    [SerializeField] private float _distanciaHuida   = 5f;
    [SerializeField] private float _distanciaFuga    = 12f;

    [Header("Layer 1 — Matar enemigo")]
    [Tooltip("Distancia óptima que mantiene del enemigo mientras dispara")]
    [SerializeField] private float _distanciaOptima  = 6f;

    [Header("Layer 0 — Explorar")]
    [SerializeField] private float _wanderInterval   = 3f;
    [SerializeField] private float _wanderRadius     = 20f;

    [Header("Experimento")]
    [Tooltip("Si está activo, Layer 2 navega directamente hacia la moneda aunque no la vea. " +
             "Si está desactivado, Layer 2 solo se activa cuando un rayo detecta la moneda.")]
    [SerializeField] private bool  _usarPosicionDirectaMoneda = true;

    [Header("Debug")]
    [SerializeField] private bool  _showDebugLog     = true;
    [SerializeField] private bool  _drawRays         = true;

    // Componentes
    private NavMeshAgent _navAgent;
    private Rigidbody    _rigidbody;
    private Animator     _animator;

    // Estado
    private float   _wanderTimer        = 0f;
    private string  _activeBehaviorName = "";
    private bool    _isDead             = false;
    private float   _stuckTimer         = 0f;
    private Vector3 _lastPosition       = Vector3.zero;
    private const float STUCK_INTERVAL  = 2f;
    private const float STUCK_THRESHOLD = 0.2f;

    // Referencia a la moneda activa (posición directa, para comparación justa con PPO)
    private Transform _monedaActiva = null;

    // Resultado de cada rayo
    private struct ResultadoRayo
    {
        public bool      golpeaEnemigo;
        public bool      golpeaMoneda;
        public float     distancia;
        public Transform objetivo;
        public Vector3   punto;
    }

    // ── Inicialización ────────────────────────────────────────────────────────

    private void Start()
    {
        _navAgent  = GetComponent<NavMeshAgent>();
        _rigidbody = GetComponent<Rigidbody>();
        _animator  = GetComponent<Animator>();

        if (_rigidbody != null)
        {
            _rigidbody.isKinematic = true;
            _rigidbody.useGravity  = false;
        }

        PlayerMovement pm          = GetComponent<PlayerMovement>();
        _navAgent.speed            = (pm != null) ? pm.speed : 6f;
        _navAgent.angularSpeed     = 360f;
        _navAgent.acceleration     = 100f;
        _navAgent.stoppingDistance = 0.5f;

        if (_playerMovement != null)
            _playerMovement.enabled = false;

        StartCoroutine(InicializarNavMesh());
    }

    private System.Collections.IEnumerator InicializarNavMesh()
    {
        yield return null;
        if (_navAgent.isOnNavMesh) { Debug.Log("[Subsumption] En NavMesh."); yield break; }

        NavMeshHit hit;
        if (NavMesh.SamplePosition(transform.position, out hit, 10f, NavMesh.AllAreas))
        {
            _navAgent.enabled = false;
            yield return null;
            transform.position = hit.position;
            yield return null;
            _navAgent.enabled          = true;
            _navAgent.speed            = GetComponent<PlayerMovement>()?.speed ?? 6f;
            _navAgent.angularSpeed     = 360f;
            _navAgent.acceleration     = 100f;
            _navAgent.stoppingDistance = 0.5f;
            yield return null;
        }
        Debug.Log("[Subsumption] NavMesh listo. isOnNavMesh=" + _navAgent.isOnNavMesh);
    }

    // ── Bucle principal ───────────────────────────────────────────────────────

    private void Update()
    {
        if (_playerHealth != null && _playerHealth.currentHealth <= 0)
        {
            if (!_isDead) OnDeath();
            return;
        }
        if (!_navAgent.isActiveAndEnabled || !_navAgent.isOnNavMesh) return;

        // Actualizar referencia a la moneda activa cada frame
        ActualizarMoneda();

        // Lanzar todos los rayos una sola vez por frame
        ResultadoRayo[] rayos = LanzarRayos();

        // Pre-calcular enemigo más cercano (se reutiliza en L1 y L2)
        ResultadoRayo enemigoMasCercano = default;
        float distEnemigo = float.MaxValue;
        foreach (var r in rayos)
        {
            if (r.golpeaEnemigo && r.distancia < distEnemigo)
            {
                distEnemigo       = r.distancia;
                enemigoMasCercano = r;
            }
        }

        // ── LAYER 3 — Huir ────────────────────────────────────────────────────
        // Condición: varios rayos detectan enemigos muy cercanos
        int rayosEnemigoCerca = 0;
        Vector3 direccionAmenaza = Vector3.zero;
        foreach (var r in rayos)
        {
            if (r.golpeaEnemigo && r.distancia <= _distanciaHuida)
            {
                rayosEnemigoCerca++;
                direccionAmenaza += (transform.position - r.punto);
            }
        }

        if (rayosEnemigoCerca >= _rayosParaHuir)
        {
            LogBehavior("Huir [L3]");
            Vector3 puntoFuga = GetPuntoFuga(direccionAmenaza.normalized);
            SetDestinacionSegura(puntoFuga);
            UpdateAnimation();
            return;
        }

        // ── LAYER 2 — Recoger moneda ──────────────────────────────────────────
        // Modo normal: navega directamente a la moneda si existe en la escena.
        // Modo experimento: solo se activa cuando un rayo la detecta.
        ResultadoRayo monedaDetectada = default;
        if (!_usarPosicionDirectaMoneda)
        {
            foreach (var r in rayos)
            {
                if (r.golpeaMoneda) { monedaDetectada = r; break; }
            }
        }

        bool layer2Activa = _usarPosicionDirectaMoneda
            ? _monedaActiva != null
            : monedaDetectada.golpeaMoneda;

        if (layer2Activa)
        {
            LogBehavior("Recoger moneda [L2]");
            Vector3 destMoneda = _usarPosicionDirectaMoneda
                ? _monedaActiva.position
                : monedaDetectada.punto;
            SetDestinacionSegura(destMoneda);

            // Disparo oportunista: si hay un enemigo visible, apuntamos y disparamos
            // sin abandonar la ruta hacia la moneda
            if (enemigoMasCercano.golpeaEnemigo)
            {
                AimAt(enemigoMasCercano.objetivo.position);
                if (_playerShooting != null && _playerShooting.enabled)
                    _playerShooting.Shoot();
            }

            UpdateAnimation();
            return;
        }

        // ── LAYER 1 — Matar enemigo ───────────────────────────────────────────
        // Condición: algún rayo detecta un enemigo (solo alcanzable si no hay moneda)
        if (enemigoMasCercano.golpeaEnemigo)
        {
            LogBehavior("Matar enemigo [L1]");
            AimAt(enemigoMasCercano.objetivo.position);

            if (_playerShooting != null && _playerShooting.enabled)
                _playerShooting.Shoot();

            if (distEnemigo < _distanciaOptima)
            {
                Vector3 retroceso = transform.position +
                    (transform.position - enemigoMasCercano.objetivo.position).normalized * _distanciaOptima;
                SetDestinacionSegura(retroceso);
            }
            else
            {
                Vector3 lateral = Vector3.Cross(Vector3.up,
                    (enemigoMasCercano.objetivo.position - transform.position).normalized);
                SetDestinacionSegura(transform.position + lateral * 3f);
            }

            UpdateAnimation();
            return;
        }

        // ── LAYER 0 — Explorar ────────────────────────────────────────────────
        // Modo normal: prácticamente nunca se alcanza (Layer 2 siempre activa).
        // Modo experimento (_usarPosicionDirectaMoneda=false): comportamiento
        // principal cuando la moneda no está en el campo de visión.
        LogBehavior("Explorar [L0]");
        AntiAtasco();
        _wanderTimer -= Time.deltaTime;
        if (_wanderTimer <= 0f || !_navAgent.hasPath || _navAgent.remainingDistance < 0.5f)
        {
            SetDestinacionSegura(GetPuntoNavMeshAleatorio());
            _wanderTimer = _wanderInterval;
        }
        UpdateAnimation();
    }

    // ── Búsqueda de la moneda activa ──────────────────────────────────────────

    private void ActualizarMoneda()
    {
        // Si la referencia es válida y el objeto sigue activo, reutilizarla
        if (_monedaActiva != null && _monedaActiva.gameObject.activeInHierarchy)
            return;

        // Buscar la moneda por tag (se ejecuta solo cuando es necesario)
        GameObject coin = GameObject.FindWithTag("Collectible");
        _monedaActiva = (coin != null && coin.activeInHierarchy) ? coin.transform : null;
    }

    // ── Sistema de rayos (réplica del RayPerceptionSensor3D) ─────────────────

    private ResultadoRayo[] LanzarRayos()
    {
        ResultadoRayo[] resultados = new ResultadoRayo[_numRayos];
        Vector3 origen = transform.position + Vector3.up * _alturaRayo;

        for (int i = 0; i < _numRayos; i++)
        {
            // Distribuir los rayos uniformemente en el arco de _anguloTotal
            float t     = (_numRayos > 1) ? (float)i / (_numRayos - 1) : 0.5f;
            float angulo = -_anguloTotal * 0.5f + _anguloTotal * t;
            Vector3 dir  = Quaternion.Euler(0f, angulo, 0f) * transform.forward;

            RaycastHit hit;
            if (Physics.Raycast(origen, dir, out hit, _rangoRayo))
            {
                resultados[i].golpeaEnemigo = hit.collider.CompareTag("Enemy");
                resultados[i].golpeaMoneda  = hit.collider.CompareTag("Collectible");
                resultados[i].distancia     = hit.distance;
                resultados[i].objetivo      = hit.transform;
                resultados[i].punto         = hit.point;

                if (_drawRays)
                    Debug.DrawRay(origen, dir * hit.distance,
                        resultados[i].golpeaEnemigo ? Color.red :
                        resultados[i].golpeaMoneda  ? Color.yellow : Color.white);
            }
            else
            {
                resultados[i].distancia = _rangoRayo;
                if (_drawRays)
                    Debug.DrawRay(origen, dir * _rangoRayo, Color.green);
            }
        }
        return resultados;
    }

    // ── Navegación ────────────────────────────────────────────────────────────

    private Vector3 GetPuntoFuga(Vector3 direccionHuida)
    {
        float[] angulos = { 0f, 45f, -45f, 90f, -90f, 135f, -135f, 180f };
        foreach (float angulo in angulos)
        {
            Vector3 dir       = Quaternion.Euler(0f, angulo, 0f) * direccionHuida;
            Vector3 candidato = transform.position + dir * _distanciaFuga;
            NavMeshHit hit;
            if (!NavMesh.SamplePosition(candidato, out hit, _distanciaFuga * 0.5f, NavMesh.AllAreas)) continue;
            NavMeshPath path = new NavMeshPath();
            if (_navAgent.CalculatePath(hit.position, path) && path.status == NavMeshPathStatus.PathComplete)
                return hit.position;
        }
        return GetPuntoNavMeshAleatorio();
    }

    private Vector3 GetPuntoNavMeshAleatorio()
    {
        for (int i = 0; i < 10; i++)
        {
            Vector3 rand = transform.position + new Vector3(
                Random.Range(-_wanderRadius, _wanderRadius), 0f,
                Random.Range(-_wanderRadius, _wanderRadius));
            NavMeshHit hit;
            if (NavMesh.SamplePosition(rand, out hit, _wanderRadius, NavMesh.AllAreas))
                return hit.position;
        }
        return transform.position;
    }

    private void SetDestinacionSegura(Vector3 destino)
    {
        if (_navAgent.isActiveAndEnabled && _navAgent.isOnNavMesh)
            _navAgent.SetDestination(destino);
    }

    // ── Utilidades ────────────────────────────────────────────────────────────

    private void AimAt(Vector3 objetivo)
    {
        Vector3 dir = objetivo - transform.position;
        dir.y = 0f;
        if (dir.sqrMagnitude > 0.01f)
        {
            _navAgent.updateRotation = false;
            transform.rotation = Quaternion.Slerp(transform.rotation,
                Quaternion.LookRotation(dir.normalized), Time.deltaTime * 15f);
        }
        else _navAgent.updateRotation = true;
    }

    private void AntiAtasco()
    {
        _stuckTimer += Time.deltaTime;
        if (_stuckTimer < STUCK_INTERVAL) return;
        if (Vector3.Distance(transform.position, _lastPosition) < STUCK_THRESHOLD)
        {
            SetDestinacionSegura(GetPuntoNavMeshAleatorio());
            Debug.Log("[Subsumption] Atascado, nuevo destino asignado.");
        }
        _lastPosition = transform.position;
        _stuckTimer   = 0f;
    }

    private void OnDeath()
    {
        _isDead = true;
        if (_navAgent.isActiveAndEnabled && _navAgent.isOnNavMesh)
        {
            _navAgent.ResetPath();
            _navAgent.velocity = Vector3.zero;
        }
        if (_animator != null) _animator.SetBool("IsWalking", false);
        Debug.Log("[Subsumption] Agente muerto.");
    }

    private void UpdateAnimation()
    {
        if (_animator == null) return;
        _animator.SetBool("IsWalking", _navAgent.velocity.magnitude > 0.1f);
    }

    private void LogBehavior(string name)
    {
        if (_showDebugLog && name != _activeBehaviorName)
        {
            _activeBehaviorName = name;
            Debug.Log("[Subsumption] Capa activa: " + name);
        }
    }

    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(transform.position, _distanciaHuida);
        Gizmos.color = Color.blue;
        Gizmos.DrawWireSphere(transform.position, _rangoRayo);
        Gizmos.color = Color.green;
        Gizmos.DrawWireSphere(transform.position, _distanciaOptima);
    }
}
