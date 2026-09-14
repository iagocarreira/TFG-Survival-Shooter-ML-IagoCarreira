using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class PlayerAgent : Agent
{
    public enum TrainingMode { PPO_Continuous, PPO_Aimbot, DQN_Discrete, DQN_Brujula, DQN_Rayos, FSM_o_Jugador, Subsumption }
    public enum RunMode { Inference, TrainDQN }

    [Header("Configuración de Entrenamiento")]
    public TrainingMode currentMode = TrainingMode.PPO_Continuous;
    public RunMode currentRunMode = RunMode.Inference;

    [Header("Configuración de Visión")]
    [SerializeField] private float _shootingRange = 10f; // Igual que el Ray Length del sensor

    [Header("Límite de pasos sin enemigos (PPO_Aimbot / DQN)")]
    [Tooltip("Pasos máximos por episodio cuando no hay enemigos activos. 0 = sin límite siempre.")]
    [SerializeField] private int _maxStepsSinEnemigos = 5000;

    [Header("Recompensas por paso")]
    [Tooltip("Escenario 0 (supervivencia pura): activa +0.01/n_enemigos por paso en lugar de la penalización fija.\n" +
             "Escenario 1 (con monedas): desactivado, se usa -0.01 fijo.")]
    [SerializeField] private bool _supervivenciaPura = false;

    [Header("Referencias")]
    public Action OnEpisodeBeginEvent;
    /// <summary>
    /// Se dispara al comienzo de cada episodio nuevo con los datos del episodio anterior.
    /// Parámetros: rewardTotal, monedasRecogidas, pasosRealizados, victoria
    /// </summary>
    public Action<float, int, int, bool> OnEpisodeTerminado;

    [SerializeField] private EnemySpawnerManager _enemySpawnerManager;
    [SerializeField] private PlayerMovement _playerMovement;
    [SerializeField] private PlayerShooting _playerShooting;
    [SerializeField] private PlayerHealth _playerHealth;
    [SerializeField] private PlayerInputs _playerInputs;
    [SerializeField] private Transform _raySensorTransform;

    private int _floorMask;
    private Rigidbody _rigidbody;

    // Contador de monedas por episodio (condición de victoria)
    private int   _monedasEpisodio   = 0;
    private const int   MonedasParaGanar   = 5;
    private const float RecompensaVictoria = 200f;

    // Estadísticas del episodio actual (para el recorder)
    private float _rewardEpisodio   = 0f;
    private int   _pasosEpisodio    = 0;
    private bool  _victoriaEpisodio = false;

    private new void Awake()
    {
        Physics.queriesHitTriggers = true;
        _floorMask = LayerMask.GetMask("Floor");
        _rigidbody = GetComponent<Rigidbody>();

        if (currentRunMode == RunMode.TrainDQN)
        {
            Time.timeScale = 10f;
            Debug.Log("Modo TrainDQN: TimeScale = 10");
        }
        else
        {
            Time.timeScale = 1f;
            Debug.Log("Modo Inference: TimeScale = 1");
        }
    }

    void FixedUpdate()
    {
        // Subsunción y FSM gestionan su propio movimiento externamente.
        // Para los modos ML el movimiento lo controla OnActionReceived.
        if (currentMode == TrainingMode.Subsumption) return;
        if (currentMode != TrainingMode.DQN_Discrete &&
            currentMode != TrainingMode.DQN_Brujula  &&
            currentMode != TrainingMode.DQN_Rayos    &&
            currentMode != TrainingMode.FSM_o_Jugador) return;

        // Modo FSM_o_Jugador: aimbot para el jugador humano.
        // El movimiento (WASD) lo gestiona PlayerMovement.FixedUpdate.
        // Aquí solo manejamos la rotación automática hacia el enemigo más cercano.
        if (currentMode == TrainingMode.FSM_o_Jugador)
        {
            if (_enemySpawnerManager != null)
            {
                Vector3 closestEnemy = _enemySpawnerManager.GetClosestEnemyPositionFromPlayer();
                if (closestEnemy != Vector3.zero)
                {
                    float dist = Vector3.Distance(transform.localPosition, closestEnemy);
                    if (dist <= _shootingRange)
                    {
                        Vector3 dir = closestEnemy - transform.localPosition;
                        dir.y = 0f;
                        dir.Normalize();
                        _playerMovement.Turning(dir.x, dir.z);
                        return;
                    }
                }
            }
            // Sin enemigo en rango: el jugador apunta con el ratón
            _playerMovement.Turning(Input.mousePosition);
            return;
        }
    }

    // Mantiene el sensor de rayos siempre horizontal
    private void LateUpdate()
    {
        if (_raySensorTransform == null) return;
        Vector3 euler = _raySensorTransform.eulerAngles;
        _raySensorTransform.eulerAngles = new Vector3(0f, euler.y, 0f);
    }

    public override void OnEpisodeBegin()
    {
        if (currentMode == TrainingMode.FSM_o_Jugador || currentMode == TrainingMode.Subsumption)
            return;

        // Emitir estadísticas del episodio que acaba de terminar
        // (solo si hubo al menos un paso, para no emitir datos vacíos del episodio 0)
        if (_pasosEpisodio > 0)
            OnEpisodeTerminado?.Invoke(_rewardEpisodio, _monedasEpisodio, _pasosEpisodio, _victoriaEpisodio);

        // Resetear contadores para el nuevo episodio
        _monedasEpisodio   = 0;
        _rewardEpisodio    = 0f;
        _pasosEpisodio     = 0;
        _victoriaEpisodio  = false;

        // Resetear posición, salud y animador del jugador
        ResetPlayer();

        // Respawnear la moneda según el radio actual del curriculum,
        // para que no quede en una posición lejana del episodio anterior.
        CollectibleItem coin = FindFirstObjectByType<CollectibleItem>();
        coin?.RespawnInNewLocation();

        // Notificar al SurvivalRecorder (si está presente y en mlAgentsMode)
        // para que reinicie el contador del episodio sin recargar la escena.
        var sr = FindFirstObjectByType<SurvivalRecorder>();
        if (sr != null && sr.mlAgentsMode)
            sr.NotificarNuevoEpisodio();

        OnEpisodeBeginEvent?.Invoke();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (currentMode == TrainingMode.PPO_Continuous ||
            currentMode == TrainingMode.PPO_Aimbot      ||
            currentMode == TrainingMode.DQN_Brujula)
        {
            // Brújula: dirección normalizada (X, Z) hacia la moneda más cercana.
            // PPO_Continuous y DQN_Brujula: Vector Observation Space Size = 2
            // (el RayPerceptionSensor3D añade sus propias observaciones aparte)
            Vector3 coinDir = GetDirectionToNearestCoin();
            sensor.AddObservation(coinDir.x);
            sensor.AddObservation(coinDir.z);
        }
        // DQN_Discrete (visual): sin observaciones vectoriales, solo CameraSensor.
        // DQN_Rayos: sin observaciones vectoriales, solo RayPerceptionSensor3D.
        // Vector Observation Space Size = 0 en Behavior Parameters para ambos modos.
    }

    /// <summary>
    /// Devuelve la dirección normalizada desde el jugador hasta la moneda más cercana.
    /// Devuelve Vector3.zero si no hay moneda en escena.
    /// </summary>
    private Vector3 GetDirectionToNearestCoin()
    {
        CollectibleItem coin = FindFirstObjectByType<CollectibleItem>();
        if (coin == null) return Vector3.zero;

        Vector3 dir = coin.transform.position - transform.position;
        dir.y = 0f;
        if (dir.sqrMagnitude > 0.01f)
            dir.Normalize();
        return dir;
    }

    /// <summary>
    /// Llamado por CollectibleItem cuando el jugador recoge una moneda.
    /// Gestiona la recompensa y comprueba la condición de victoria.
    /// La victoria (5 monedas → EndEpisode) solo se activa cuando hay enemigos,
    /// ya que sin presión recoger 5 monedas no supone ningún reto real.
    /// </summary>
    public void NotificarMonedaRecogida()
    {
        AddReward(100f);
        _rewardEpisodio += 100f;
        _monedasEpisodio++;

        // Victoria siempre activa en TODAS las fases:
        // - Sin enemigos: enseña desde el principio que 5 monedas = objetivo (+200 bonus)
        // - Con enemigos: igual, pero además hay presión de supervivencia
        if (_monedasEpisodio >= MonedasParaGanar)
        {
            AddReward(RecompensaVictoria);
            _rewardEpisodio  += RecompensaVictoria;
            _victoriaEpisodio = true;
            Debug.Log($"¡Victoria! {MonedasParaGanar} monedas recogidas → episodio terminado.");
            EndEpisode();
        }
    }

    /// <summary>
    /// Devuelve true cuando el curriculum ha activado el spawn de enemigos
    /// (normal_enemy_spawn_rate menor que el valor de bloqueo 9999).
    /// </summary>
    private bool EnemigosActivos()
    {
        if (!Academy.IsInitialized) return false;
        float rate = Academy.Instance.EnvironmentParameters
                            .GetWithDefault("normal_enemy_spawn_rate", 9999f);
        return rate < 9999f;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (currentMode == TrainingMode.FSM_o_Jugador) return;
        if (currentMode == TrainingMode.Subsumption) return;
        if (_playerHealth != null && _playerHealth.IsDead) return; // no actuar durante animación de muerte

        _pasosEpisodio++;

        if (currentMode == TrainingMode.PPO_Continuous)
        {
            // Penalización por existir (incentiva eficiencia)
            AddReward(-0.01f);
            _rewardEpisodio -= 0.01f;

            ActionSegment<float> continuousActions = actions.ContinuousActions;
            ActionSegment<int> discreteActions = actions.DiscreteActions;

            float movementHorizontal = continuousActions[0];
            float movementVertical = continuousActions[1];
            _playerMovement.Move(movementHorizontal, movementVertical);

            float rotationX = continuousActions[2];
            float rotationZ = continuousActions[3];
            _playerMovement.Turning(rotationX, rotationZ);

            if (discreteActions.Length > 0 && discreteActions[0] == 1)
            {
                _playerShooting.Shoot();
            }
        }
        else if (currentMode == TrainingMode.PPO_Aimbot)
        {
            // Penalización por existir (incentiva eficiencia)
            AddReward(-0.01f);
            _rewardEpisodio -= 0.01f;

            // Limite de pasos cuando no hay enemigos: evita episodios infinitos en Fase 1.
            // Con enemigos activos el episodio solo termina por muerte o victoria.
            if (_maxStepsSinEnemigos > 0 && !EnemigosActivos() && _pasosEpisodio >= _maxStepsSinEnemigos)
            {
                Debug.Log($"[PPO_Aimbot] Timeout sin enemigos tras {_pasosEpisodio} pasos.");
                EndEpisode();
                return;
            }

            // Aimbot: apunta automáticamente al enemigo más cercano si está en rango
            Vector3 closestEnemyAimbot = _enemySpawnerManager.GetClosestEnemyPositionFromPlayer();
            if (closestEnemyAimbot != Vector3.zero)
            {
                float distAimbot = Vector3.Distance(transform.localPosition, closestEnemyAimbot);
                if (distAimbot <= _shootingRange)
                {
                    Vector3 dirAimbot = closestEnemyAimbot - transform.localPosition;
                    dirAimbot.y = 0f;
                    dirAimbot.Normalize();
                    _playerMovement.Turning(dirAimbot.x, dirAimbot.z);
                }
            }

            // Movimiento continuo: el agente decide hacia dónde moverse
            float movH = actions.ContinuousActions[0];
            float movV = actions.ContinuousActions[1];
            _playerMovement.Move(movH, movV);

            // Disparo discreto: 0 = no disparar, 1 = disparar
            if (actions.DiscreteActions.Length > 0 && actions.DiscreteActions[0] == 1)
            {
                _playerShooting.Shoot();
            }
        }
        else if (currentMode == TrainingMode.DQN_Discrete ||
                 currentMode == TrainingMode.DQN_Brujula  ||
                 currentMode == TrainingMode.DQN_Rayos)
        {
            if (_supervivenciaPura)
            {
                // Escenario 0: recompensa por sobrevivir, inversamente proporcional al nº de enemigos
                float r = 0.01f / (float)(_enemySpawnerManager.EnemyCount == 0 ? 1 : _enemySpawnerManager.EnemyCount);
                AddReward(r);
                _rewardEpisodio += r;
            }
            else
            {
                // Escenario 1: penalización fija por paso (incentiva eficiencia)
                AddReward(-0.01f);
                _rewardEpisodio -= 0.01f;
            }

            // Límite de pasos cuando no hay enemigos: evita episodios infinitos en fases de curriculum sin enemigos.
            if (_maxStepsSinEnemigos > 0 && !EnemigosActivos() && _pasosEpisodio >= _maxStepsSinEnemigos)
            {
                Debug.Log($"[DQN] Timeout sin enemigos tras {_pasosEpisodio} pasos.");
                EndEpisode();
                return;
            }

            int action = actions.DiscreteActions[0];
            float movementHorizontal = 0f;
            float movementVertical = 0f;

            // Aimbot: apunta al enemigo más cercano si está en rango
            Vector3 closestEnemy = _enemySpawnerManager.GetClosestEnemyPositionFromPlayer();
            if (closestEnemy != Vector3.zero)
            {
                float distance = Vector3.Distance(transform.localPosition, closestEnemy);
                if (distance <= _shootingRange)
                {
                    Vector3 direction = closestEnemy - transform.localPosition;
                    direction.y = 0f;
                    direction.Normalize();
                    _playerMovement.Turning(direction.x, direction.z);
                }
            }

            switch (action)
            {
                case 0: break;                              // Quieto
                case 1: movementVertical = 1f; break;      // Arriba
                case 2: movementVertical = -1f; break;     // Abajo
                case 3: movementHorizontal = -1f; break;   // Izquierda
                case 4: movementHorizontal = 1f; break;    // Derecha
                case 5: _playerShooting.Shoot(); break;    // Disparar
            }

            _playerMovement.Move(movementHorizontal, movementVertical);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (currentMode == TrainingMode.FSM_o_Jugador) return;

        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;

        if (currentMode == TrainingMode.PPO_Continuous)
        {
            continuousActions[0] = _playerInputs.HorizontalAxis;
            continuousActions[1] = _playerInputs.VerticalAxis;
            HeuristicAim(continuousActions, discreteActions);
        }
        else if (currentMode == TrainingMode.PPO_Aimbot)
        {
            // Movimiento manual; el aimbot gestiona la rotación en OnActionReceived
            continuousActions[0] = _playerInputs.HorizontalAxis;
            continuousActions[1] = _playerInputs.VerticalAxis;
            discreteActions[0]   = _playerInputs.IsShooting ? 1 : 0;
        }
        else if (currentMode == TrainingMode.DQN_Discrete ||
                 currentMode == TrainingMode.DQN_Brujula  ||
                 currentMode == TrainingMode.DQN_Rayos)
        {
            discreteActions[0] = 0;

            if (_playerInputs.VerticalAxis > 0)          discreteActions[0] = 1;
            else if (_playerInputs.VerticalAxis < 0)     discreteActions[0] = 2;
            else if (_playerInputs.HorizontalAxis < 0)   discreteActions[0] = 3;
            else if (_playerInputs.HorizontalAxis > 0)   discreteActions[0] = 4;
            else if (_playerInputs.IsShooting)           discreteActions[0] = 5;
        }
    }

    /// <summary>
    /// Igual que AddReward pero también actualiza el acumulado interno del recorder.
    /// Usar para recompensas aplicadas desde scripts externos (p.ej. PlayerHealth).
    /// </summary>
    public void AddTrackedReward(float reward)
    {
        AddReward(reward);
        _rewardEpisodio += reward;
    }

    public void ResetPlayer()
    {
        transform.localPosition = Vector3.zero;
        _playerHealth.ResetHealth();
    }

    private void HeuristicAim(ActionSegment<float> continuousActions, ActionSegment<int> discreteActions)
    {
        Ray camRay = Camera.main.ScreenPointToRay(Input.mousePosition);
        Vector3 playerToMouse = Vector3.zero;
        RaycastHit floorHit;

        if (Physics.Raycast(camRay, out floorHit, 100f, _floorMask))
        {
            playerToMouse = floorHit.point - transform.localPosition;
            playerToMouse.y = 0f;
            playerToMouse.Normalize();
        }

        continuousActions[2] = playerToMouse.x;
        continuousActions[3] = playerToMouse.z;
        discreteActions[0] = _playerInputs.IsShooting ? 1 : 0;
    }
}
