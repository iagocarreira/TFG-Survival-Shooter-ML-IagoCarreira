using UnityEngine;

public class FSMBrain : MonoBehaviour
{
    [Header("Referencias")]
    [SerializeField] private FSMAgent _agent;
    // 🔥 ELIMINADO: Ya no necesitamos FSMPlayerHealth
    
    [Header("Estados")]
    public FSMState searchState;
    public FSMState combatState;
    public FSMState fleeState;
    
    [Header("Configuración de Visión y Huida")]
    public float detectionRange = 10f;
    public int initialEnemiesToFlee = 2; // Empieza huyendo con 2
    public float maxFleeTime = 10f;      // Segundos máximos huyendo
    
    // Variables internas de control
    private FSMState _currentState;
    private int _currentEnemiesRequiredToFlee;
    private float _fleeStartTime;
    private int _shootableMask;
    private PlayerAgent _playerAgent;

    public FSMState CurrentState => _currentState;

    [Header("Debug")]
    public bool showGizmos = true;
    
    private void Start()
    {
        if (_agent == null) _agent = GetComponent<FSMAgent>();

        _playerAgent = GetComponent<PlayerAgent>();
            
        _shootableMask = LayerMask.GetMask("Shootable"); // La capa de tus enemigos
        _currentEnemiesRequiredToFlee = initialEnemiesToFlee;
        
        if (_playerAgent == null || _playerAgent.currentMode == PlayerAgent.TrainingMode.FSM_o_Jugador)
        {
            TransitionToState(searchState);
        }
    }
    
    private void Update()
    {

        if (_playerAgent != null && _playerAgent.currentMode != PlayerAgent.TrainingMode.FSM_o_Jugador)
        {
            return;
        }

        if (_currentState == null) return;

        
        if (_agent.PlayerHealth.currentHealth <= 0)
        {
            Debug.Log(" [FSM Brain] El cuerpo ha muerto - Apagando cerebro FSM.");
            _currentState = null;
            this.enabled = false; // El cerebro se apaga a sí mismo para siempre
            return;
        }
        
        DecideTransition();
        _currentState.UpdateState(_agent);
    }
    
    
    
    private void DecideTransition()
    {
        // Contamos cuántos enemigos vivos hay exactamente a nuestro alrededor
        int enemiesInRange = CountEnemiesInDetectionRange();
        bool hasEnemies = enemiesInRange > 0;
        
        // --- LÓGICA MIENTRAS ESTAMOS HUYENDO ---
        if (_currentState == fleeState)
        {
            // Salimos de la huida si bajamos del número de enemigos que nos dio miedo, 
            // O si ya pasaron los 10 segundos de pánico
            if (enemiesInRange < _currentEnemiesRequiredToFlee || Time.time - _fleeStartTime >= maxFleeTime)
            {
                Debug.Log($" Brain: Fin de huida. Temeridad aumentada: de {_currentEnemiesRequiredToFlee} a {_currentEnemiesRequiredToFlee * 2} enemigos.");
                
                // Multiplicamos x2 la valentía para la próxima vez (2 -> 4 -> 8 -> 16...)
                _currentEnemiesRequiredToFlee *= 2; 
                
                if (hasEnemies) TransitionToState(combatState);
                else TransitionToState(searchState);
            }
            return; // Mientras huimos, ignoramos el resto de decisiones
        }

        // --- LÓGICA PARA ENTRAR EN PÁNICO ---
        if (enemiesInRange >= _currentEnemiesRequiredToFlee)
        {
            Debug.Log($" Brain: ¡PÁNICO! {enemiesInRange} enemigos detectados (Límite: {_currentEnemiesRequiredToFlee}) -> Huyendo!");
            TransitionToState(fleeState);
            return;
        }

        // --- LÓGICA NORMAL (COMBATE O BÚSQUEDA) ---
        if (hasEnemies && _currentState != combatState)
        {
            TransitionToState(combatState);
            return;
        }
        
        if (!hasEnemies && _currentState != searchState)
        {
            TransitionToState(searchState);
            return;
        }
    }
    
    private void TransitionToState(FSMState newState)
    {
        if (newState != null && newState != _currentState)
        {
            _currentState = newState;
            
            // Si el nuevo estado es el de Huida, registramos a qué hora empezó
            if (newState == fleeState) 
            {
                _fleeStartTime = Time.time;
            }
            
            _currentState.EnterState(_agent);
        }
    }

    //  NUEVO: Cuenta los enemigos reales usando físicas en lugar de depender del Spawner
    private int CountEnemiesInDetectionRange()
    {
        int count = 0;
        Collider[] hits = Physics.OverlapSphere(transform.position, detectionRange, _shootableMask);
        
        foreach (Collider hit in hits)
        {
            EnemyHealth enemyHealth = hit.GetComponent<EnemyHealth>();
            // Solo los contamos si están vivos
            if (enemyHealth != null && enemyHealth.currentHealth > 0)
            {
                count++;
            }
        }
        return count;
    }


    private void OnDrawGizmosSelected()
    {
        if (!showGizmos) return;
        
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(transform.position, detectionRange);
    }
}