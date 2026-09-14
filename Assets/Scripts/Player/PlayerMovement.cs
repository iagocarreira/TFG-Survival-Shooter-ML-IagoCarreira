using UnityEngine;

[RequireComponent(typeof(PlayerInputs))]
public class PlayerMovement : MonoBehaviour
{
    public float speed = 6f;  

    [SerializeField] private PlayerInputs _playerInputs;

    Vector3 movement;                   
    Animator anim;
    Rigidbody playerRigidbody;
    int floorMask;
    float camRayLength = 100f;
    private bool _isAgent = false;          

    void Awake()
    {
        floorMask = LayerMask.GetMask("Floor");
        anim = GetComponent<Animator>();
        playerRigidbody = GetComponent<Rigidbody>();

        _isAgent = GetComponent<PlayerAgent>() != null;
    }

    void FixedUpdate()
    {
        PlayerAgent agent = GetComponent<PlayerAgent>();

        // Modos ML-Agents (DQN, PPO): OnActionReceived gestiona el movimiento.
        if (agent != null && agent.currentMode != PlayerAgent.TrainingMode.FSM_o_Jugador
                          && agent.currentMode != PlayerAgent.TrainingMode.Subsumption)
            return;

        if (agent != null && agent.currentMode != PlayerAgent.TrainingMode.FSM_o_Jugador)
        {
            // FSM / Subsunción: el estado llama a Move() y Animating() directamente.
            return;
        }

        // Jugador humano (sin PlayerAgent) O modo FSM_o_Jugador con aimbot
        Move(_playerInputs.HorizontalAxis, _playerInputs.VerticalAxis);
        Animating(_playerInputs.HorizontalAxis, _playerInputs.VerticalAxis);

        // La rotación la gestiona PlayerAgent.FixedUpdate (aimbot) en FSM_o_Jugador;
        // en modo humano puro (sin agente) usamos el ratón.
        if (agent == null)
            Turning(Input.mousePosition);
    }

    public void Move(float h, float v)
    {
        movement.Set(h, 0f, v);
        // Time.fixedDeltaTime es constante (0.02s); evita que el movimiento dependa
        // del framerate cuando Move() se llama desde Update (modo FSM).
        movement = movement.normalized * speed * Time.fixedDeltaTime;
        playerRigidbody.MovePosition(transform.position + movement);
    }

    public void Turning(float x, float z)
    {
        Vector3 lookAt = new Vector3(x, 0, z);
        if (lookAt == Vector3.zero)
            return;
            
        Quaternion newRotation = Quaternion.LookRotation(new Vector3(x, 0, z));
        playerRigidbody.MoveRotation(newRotation);
    }

    public void Turning(Vector3 mousePosition)
    {
        Ray camRay = Camera.main.ScreenPointToRay(mousePosition);
        RaycastHit floorHit;

        if (Physics.Raycast(camRay, out floorHit, camRayLength, floorMask))
        {
            Vector3 playerToMouse = floorHit.point - transform.position;
            playerToMouse.y = 0f;
            Quaternion newRotation = Quaternion.LookRotation(playerToMouse);
            playerRigidbody.MoveRotation(newRotation);
        }
    }

    public void Animating(float h, float v)
    {
        bool walking = h != 0f || v != 0f;
        anim.SetBool("IsWalking", walking);
    }
}