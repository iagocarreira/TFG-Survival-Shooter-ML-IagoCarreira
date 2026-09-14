using UnityEngine;

public class PlayerShooting : MonoBehaviour
{
    public int damagePerShot = 20;
    public float timeBetweenBullets = 0.15f;
    public float range = 100f;

    [SerializeField] private PlayerInputs _playerInputs;
    [SerializeField] private PlayerAgent _playerAgent;

    [Header("Recompensas ML")]
    [Tooltip("Escenario 0: -0.1  |  Escenario 1: -0.02")]
    [SerializeField] private float _missPenalty = -0.02f;

    float timer;
    float _debugLogTimer;
    Ray shootRay = new Ray();
    RaycastHit shootHit;
    int shootableMask;
    ParticleSystem gunParticles;
    LineRenderer gunLine;
    AudioSource gunAudio;
    Light gunLight;
    float effectsDisplayTime = 0.2f;


    void Awake ()
    {
        shootableMask = LayerMask.GetMask ("Shootable");
        gunParticles = GetComponent<ParticleSystem> ();
        gunLine = GetComponent <LineRenderer> ();
        gunAudio = GetComponent<AudioSource> ();
        gunLight = GetComponent<Light> ();
    }


    void Update()
    {
        timer += Time.deltaTime;
        _debugLogTimer += Time.deltaTime;

        if (_playerInputs.IsShooting)
            Shoot();

        if (timer >= timeBetweenBullets * effectsDisplayTime)
            DisableEffects();
    }


    public void DisableEffects ()
    {
        if (gunLine  != null) gunLine.enabled  = false;
        if (gunLight != null) gunLight.enabled = false;
    }


    // Jugador humano y DQN: disparan en la dirección que mira el cañón.
    public void Shoot()
    {
        FireRay(transform.forward);
    }

    // FSM / Subsunción: disparan directamente hacia el objetivo sin depender de
    // que MoveRotation haya aplicado la rotación en el physics step actual.
    public void ShootToward(Vector3 targetWorldPosition)
    {
        Vector3 dir = (targetWorldPosition - transform.position).normalized;
        FireRay(dir);
    }

    private void FireRay(Vector3 direction)
    {
        if (timer < timeBetweenBullets || Time.timeScale == 0)
            return;

        _debugLogTimer = 0f;
        timer = 0f;

        if (gunAudio     != null) gunAudio.Play();
        if (gunLight     != null) gunLight.enabled = true;
        if (gunParticles != null) { gunParticles.Stop(); gunParticles.Play(); }
        if (gunLine      != null) { gunLine.enabled = true; gunLine.SetPosition(0, transform.position); }

        shootRay.origin    = transform.position;
        shootRay.direction = direction;

        bool didHitEnemy = false;

        if (Physics.Raycast(shootRay, out shootHit, range, shootableMask))
        {
            EnemyHealth enemyHealth = shootHit.collider.GetComponent<EnemyHealth>();
            if (enemyHealth != null)
            {
                enemyHealth.TakeDamage(damagePerShot, shootHit.point);
                didHitEnemy = true;
            }
            if (gunLine != null) gunLine.SetPosition(1, shootHit.point);
        }
        else
        {
            if (gunLine != null) gunLine.SetPosition(1, shootRay.origin + shootRay.direction * range);
        }

        if (!didHitEnemy && _playerAgent != null)
            _playerAgent.AddTrackedReward(_missPenalty);
    }
}
