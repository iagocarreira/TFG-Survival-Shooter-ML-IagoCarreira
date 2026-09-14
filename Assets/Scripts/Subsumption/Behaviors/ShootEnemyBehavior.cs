using UnityEngine;

/// <summary>
/// Layer 3 - Disparar (prioridad más alta)
/// Si hay un enemigo en rango de disparo, el agente dispara.
/// Corre en PARALELO con el movimiento — no suprime el desplazamiento,
/// solo añade la acción de disparar encima del comportamiento activo.
/// </summary>
public class ShootEnemyBehavior : IBehavior
{
    public int Priority => 3;
    public string Name => "Disparar";

    private readonly Transform _agentTransform;
    private readonly float _shootRange;

    public ShootEnemyBehavior(Transform agentTransform, float shootRange = 10f)
    {
        _agentTransform = agentTransform;
        _shootRange = shootRange;
    }

    public bool IsActive()
    {
        return GetNearestEnemyDistance() <= _shootRange;
    }

    // No controla el movimiento — solo dispara
    public Vector2 GetMovement() => Vector2.zero;

    public bool WantsToShoot() => IsActive();

    private float GetNearestEnemyDistance()
    {
        EnemyHealth[] enemies = Object.FindObjectsByType<EnemyHealth>(FindObjectsSortMode.None);
        float minDist = float.MaxValue;
        foreach (EnemyHealth e in enemies)
        {
            float d = Vector3.Distance(_agentTransform.position, e.transform.position);
            if (d < minDist) minDist = d;
        }
        return minDist;
    }
}
