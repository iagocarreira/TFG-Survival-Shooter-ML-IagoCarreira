using UnityEngine;

/// <summary>
/// Layer 2 - Huir de enemigos (prioridad alta)
/// Si hay un enemigo demasiado cerca, el agente huye en dirección opuesta.
/// Suprime la búsqueda de moneda (Layer 1) y la exploración (Layer 0).
/// </summary>
public class FleeEnemyBehavior : IBehavior
{
    public int Priority => 2;
    public string Name => "HuirEnemigo";

    private readonly Transform _agentTransform;
    private readonly float _fleeDistance;

    public FleeEnemyBehavior(Transform agentTransform, float fleeDistance = 8f)
    {
        _agentTransform = agentTransform;
        _fleeDistance = fleeDistance;
    }

    public bool IsActive()
    {
        return GetNearestEnemyDistance() < _fleeDistance;
    }

    public Vector2 GetMovement()
    {
        Vector3 nearestEnemyPos = GetNearestEnemyPosition();
        if (nearestEnemyPos == Vector3.zero) return Vector2.zero;

        // Dirección opuesta al enemigo más cercano
        Vector3 fleeDir = (_agentTransform.position - nearestEnemyPos);
        fleeDir.y = 0f;
        fleeDir.Normalize();
        return new Vector2(fleeDir.x, fleeDir.z);
    }

    public bool WantsToShoot() => false;

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

    private Vector3 GetNearestEnemyPosition()
    {
        EnemyHealth[] enemies = Object.FindObjectsByType<EnemyHealth>(FindObjectsSortMode.None);
        if (enemies.Length == 0) return Vector3.zero;

        Vector3 nearest = Vector3.zero;
        float minDist = float.MaxValue;
        foreach (EnemyHealth e in enemies)
        {
            float d = Vector3.Distance(_agentTransform.position, e.transform.position);
            if (d < minDist) { minDist = d; nearest = e.transform.position; }
        }
        return nearest;
    }
}
