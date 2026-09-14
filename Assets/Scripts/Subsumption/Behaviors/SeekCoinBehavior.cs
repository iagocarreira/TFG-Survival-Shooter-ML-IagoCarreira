using UnityEngine;

/// <summary>
/// Layer 1 - Buscar moneda (prioridad media-baja)
/// Si hay una moneda en el mapa, el agente se dirige hacia ella.
/// Suprime el comportamiento de exploración (Layer 0).
/// </summary>
public class SeekCoinBehavior : IBehavior
{
    public int Priority => 1;
    public string Name => "BuscarMoneda";

    private readonly Transform _agentTransform;

    public SeekCoinBehavior(Transform agentTransform)
    {
        _agentTransform = agentTransform;
    }

    public bool IsActive()
    {
        return GetNearestCoin() != null;
    }

    public Vector2 GetMovement()
    {
        Transform coin = GetNearestCoin();
        if (coin == null) return Vector2.zero;

        Vector3 dir = (coin.position - _agentTransform.position);
        dir.y = 0f;
        dir.Normalize();
        return new Vector2(dir.x, dir.z);
    }

    public bool WantsToShoot() => false;

    private Transform GetNearestCoin()
    {
        CollectibleItem[] coins = Object.FindObjectsByType<CollectibleItem>(FindObjectsSortMode.None);
        if (coins.Length == 0) return null;

        Transform nearest = null;
        float minDist = float.MaxValue;
        foreach (CollectibleItem c in coins)
        {
            float d = Vector3.Distance(_agentTransform.position, c.transform.position);
            if (d < minDist) { minDist = d; nearest = c.transform; }
        }
        return nearest;
    }
}
