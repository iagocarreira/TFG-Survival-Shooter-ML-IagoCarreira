using UnityEngine;

/// <summary>
/// Layer 0 - Explorar (prioridad más baja)
/// El agente se mueve aleatoriamente cuando no tiene nada mejor que hacer.
/// Siempre está activo — es el comportamiento base de la subsunción.
/// </summary>
public class WanderBehavior : IBehavior
{
    public int Priority => 0;
    public string Name => "Explorar";

    private Vector2 _currentDirection;
    private float _changeDirectionTimer;
    private readonly float _changeDirectionInterval = 2.5f;

    public WanderBehavior()
    {
        _currentDirection = Random.insideUnitCircle.normalized;
        _changeDirectionTimer = _changeDirectionInterval;
    }

    public bool IsActive() => true; // siempre activo como comportamiento base

    public Vector2 GetMovement()
    {
        _changeDirectionTimer -= Time.fixedDeltaTime;
        if (_changeDirectionTimer <= 0f)
        {
            _currentDirection = Random.insideUnitCircle.normalized;
            _changeDirectionTimer = _changeDirectionInterval;
        }
        return _currentDirection;
    }

    public bool WantsToShoot() => false;
}
