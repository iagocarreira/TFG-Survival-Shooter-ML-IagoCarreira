using UnityEngine;

/// <summary>
/// Interfaz base para todos los comportamientos de la arquitectura de subsunción.
/// Cada comportamiento tiene una prioridad y puede suprimir a los de menor prioridad.
///
/// Jerarquía (mayor número = mayor prioridad):
///   Layer 0 - Explorar/Deambular    (más baja)
///   Layer 1 - Buscar moneda
///   Layer 2 - Huir de enemigos
///   Layer 3 - Disparar              (más alta, corre en paralelo)
/// </summary>
public interface IBehavior
{
    /// <summary>Prioridad del comportamiento (mayor = más importante)</summary>
    int Priority { get; }

    /// <summary>Nombre descriptivo para debug</summary>
    string Name { get; }

    /// <summary>¿Está activo este comportamiento ahora mismo?</summary>
    bool IsActive();

    /// <summary>Movimiento deseado (x = horizontal, y = vertical). Normalizado [-1, 1]</summary>
    Vector2 GetMovement();

    /// <summary>¿Este comportamiento quiere disparar?</summary>
    bool WantsToShoot();
}
