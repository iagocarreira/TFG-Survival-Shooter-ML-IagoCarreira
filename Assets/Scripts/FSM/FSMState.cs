using UnityEngine;

// Versión simple sin namespaces
public abstract class FSMState : ScriptableObject
{
    public abstract void EnterState(FSMAgent controller);
    public abstract void UpdateState(FSMAgent controller);
    public abstract void CheckTransitions(FSMAgent controller);
}