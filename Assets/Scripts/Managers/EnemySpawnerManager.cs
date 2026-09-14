using System;
using System.Collections.Generic;
using UnityEngine;

public class EnemySpawnerManager : MonoBehaviour
{
    public int EnemyCount => _enemies.Count;
    public Transform PlayerTransform => _playerTransform;
    [SerializeField] private Transform _playerTransform;
    public PlayerHealth PlayerHealth => _playerHealth;
    [SerializeField] private PlayerHealth _playerHealth;
    [SerializeField] private PlayerAgent _playerAgent;

    [Header("Recompensas de Combate")]
    [Tooltip("Activar en Escenario 1 (matar y sobrevivir). Desactivar en Escenario 2 (recoger objeto).")]
    [SerializeField] private bool _enableCombatRewards = false;
    [SerializeField] private float _hitReward = 0.3f;
    [SerializeField] private float _killReward = 1f;

    private List<GameObject> _enemies = new List<GameObject>();

    public void ResetEnemies()
    {
        foreach (GameObject enemy in _enemies)
        {
            Destroy(enemy);
        }
        _enemies.Clear();
    }

    public void AddNewEnemy(GameObject enemy)
    {
        _enemies.Add(enemy.gameObject);
        enemy.GetComponent<EnemyHealth>().OnDeath += OnEnemyDeathHandler;
        enemy.GetComponent<EnemyHealth>().OnHit += OnEnemyHitHandler;
    }

    public List<Vector3> GetAllSpawnPointPositions()
    {
        List<Vector3> positions = new List<Vector3>();
        EnemySpawner[] spawners = GetComponentsInChildren<EnemySpawner>();
        foreach (EnemySpawner spawner in spawners)
        {
            if (spawner.SpawnPoints == null) continue;
            foreach (Transform sp in spawner.SpawnPoints)
            {
                if (sp != null) positions.Add(sp.position);
            }
        }
        return positions;
    }

    public Vector3 GetClosestEnemyPositionFromPlayer()
    {
        if (_enemies.Count == 0)
            return Vector3.zero;

        SortEnemiesByDistance();
        return _enemies[0].transform.localPosition;
    }

    private void OnEnemyHitHandler()
    {
        if (_enableCombatRewards && _playerAgent != null)
            _playerAgent.AddReward(_hitReward);
    }

    private void OnEnemyDeathHandler(EnemyHealth enemy)
    {
        if (_enableCombatRewards && _playerAgent != null)
            _playerAgent.AddReward(_killReward);

        enemy.OnDeath -= OnEnemyDeathHandler;
        enemy.OnHit -= OnEnemyHitHandler;
        _enemies.Remove(enemy.gameObject);
    }

    private void SortEnemiesByDistance()
    {
        _enemies.Sort(delegate(GameObject a, GameObject b)
        {
            return Vector3.Distance(_playerTransform.transform.localPosition, a.transform.localPosition)
            .CompareTo(
                Vector3.Distance(_playerTransform.transform.localPosition, b.transform.localPosition));
        });
    }
}
