﻿using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.SceneManagement;
using Unity.MLAgents.Policies;


public class PlayerHealth : MonoBehaviour
{
    public int startingHealth = 100;
    public int currentHealth;
    public Slider healthSlider;
    public Image damageImage;
    public AudioClip deathClip;
    public float flashSpeed = 5f;
    public Color flashColour = new Color(1f, 0f, 0f, 0.1f);

    [SerializeField] private PlayerAgent _playerAgent;

    [Header("Recompensas ML")]
    [Tooltip("Escenario 0: -1.0  |  Escenario 1: 0 (sin penalización por daño)")]
    [SerializeField] private float _damagePenalty = 0f;
    [Tooltip("Escenario 0: -5.0  |  Escenario 1: -300")]
    [SerializeField] private float _deathPenalty  = -300f;

    Animator anim;
    AudioSource playerAudio;
    PlayerMovement playerMovement;
    PlayerShooting playerShooting;
    bool isDead;
    bool damaged;

    public bool IsDead => isDead;


    void Awake ()
    {
        anim = GetComponent <Animator> ();
        playerAudio = GetComponent <AudioSource> ();
        playerMovement = GetComponent <PlayerMovement> ();
        playerShooting = GetComponentInChildren <PlayerShooting> ();

        ResetHealth();
    }


    void Update ()
    {
        if (_playerAgent == null)
        {
            if(damaged)
            {
                damageImage.color = flashColour;
            }
            else
            {
                damageImage.color = Color.Lerp (damageImage.color, Color.clear, flashSpeed * Time.deltaTime);
            }
        }
        damaged = false;
    }

    public void ResetHealth()
    {
        // Re-obtener referencias por si se llama antes del Awake de esta instancia
        // (ocurre cuando SurvivalRecorder recarga la escena y ML-Agents dispara
        // OnEpisodeBegin antes de que Awake haya corrido en el nuevo objeto)
        if (anim          == null) anim          = GetComponent<Animator>();
        if (playerMovement == null) playerMovement = GetComponent<PlayerMovement>();
        if (playerShooting == null) playerShooting = GetComponentInChildren<PlayerShooting>();

        currentHealth = startingHealth;
        damaged       = false;
        isDead        = false;

        if (playerMovement != null) playerMovement.enabled = true;
        if (playerShooting != null) playerShooting.enabled = true;

        // Sacar al animador del estado "Die" y volver al estado base
        if (anim != null) { anim.Rebind(); anim.Update(0f); }

        if (healthSlider != null)
            healthSlider.value = currentHealth;
    }

    public void TakeDamage (int amount)
    {
        damaged = true;
        currentHealth -= amount;

        if (healthSlider != null)
            healthSlider.value = currentHealth;

        playerAudio.Play();

        if (_playerAgent != null && _damagePenalty != 0f)
            _playerAgent.AddTrackedReward(_damagePenalty);

        if(currentHealth <= 0 && !isDead)
        {
            Death ();
        }
    }


    void Death()
    {
        isDead = true;

        playerAudio.clip = deathClip;
        playerAudio.Play ();

        playerMovement.enabled = false;
        playerShooting.enabled = false;

        if (_playerAgent != null)
        {
            _playerAgent.AddTrackedReward(_deathPenalty);
            playerShooting.DisableEffects();
            anim.SetTrigger("Die");
            StartCoroutine(ReiniciarEpisodioTrasAnimacion());
        }
        else
        {
            // Modo jugador humano: animación + recarga de escena
            playerShooting.DisableEffects ();
            anim.SetTrigger ("Die");
            StartCoroutine(GameOverTrasAnimacion());
        }
    }

    private IEnumerator ReiniciarEpisodioTrasAnimacion()
    {
        // Pequeña pausa para que la animación de muerte se aprecie (inferencia)
        // En entrenamiento (TimeScale = 10) esto dura <0.3 s reales
        yield return new WaitForSeconds(2f);
        _playerAgent.EndEpisode();
    }

    private IEnumerator GameOverTrasAnimacion()
    {
        yield return new WaitForSeconds(2.5f);
        // Si hay un SurvivalRecorder activo, él gestiona la recarga de escena.
        if (FindFirstObjectByType<SurvivalRecorder>() == null)
            SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }

    public void RestartLevel()
    {
        if (_playerAgent == null)
        {
            SceneManager.LoadScene (0);
        }
    }

    private void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.tag == "DeathWall")
        {
            Death();
        }
    }
}