using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Graba los resultados de experimentos para agentes FSM y Subsunción,
/// que no usan el ciclo de episodios de ML-Agents.
///
/// Persiste entre recargas de escena (DontDestroyOnLoad + Singleton).
/// Cuando el jugador muere, espera unos segundos y recarga la escena
/// automáticamente para iniciar el siguiente episodio.
///
/// Setup:
///   1. Crea un GameObject vacío llamado "ExperimentManager".
///   2. Añade este componente a ese GameObject.
///   3. Configura nombreExperimento y episodiosObjetivo en el Inspector.
///   4. Asegúrate de que CollectibleItem.OnCoinCollected existe (ya está añadido).
///   5. pon currentMode = FSM_o_Jugador o Subsumption en PlayerAgent.
/// </summary>
public class SurvivalRecorder : MonoBehaviour
{
    [Header("Configuración")]
    [SerializeField] private string nombreExperimento  = "FSM_Escenario1";
    [SerializeField] private int    episodiosObjetivo  = 30;
    [SerializeField] private int    monedasParaVictoria = 5;     // monedas necesarias para ganar
    [SerializeField] private float  esperaTrasDerrota  = 2.5f;  // s antes de recargar

    [Header("Modo ML-Agents (DQN / PPO controlados desde Python)")]
    [Tooltip("Actívalo cuando Python controle al agente (DQN_Rayos, etc.).\n" +
             "SurvivalRecorder solo grabará datos; ML-Agents gestiona el reset del episodio.\n" +
             "PlayerAgent.OnEpisodeBegin() llamará a NotificarNuevoEpisodio() automáticamente.")]
    [SerializeField] public bool mlAgentsMode = false;

    // ─── Singleton ────────────────────────────────────────────────────────────
    private static SurvivalRecorder _instance;

    // ─── Estado del episodio actual ───────────────────────────────────────────
    private float _tiempoInicioEpisodio;
    private int   _monedasEpisodio;
    private bool  _episodioActivo    = false;
    private bool  _esperandoReinicio = false;

    // ─── Historial ────────────────────────────────────────────────────────────
    private struct DatosEpisodio
    {
        public float duracionSegundos;
        public int   monedas;
        public bool  victoria;   // ≥1 moneda recogida
    }

    private List<DatosEpisodio> _resultados = new List<DatosEpisodio>();
    private string _rutaArchivo;
    private bool   _terminado = false;

    // ─── Referencia a PlayerHealth (se reobtiene tras cada carga de escena) ───
    private PlayerHealth _playerHealth;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }
        _instance = this;
        DontDestroyOnLoad(gameObject);

        string nombre  = $"resultados_{nombreExperimento}_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
        string carpeta = Application.dataPath.Replace("/Assets", "");
        _rutaArchivo   = Path.Combine(carpeta, nombre);

        Debug.Log($"[SurvivalRecorder] Iniciado '{nombreExperimento}' "
                + $"({episodiosObjetivo} episodios)\n→ {_rutaArchivo}");
    }

    private void OnEnable()
    {
        SceneManager.sceneLoaded    += OnSceneLoaded;
        CollectibleItem.OnCoinCollected += OnCoinCollected;
    }

    private void OnDisable()
    {
        SceneManager.sceneLoaded    -= OnSceneLoaded;
        CollectibleItem.OnCoinCollected -= OnCoinCollected;
    }

    private void Start()
    {
        if (mlAgentsMode)
        {
            // En modo ML-Agents la escena no se recarga entre episodios.
            // El primer episodio empieza aquí; los siguientes los señaliza
            // PlayerAgent.OnEpisodeBegin() → NotificarNuevoEpisodio().
            _playerHealth         = FindFirstObjectByType<PlayerHealth>();
            _tiempoInicioEpisodio = Time.time;
            _monedasEpisodio      = 0;
            _episodioActivo       = true;
            _esperandoReinicio    = false;
            Debug.Log($"[SurvivalRecorder] Modo ML-Agents activo. Episodio 1/{episodiosObjetivo} iniciado.");
        }
        else
        {
            StartCoroutine(InicializarEpisodio());
        }
    }

    // =========================================================================
    // Eventos de escena y moneda
    // =========================================================================

    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        if (_terminado) return;
        if (mlAgentsMode) return;
        StartCoroutine(InicializarEpisodio());
    }

    private void OnCoinCollected()
    {
        if (!_episodioActivo || _esperandoReinicio) return;
        _monedasEpisodio++;
        Debug.Log($"[SurvivalRecorder] Moneda recogida. Total este episodio: {_monedasEpisodio}");

        // Victoria: se alcanzó el número de monedas requerido → terminar episodio ahora
        if (_monedasEpisodio >= monedasParaVictoria)
        {
            _episodioActivo    = false;
            _esperandoReinicio = true;

            float duracion = Time.time - _tiempoInicioEpisodio;
            RegistrarEpisodio(duracion, _monedasEpisodio, victoria: true);

            Debug.Log($"[SurvivalRecorder] ¡Victoria! {_monedasEpisodio} monedas en {duracion:F1}s");

            if (!_terminado)
                StartCoroutine(ReiniciarTrasDerrota());
        }
    }

    // =========================================================================
    // Lógica de episodio
    // =========================================================================

    private IEnumerator InicializarEpisodio()
    {
        // Guardia: si ya hay un episodio activo (p.ej. Start y OnSceneLoaded disparan a la vez),
        // el segundo en llegar sale inmediatamente sin duplicar el episodio.
        if (_episodioActivo) yield break;
        _episodioActivo = true; // marcar antes del yield para bloquear llamadas concurrentes

        // Esperar un frame para que todos los Awake/Start de la escena se ejecuten.
        yield return null;

        _playerHealth           = FindFirstObjectByType<PlayerHealth>();
        _monedasEpisodio        = 0;
        _tiempoInicioEpisodio   = Time.time;
        _esperandoReinicio      = false;

        int epActual = _resultados.Count + 1;
        Debug.Log($"[SurvivalRecorder] ▶ Episodio {epActual}/{episodiosObjetivo} iniciado.");
    }

    private void Update()
    {
        if (_terminado || !_episodioActivo || _esperandoReinicio) return;
        if (_playerHealth == null) return;

        if (_playerHealth.IsDead)
        {
            _episodioActivo    = false;
            _esperandoReinicio = true;

            float duracion = Time.time - _tiempoInicioEpisodio;
            RegistrarEpisodio(duracion, _monedasEpisodio, victoria: false); // muerte = derrota siempre

            if (!_terminado)
                StartCoroutine(ReiniciarTrasDerrota());
        }
    }

    private IEnumerator ReiniciarTrasDerrota()
    {
        if (mlAgentsMode)
        {
            // ML-Agents (PlayerHealth → EndEpisode → OnEpisodeBegin) gestiona el reset.
            // Solo esperamos a que PlayerAgent llame a NotificarNuevoEpisodio().
            yield break;
        }
        yield return new WaitForSeconds(esperaTrasDerrota);
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }

    /// <summary>
    /// Llamado por PlayerAgent.OnEpisodeBegin() cuando mlAgentsMode = true.
    /// Reinicia el estado del episodio sin recargar la escena.
    /// </summary>
    public void NotificarNuevoEpisodio()
    {
        if (!mlAgentsMode || _terminado) return;

        _monedasEpisodio      = 0;
        _tiempoInicioEpisodio = Time.time;
        _episodioActivo       = true;
        _esperandoReinicio    = false;

        int epSiguiente = _resultados.Count + 1;
        Debug.Log($"[SurvivalRecorder] ▶ Episodio {epSiguiente}/{episodiosObjetivo} iniciado (ML-Agents).");
    }

    // =========================================================================
    // Registro y escritura
    // =========================================================================

    private void RegistrarEpisodio(float duracion, int monedas, bool victoria)
    {
        _resultados.Add(new DatosEpisodio
        {
            duracionSegundos = duracion,
            monedas          = monedas,
            victoria         = victoria,
        });

        int ep  = _resultados.Count;
        string tag = victoria ? " 🏆" : "";
        Debug.Log($"[SurvivalRecorder] Ep {ep}/{episodiosObjetivo} | "
                + $"Duración: {duracion:F1}s | Monedas: {monedas}{tag}");

        EscribirArchivo(finalizado: ep >= episodiosObjetivo);

        if (ep >= episodiosObjetivo)
        {
            _terminado = true;
            Debug.Log($"[SurvivalRecorder] ✅ {episodiosObjetivo} episodios completados.\n"
                    + $"Archivo: {_rutaArchivo}");
#if UNITY_EDITOR
            EditorApplication.isPlaying = false;
#endif
        }
    }

    private void EscribirArchivo(bool finalizado)
    {
        int n = _resultados.Count;
        if (n == 0) return;

        float durMedia    = 0f, durMejor = 0f, durPeor = float.MaxValue, durSumaSq = 0f;
        float monedaMedia = 0f;
        int   victorias   = 0;

        foreach (var ep in _resultados)
        {
            durMedia    += ep.duracionSegundos;
            if (ep.duracionSegundos > durMejor) durMejor = ep.duracionSegundos;
            if (ep.duracionSegundos < durPeor)  durPeor  = ep.duracionSegundos;
            monedaMedia += ep.monedas;
            if (ep.victoria) victorias++;
        }
        durMedia    /= n;
        monedaMedia /= n;

        foreach (var ep in _resultados)
            durSumaSq += (ep.duracionSegundos - durMedia) * (ep.duracionSegundos - durMedia);
        float durStd = Mathf.Sqrt(durSumaSq / n);

        var    sb   = new StringBuilder();
        string sep  = new string('─', 66);
        string sep2 = new string('=', 66);

        sb.AppendLine(sep2);
        sb.AppendLine($"  RESULTADOS — {nombreExperimento}");
        sb.AppendLine(sep2);
        sb.AppendLine($"  Fecha:        {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"  Checkpoint:   N/A (Simbólico)");
        sb.AppendLine($"  Episodios:    {n}/{episodiosObjetivo}  "
                    + (finalizado ? "(COMPLETADO)" : "(EN PROGRESO)"));
        sb.AppendLine();
        sb.AppendLine($"{"Ep",4}  {"Dur(s)",8}  {"Monedas",8}  {"Victoria",8}");
        sb.AppendLine(sep);

        for (int i = 0; i < _resultados.Count; i++)
        {
            var ep = _resultados[i];
            string v = ep.victoria ? "SI" : "NO";
            sb.AppendLine($"{i + 1,4}  {ep.duracionSegundos,8:F2}  {ep.monedas,8}  {v,8}");
        }

        sb.AppendLine(sep);
        sb.AppendLine();
        sb.AppendLine($"  Victorias:              {victorias}/{n}  ({100f * victorias / n:F0}%)");
        sb.AppendLine($"  Dur(s) — media: {durMedia,8:F2}  mejor: {durMejor,8:F2}  peor: {durPeor,8:F2}");
        sb.AppendLine($"  Media monedas/episodio: {monedaMedia,8:F2}");
        sb.AppendLine($"  Media dur/episodio (s): {durMedia,8:F2}");
        sb.AppendLine(sep2);

        try
        {
            File.WriteAllText(_rutaArchivo, sb.ToString(), Encoding.UTF8);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[SurvivalRecorder] Error al guardar: {ex.Message}");
        }
    }
}
