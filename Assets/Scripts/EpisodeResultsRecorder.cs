using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Graba los resultados de cada episodio en un .txt.
/// El archivo se actualiza tras CADA episodio, así que si paras Unity
/// a mitad tendrás los resultados parciales.
/// Al llegar a 'Episodios Objetivo' escribe el resumen final y para el Play.
///
/// Setup:
///   1. Añade este componente a cualquier GameObject (p. ej. el Player).
///   2. Arrastra el PlayerAgent al campo 'Agent'.
///   3. Pon el nombre del experimento (aparecerá en el archivo y en el nombre del fichero).
///   4. El .txt se guarda en la raíz del proyecto (junto a Assets/).
/// </summary>
public class EpisodeResultsRecorder : MonoBehaviour
{
    [Header("Referencia")]
    [SerializeField] private PlayerAgent agent;

    [Header("Configuración")]
    [SerializeField] private string nombreExperimento  = "PPO_Brujula";
    [SerializeField] private int    episodiosObjetivo  = 30;

    // ─── Datos internos ───────────────────────────────────────────────────────

    private struct DatosEpisodio
    {
        public float reward;
        public int   monedas;
        public int   pasos;
        public float duracionSegundos;
        public bool  victoria;
    }

    private List<DatosEpisodio> _resultados = new List<DatosEpisodio>();
    private string _rutaArchivo;
    private bool   _terminado = false;
    private float  _tiempoInicioEpisodio = 0f;

    // ─── Ciclo de vida ────────────────────────────────────────────────────────

    private void Start()
    {
        if (agent == null)
        {
            Debug.LogError("[EpisodeResultsRecorder] No hay PlayerAgent asignado.");
            enabled = false;
            return;
        }

        // Nombre de archivo con timestamp para no sobreescribir runs anteriores
        string nombre = $"resultados_{nombreExperimento}_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
        string carpeta = Application.dataPath.Replace("/Assets", "");
        _rutaArchivo = Path.Combine(carpeta, nombre);

        agent.OnEpisodeTerminado += OnEpisodeTerminado;
        agent.OnEpisodeBeginEvent += () => _tiempoInicioEpisodio = Time.time;
        _tiempoInicioEpisodio = Time.time;

        Debug.Log($"[EpisodeResultsRecorder] Grabando '{nombreExperimento}' "
                + $"({episodiosObjetivo} episodios)\n→ {_rutaArchivo}");
    }

    private void OnDestroy()
    {
        if (agent != null)
            agent.OnEpisodeTerminado -= OnEpisodeTerminado;
    }

    // ─── Callback del agente ──────────────────────────────────────────────────

    private void OnEpisodeTerminado(float reward, int monedas, int pasos, bool victoria)
    {
        if (_terminado) return;

        float duracion = Time.time - _tiempoInicioEpisodio;
        _resultados.Add(new DatosEpisodio
        {
            reward            = reward,
            monedas           = monedas,
            pasos             = pasos,
            duracionSegundos  = duracion,
            victoria          = victoria
        });

        int ep = _resultados.Count;
        string tag = victoria ? " 🏆" : "";
        Debug.Log($"[Recorder] Ep {ep}/{episodiosObjetivo} | "
                + $"Reward: {reward:F2} | Monedas: {monedas} | Pasos: {pasos}{tag}");

        // Guardar el archivo tras cada episodio
        EscribirArchivo(finalizado: ep >= episodiosObjetivo);

        if (ep >= episodiosObjetivo)
        {
            _terminado = true;
            Debug.Log($"[EpisodeResultsRecorder] ✅ {episodiosObjetivo} episodios completados.\n"
                    + $"Archivo: {_rutaArchivo}");

#if UNITY_EDITOR
            // Parar el Play automáticamente al terminar
            EditorApplication.isPlaying = false;
#endif
        }
    }

    // ─── Escritura del archivo ────────────────────────────────────────────────

    private void EscribirArchivo(bool finalizado)
    {
        int n = _resultados.Count;
        if (n == 0) return;

        float rewardMedio  = 0f;
        float rewardMejor  = float.MinValue;
        float rewardPeor   = float.MaxValue;
        float rewardSumaSq = 0f;
        float monedaMedia  = 0f;
        float pasosMedio   = 0f;
        float durMedia     = 0f;
        float durSumaSq    = 0f;
        int   victorias    = 0;

        foreach (var ep in _resultados)
        {
            rewardMedio += ep.reward;
            if (ep.reward > rewardMejor) rewardMejor = ep.reward;
            if (ep.reward < rewardPeor)  rewardPeor  = ep.reward;
            monedaMedia += ep.monedas;
            pasosMedio  += ep.pasos;
            durMedia    += ep.duracionSegundos;
            if (ep.victoria) victorias++;
        }
        rewardMedio /= n;
        monedaMedia /= n;
        pasosMedio  /= n;
        durMedia    /= n;

        foreach (var ep in _resultados)
        {
            rewardSumaSq += (ep.reward - rewardMedio) * (ep.reward - rewardMedio);
            durSumaSq    += (ep.duracionSegundos - durMedia) * (ep.duracionSegundos - durMedia);
        }
        float rewardStd = Mathf.Sqrt(rewardSumaSq / n);
        float durStd    = Mathf.Sqrt(durSumaSq / n);

        var sb = new StringBuilder();
        string sep  = new string('─', 60);
        string sep2 = new string('=', 60);

        sb.AppendLine(sep2);
        sb.AppendLine($"  RESULTADOS — {nombreExperimento}");
        sb.AppendLine(sep2);
        sb.AppendLine($"  Fecha:       {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"  Experimento: {nombreExperimento}");
        sb.AppendLine($"  Episodios:   {n}/{episodiosObjetivo}  "
                    + (finalizado ? "(COMPLETADO)" : "(EN PROGRESO)"));
        sb.AppendLine();
        sb.AppendLine($"{"Ep",4}  {"Pasos",6}  {"Dur(s)",7}  {"Reward",9}  {"Monedas",8}  {"Victoria",8}");
        sb.AppendLine(sep);

        for (int i = 0; i < _resultados.Count; i++)
        {
            var ep = _resultados[i];
            string v = ep.victoria ? "SI" : "NO";
            sb.AppendLine($"{i + 1,4}  {ep.pasos,6}  {ep.duracionSegundos,7:F1}  {ep.reward,9:F2}  {ep.monedas,8}  {v,8}");
        }

        sb.AppendLine(sep);
        sb.AppendLine();
        sb.AppendLine($"  Victorias:                {victorias}/{n}  ({100f * victorias / n:F0}%)");
        sb.AppendLine($"  Reward medio:             {rewardMedio,8:F2}");
        sb.AppendLine($"  Reward mejor:             {rewardMejor,8:F2}");
        sb.AppendLine($"  Reward peor:              {rewardPeor,8:F2}");
        sb.AppendLine($"  Desv. típica reward:      {rewardStd,8:F2}");
        sb.AppendLine($"  Duración media (s):       {durMedia,8:F2}");
        sb.AppendLine($"  Desv. típica dur. (s):    {durStd,8:F2}");
        sb.AppendLine($"  Media monedas/episodio:   {monedaMedia,8:F2}");
        sb.AppendLine($"  Media pasos/episodio:     {pasosMedio,8:F2}");
        sb.AppendLine(sep2);

        try
        {
            File.WriteAllText(_rutaArchivo, sb.ToString(), Encoding.UTF8);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[EpisodeResultsRecorder] Error al guardar: {ex.Message}");
        }
    }
}
