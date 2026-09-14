"""
analyze_results.py
==================
Lee todos los archivos resultados_*.txt generados por EpisodeResultsRecorder
y SurvivalRecorder, y produce una tabla de comparación lista para copiar en
el TFG.

Uso:
    python analyze_results.py                 # busca en el directorio actual
    python analyze_results.py ruta/a/carpeta  # busca en la carpeta indicada

Salida:
    - Tabla de texto en consola
    - Archivo comparison_table.txt con la tabla formateada
    - Archivo latex_table.tex con el código LaTeX de la tabla
"""

import re
import sys
import os
import glob
from dataclasses import dataclass, field
from typing import Optional
import statistics


# ─── Estructura de resultado ──────────────────────────────────────────────────

@dataclass
class ResultadoExperimento:
    nombre:          str
    tipo:            str          # "RL" o "Simbólico"
    n_episodios:     int
    completado:      bool

    # Métricas comunes
    dur_media:       Optional[float] = None
    dur_std:         Optional[float] = None
    moneda_media:    Optional[float] = None
    victorias:       Optional[int]   = None
    tasa_victoria:   Optional[float] = None   # 0–100 %

    # Solo RL
    reward_medio:    Optional[float] = None
    reward_std:      Optional[float] = None
    pasos_medio:     Optional[float] = None

    # Datos brutos por episodio (para re-calcular si hace falta)
    episodios_raw:   list = field(default_factory=list)


# ─── Parseo de archivos ───────────────────────────────────────────────────────

def _extraer_float(linea: str) -> Optional[float]:
    """Extrae el primer número flotante (incluyendo negativos) de una línea."""
    m = re.search(r'[-+]?\d+(?:[.,]\d+)?', linea)
    if m:
        return float(m.group().replace(',', '.'))
    return None


def _extraer_int(linea: str) -> Optional[int]:
    m = re.search(r'\d+', linea)
    return int(m.group()) if m else None


def parsear_archivo(ruta: str) -> Optional[ResultadoExperimento]:
    try:
        with open(ruta, encoding='utf-8') as f:
            contenido = f.read()
    except Exception as e:
        print(f"[!] No se pudo leer {ruta}: {e}")
        return None

    lineas = contenido.splitlines()

    # Nombre del experimento (línea "  RESULTADOS — <nombre>")
    nombre = os.path.basename(ruta).replace('.txt', '')
    for l in lineas:
        if 'RESULTADOS' in l and '—' in l:
            nombre = l.split('—', 1)[1].strip()
            break

    # Tipo: RL o Simbólico
    tipo = 'RL'
    for l in lineas:
        if 'Tipo:' in l and 'imbólico' in l:
            tipo = 'Simbólico'
            break

    # Número de episodios y estado
    n_ep = 0
    completado = False
    for l in lineas:
        if 'Episodios:' in l:
            m = re.search(r'(\d+)/\d+', l)
            if m:
                n_ep = int(m.group(1))
            completado = 'COMPLETADO' in l
            break

    res = ResultadoExperimento(
        nombre=nombre, tipo=tipo,
        n_episodios=n_ep, completado=completado
    )

    # ── Parseo del resumen (bloque tras la segunda línea "─...─") ────────────
    en_resumen = False
    lineas_sep = [i for i, l in enumerate(lineas) if l.startswith('─' * 10)]

    if len(lineas_sep) >= 2:
        for l in lineas[lineas_sep[1]:]:
            l = l.strip()
            if not l:
                continue
            ll = l.lower()

            if 'victorias' in ll:
                m = re.search(r'(\d+)/(\d+)\s*\((\d+(?:[.,]\d+)?)\s*%\)', l)
                if m:
                    res.victorias     = int(m.group(1))
                    res.tasa_victoria = float(m.group(3).replace(',', '.'))

            elif 'reward medio' in ll:
                res.reward_medio = _extraer_float(l)

            elif 'desv. típica reward' in ll:
                res.reward_std = _extraer_float(l)

            elif 'duración media' in ll:
                res.dur_media = _extraer_float(l)

            elif 'desv. típica dur' in ll:
                res.dur_std = _extraer_float(l)

            elif 'media monedas' in ll:
                res.moneda_media = _extraer_float(l)

            elif 'media pasos' in ll:
                res.pasos_medio = _extraer_float(l)

    # ── Parseo de filas crudas ────────────────────────────────────────────────
    # Buscamos las líneas de datos (entre los dos separadores "─...─")
    if len(lineas_sep) >= 2:
        for l in lineas[lineas_sep[0]+1 : lineas_sep[1]]:
            l = l.strip()
            if not l:
                continue
            partes = l.split()
            if not partes[0].isdigit():
                continue   # cabecera

            # Formato RL:  Ep  Pasos  Dur(s)  Reward  Monedas  Victoria
            # Formato Sim: Ep  Dur(s)  Monedas  Victoria
            try:
                if tipo == 'RL' and len(partes) >= 6:
                    ep = dict(
                        pasos=int(partes[1]),
                        dur=float(partes[2]),
                        reward=float(partes[3]),
                        monedas=int(partes[4]),
                        victoria=partes[5] == 'SI'
                    )
                elif tipo == 'Simbólico' and len(partes) >= 4:
                    ep = dict(
                        dur=float(partes[1]),
                        monedas=int(partes[2]),
                        victoria=partes[3] == 'SI'
                    )
                else:
                    continue
                res.episodios_raw.append(ep)
            except (ValueError, IndexError):
                continue

    # Si el resumen no tenía dur_media (archivos sin actualizar), calculamos
    if res.dur_media is None and res.episodios_raw:
        durs = [e['dur'] for e in res.episodios_raw if 'dur' in e]
        if durs:
            res.dur_media = statistics.mean(durs)
            res.dur_std   = statistics.stdev(durs) if len(durs) > 1 else 0.0

    if res.moneda_media is None and res.episodios_raw:
        ms = [e['monedas'] for e in res.episodios_raw]
        res.moneda_media = statistics.mean(ms)

    return res


# ─── Tabla de comparación ─────────────────────────────────────────────────────

def _fmt(valor, decimales=2, unidad=''):
    if valor is None:
        return '  —  '
    return f"{valor:.{decimales}f}{unidad}"


def imprimir_tabla(resultados: list):
    print()
    print('=' * 80)
    print('  COMPARACIÓN DE AGENTES')
    print('=' * 80)

    col_w = [28, 8, 14, 14, 14, 14]
    cabecera = (
        f"{'Agente':<{col_w[0]}} "
        f"{'N ep':>{col_w[1]}} "
        f"{'Dur. media (s)':>{col_w[2]}} "
        f"{'Dur. std (s)':>{col_w[3]}} "
        f"{'Monedas/ep':>{col_w[4]}} "
        f"{'Victoria (%)':>{col_w[5]}}"
    )
    sep = '-' * sum(col_w + [len(col_w)])
    print(cabecera)
    print(sep)

    for r in resultados:
        estado = '' if r.completado else ' *'
        print(
            f"{r.nombre + estado:<{col_w[0]}} "
            f"{r.n_episodios:>{col_w[1]}} "
            f"{_fmt(r.dur_media, 2, ' s'):>{col_w[2]}} "
            f"{_fmt(r.dur_std,   2, ' s'):>{col_w[3]}} "
            f"{_fmt(r.moneda_media, 2):>{col_w[4]}} "
            f"{_fmt(r.tasa_victoria, 1, ' %'):>{col_w[5]}}"
        )

    print(sep)
    print("  * Experimento incompleto")
    print()

    # Tabla RL extra (reward)
    rl = [r for r in resultados if r.tipo == 'RL']
    if rl:
        print('─' * 80)
        print('  MÉTRICAS ADICIONALES (solo agentes RL)')
        print('─' * 80)
        cabecera2 = (
            f"{'Agente':<{col_w[0]}} "
            f"{'Reward medio':>{col_w[2]}} "
            f"{'Reward std':>{col_w[3]}} "
            f"{'Pasos medios':>{col_w[4]}}"
        )
        print(cabecera2)
        print(sep)
        for r in rl:
            print(
                f"{r.nombre:<{col_w[0]}} "
                f"{_fmt(r.reward_medio, 2):>{col_w[2]}} "
                f"{_fmt(r.reward_std,   2):>{col_w[3]}} "
                f"{_fmt(r.pasos_medio,  1):>{col_w[4]}}"
            )
        print(sep)
        print()


# ─── Generación LaTeX ─────────────────────────────────────────────────────────

def generar_latex(resultados: list) -> str:
    lines = []
    lines.append(r"% ── Tabla generada automáticamente por analyze_results.py ──")
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Comparación de agentes: duración de episodio, monedas recogidas y tasa de victoria}")
    lines.append(r"  \label{tab:resultados-comparacion}")
    lines.append(r"  \begin{tabular}{lrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Agente} & \textbf{Episodios} & \textbf{Dur. media (s)} & \textbf{Monedas/ep} & \textbf{Victoria (\%)} \\")
    lines.append(r"    \midrule")

    for r in resultados:
        dur  = _fmt(r.dur_media, 2)
        if r.dur_std is not None:
            dur += f" $\\pm$ {r.dur_std:.2f}"
        mon  = _fmt(r.moneda_media, 2)
        vic  = _fmt(r.tasa_victoria, 1)
        lines.append(f"    {r.nombre} & {r.n_episodios} & {dur} & {mon} & {vic} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return '\n'.join(lines)


def generar_latex_rl(resultados: list) -> str:
    rl = [r for r in resultados if r.tipo == 'RL']
    if not rl:
        return ''
    lines = []
    lines.append(r"% ── Tabla métricas RL ──────────────────────────────────────")
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Métricas de entrenamiento de los agentes de aprendizaje por refuerzo}")
    lines.append(r"  \label{tab:resultados-rl}")
    lines.append(r"  \begin{tabular}{lrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Agente} & \textbf{Reward medio} & \textbf{Reward std} & \textbf{Pasos medios} \\")
    lines.append(r"    \midrule")
    for r in rl:
        rew = _fmt(r.reward_medio, 2)
        std = _fmt(r.reward_std,   2)
        pas = _fmt(r.pasos_medio,  1)
        lines.append(f"    {r.nombre} & {rew} & {std} & {pas} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return '\n'.join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else '.'
    patron  = os.path.join(carpeta, 'resultados_*.txt')
    archivos = sorted(glob.glob(patron))

    if not archivos:
        print(f"[!] No se encontraron archivos 'resultados_*.txt' en: {carpeta}")
        print("    Ejecuta los experimentos en Unity primero.")
        return

    print(f"[+] Archivos encontrados: {len(archivos)}")
    for a in archivos:
        print(f"    {os.path.basename(a)}")

    resultados = []
    for a in archivos:
        r = parsear_archivo(a)
        if r is not None:
            resultados.append(r)
            print(f"[✓] {r.nombre} ({r.tipo}, {r.n_episodios} ep{'*' if not r.completado else ''})")

    if not resultados:
        print("[!] No se pudo parsear ningún archivo.")
        return

    # Ordenar: primero RL, luego simbólico; dentro de cada grupo, por nombre
    resultados.sort(key=lambda r: (0 if r.tipo == 'RL' else 1, r.nombre))

    imprimir_tabla(resultados)

    # Guardar tabla de texto
    ruta_txt = os.path.join(carpeta, 'comparison_table.txt')
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        for r in resultados:
            f.write(f"{r.nombre}\t{r.tipo}\t{r.n_episodios}\t"
                    f"{r.dur_media or ''}\t{r.dur_std or ''}\t"
                    f"{r.moneda_media or ''}\t{r.tasa_victoria or ''}\t"
                    f"{r.reward_medio or ''}\t{r.reward_std or ''}\t"
                    f"{r.pasos_medio or ''}\n")
    print(f"[✓] Tabla TSV guardada en: {ruta_txt}")

    # Guardar LaTeX
    latex = generar_latex(resultados)
    latex_rl = generar_latex_rl(resultados)
    ruta_latex = os.path.join(carpeta, 'latex_tables.tex')
    with open(ruta_latex, 'w', encoding='utf-8') as f:
        f.write(latex)
        if latex_rl:
            f.write('\n\n')
            f.write(latex_rl)
    print(f"[✓] Tablas LaTeX guardadas en: {ruta_latex}")


if __name__ == '__main__':
    main()
