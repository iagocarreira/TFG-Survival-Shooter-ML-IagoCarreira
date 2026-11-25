# 🎮 Agentes autónomos en entornos 3D: IA simbólica vs. aprendizaje por refuerzo

**Trabajo de Fin de Grado**. Grado en Inteligencia Artificial, Universidade da Coruña (2025–2026). Calificación: **10/10**.

Este proyecto compara dos formas de construir agentes autónomos en un videojuego 3D de supervivencia hecho en Unity:

- **IA simbólica**: máquinas de estados finitos (**FSM**) y arquitectura de **subsunción**.
- **Aprendizaje por refuerzo (RL)**: **PPO**, con Unity ML-Agents, y **DQN**, en Python/PyTorch.

La comparación mide el **rendimiento**, el **tiempo de desarrollo** y la **capacidad de adaptación** de cada enfoque.

> 👥 Existe una segunda parte multiagente cooperativa (MAPPO, Shooter + Healer): [TFG-Survival-Shooter-ML-Multiagent](https://github.com/iagocarreira/TFG-Survival-Shooter-ML-Multiagent-IagoCarreira)

![Unity](https://img.shields.io/badge/Unity-2021.1-000000?logo=unity&logoColor=white)
![C#](https://img.shields.io/badge/C%23-239120?logo=csharp&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![ML-Agents](https://img.shields.io/badge/Unity%20ML--Agents-PPO-blue)

![Demo](Images/demo_1.gif)

---

## 🧠 Enfoques implementados

| Enfoque | Tipo | Dónde está |
|---|---|---|
| **FSM**, con estados de búsqueda, combate y huida | Simbólico | `Assets/Scripts/FSM/` |
| **Subsunción**, con capas de comportamiento priorizadas | Simbólico | `Assets/Scripts/Subsumption/` |
| **PPO** (Unity ML-Agents), incluidas variantes con imitación y *aimbot* | RL | `config/ppo_*.yaml`, `config/shooter*.yaml` |
| **DQN** propio en Python, con observaciones por rayos o por "brújula" y *reward shaping* | RL | `train-python/`, `play-python/` |

**Escenario:** el agente debe sobrevivir a oleadas de enemigos y recoger monedas. Cada episodio registra la duración, la recompensa, las monedas y si hay victoria, mediante `EpisodeResultsRecorder` y `SurvivalRecorder`.

**Técnicas de entrenamiento:** aprendizaje por imitación (demostraciones), **curriculum learning** por fases (más enemigos y más rápidos), *reward shaping* y entrenamiento por etapas (primero monedas y después supervivencia).

## 📁 Estructura

```text
Assets/Scripts/      Agentes en C#: FSM, Subsunción, PlayerAgent (ML-Agents), recorders
config/              Hiperparámetros de ML-Agents (PPO, DQN, imitación)
train-python/        Entrenamiento y evaluación del DQN (varias versiones y experimentos)
play-python/         Scripts para ejecutar los modelos DQN entrenados
analyze_results.py   Junta todos los resultados_*.txt y genera la tabla comparativa (texto y LaTeX)
Demos/               Demostraciones grabadas para imitation learning
```

## 🚀 Cómo ejecutarlo

**Requisitos:** Unity 2021.1, Python 3.x, `mlagents` y PyTorch.

```bash
# PPO con ML-Agents: abre la escena de entrenamiento en Unity y lanza
mlagents-learn config/shooter.yaml --run-id=shooter --time-scale=7

# DQN en Python, con Unity en modo Play
python train-python/train_dqn_brujula_v7.py
python play-python/play_dqn_brujula.py

# Tabla comparativa de resultados
python analyze_results.py ruta/a/resultados
```

Para ver un agente simbólico, abre la escena con el `FSMAgent` o el `SubsumptionBrain` asignado y pulsa *Play*.

## 🙏 Créditos

- El juego base es el [Survival Shooter](https://learn.unity.com/project/survival-shooter-tutorial) de **Unity Learn**.
- La integración inicial con ML-Agents parte de [juzdepeche/Unity-3D-Survival-Shooter-ML-agents](https://github.com/juzdepeche/Unity-3D-Survival-Shooter-ML-agents). Sobre ella añadí los agentes FSM y de subsunción, el DQN en Python, los nuevos escenarios (monedas y supervivencia), el curriculum, los sistemas de registro y el análisis de resultados.

---

Autor: **Iago Carreira García**. [LinkedIn](https://www.linkedin.com/in/iagocarreira)
