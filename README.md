<div align="center">

# CBMX — Condition-Based Maintenance & Survival Intelligence
### Analyse de Survie Censurée à Droite, Prédiction RUL & Maintenance à Coût Minimal

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-ElmasCore%20Charter-blue?style=flat-square)
![Validation](https://img.shields.io/badge/Validation-Unit%20GroupSplit%20Zero%20Leakage-10B981?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

</div>

---

## Problématique Scientifique

Comment modéliser conjointement le risque de défaillance censuré à droite et la durée de vie résiduelle (*RUL*) d'équipements industriels afin de minimiser le coût total de maintenance sous perte asymétrique ?

## Architecture du Domaine (`src/`)

```mermaid
flowchart LR
    TEL["src/telemetry/sensor_window.py"] --> SURV["src/survival/kaplan_meier.py"]
    TEL --> RUL["src/rul/rul_calibrator.py"]
    SURV --> DEC["src/decision/maintenance_policy.py"]
    TEL --> EVAL["src/evaluation/group_split.py"]
```

## Démarrage Rapide

```bash
pytest
python main.py
```

## Dictionnaire des Variables

| Variable | Unité | Description |
|---|---|---|
| `unit_id` | `int` | Identifiant unique d'équipement |
| `cycle` | `int` | Cycle opérationnel monotone |
| `temp_sensor` | `K` | Température capteur turbine |
| `vibration_sensor` | `mm/s` | Amplitude vibratoire |
| `failed` | `bool` | Indicateur d'événement (False = censuré) |
