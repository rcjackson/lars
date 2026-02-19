# Radar Image Labelling Codebook

A reference guide for annotators labelling radar imagery with LARS. Use this codebook to ensure consistent, reproducible labels across all annotators and sessions.

---

## 1. Overview

The purpose of this section is to label radar imagery for warm-season precipitation

- **Radar type:** ARM CSAPR2
- **Data source:** csapr2cmac.c1 datastream
- **Geographic scope:** Bankhead National Forest
- **Labelling task:** scene classification

---

## 2. Data Description

### 2.1 Input Fields

| Field | Units | Description |
|-------|-------|-------------|
| Reflectivity (Z) | dBZ | Intensity of returned radar signal |


### 2.2 Image Format

- **Spatial resolution:** 1 km by 1 km
- **Temporal resolution:** 10-minute intervals
- **Projection:** Polar coordinates projected onto 
- **Color scale:** (describe or attach scale reference)

---

## 3. Label Classes

Each image or region-of-interest must be assigned exactly one primary class. 

### 3.1 Primary Classes

| Label | Description |
|------------------------|-----------------------------------------------------------------------------|
| No Precipitation | No significant return; background noise only. It looks like a blue quasi-circle in the center of the image with some yellows mixed in. |
| Stratiform Precipitation | Widespread, precipitation mostly below 40 dBZ (green colors) |
| Isolated Convection | Intense, localized cores; high reflectivity (≥ 45 dBZ, localized dark orange to red colors) |
| Linear Convection | A region of red colors (reflectivity > 45 dBZ) organized into a quasi-linear structure |
| Ambiguous / Uncertain | Cannot be classified with confidence |


---

## 5. Labelling Procedure

1. Use :code:`lars.preprocessing.preprocess_radar_data` to generate images and a .csv file
2. The csv file will label all categories as UNKNOWN. This is just a placeholder for hand labelling.
3. According to the criteria above, label all images in the 'file_path' column of the .csv file.

---

## 6. Annotator Guidelines

- When in doubt, default to the **more conservative** class (e.g. Stratiform over Convective).
- Use the provided example gallery (Section 8) to calibrate your judgement.
- Inter-annotator agreement should be checked periodically; raise disagreements with the team lead.


---

## 7. Quality Control

| Check | Method |
|-------|--------|
| Completeness | All images have a primary label |
| Consistency | Random sample reviewed by second annotator |
| Agreement metric | Cohen's κ computed per annotator pair |
| Outlier review | Labels deviating from model predictions flagged for review |

---

## 8. Example Gallery

*(Attach or link representative images for each primary class here.)*

| Class | Example Image | Notes |
|--------------------------|-----------------------------------|--------------------------|
| Stratiform Rain | `examples/stratiform_01.png` | Clear bright band at 2 km |
| Convective Rain | `examples/convective_01.png` | 55 dBZ core, anvil visible |
| No Precipitation | `examples/clutter_01.png` | Stationary radial spokes |

---

## 9. Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-02-19 | Robert Jackson | Initial release |

---

## 10. References

- Rinehart, R. E. (2004). *Radar for Meteorologists* (4th ed.).
- American Meteorological Society Glossary: https://glossary.ametsoc.org

