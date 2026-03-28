# Epics 6, 7, 8: Future Epics — Design Spec

**Date:** 2026-03-28
**Author:** Hasan Goni
**Status:** Approved

---

## Epic 6: AOI Bond Wire & Surface Defect Synthesis

**Package:** `udm_epic6/`
**CLI:** `udm-epic6`

Generates synthetic AOI images with bond wire defects (bent, broken, lifted).
Uses bezier curve geometry for wire paths and simple optical rendering for
specular reflections on wire surfaces.

**Defect types:** bent wire, broken wire, wire lift-off
**Key modules:** wire geometry, defect generator, AOI renderer

---

## Epic 7: Chromasense Multi-Spectral Integration

**Package:** `udm_epic7/`
**CLI:** `udm-epic7`

Generates synthetic multi-spectral images matching Chromasense equipment.
Models material reflectance at multiple wavelengths (450nm-850nm) to detect
delamination, contamination, and oxidation defects invisible in standard imaging.

**Key modules:** wavelength model, defect spectra, spectral renderer

---

## Epic 8: Universal Model Support & Integration

**Package:** `udm_epic8/`
**CLI:** `udm-epic8`

Unified pipeline combining all epics (1-7) into one cross-modality synthetic
data generation system. Includes a modality registry, dataset export to
COCO/YOLO/HuggingFace formats, and cross-modality evaluation.

**Key modules:** unified pipeline, modality registry, dataset export
