# Downy Brome Habitat Suitability Mapping — Project Brainstorming

*NOTE: Much of the content in this document is generated with AI assistance and has been edited to correct and verify info. All sources should be independently verified before relying on them for analysis decisions.*

## Notes Regarding AI Prompting and Limitations

### From Andrew
- Utilized methods from the GenAI lecture slides and understanding of prompting to generate this document with Claude Sonnet and Opus 4.6 Claude Opus 4.6. Aim is to help us brainstorm potential analytical workflows, methods, predictor selection, etc. Will be useful as a first-pass but we will want to interrogate the suggestions and iterate for better results and arrive to informed decisions as we work through the project.
- Opus way overshot the document, went back and cleaned it up so it is more of a high-level overview to help with brainstorming
- Removed the hallucinated dataset info, formatted table into a new file that we can populate when we work on it
  - No data files were uploaded in adherence with the provided letter specifying that the data is for academic and research purposes only and must not be published distributed, or shared publicly. I mentioned the filenames and gave descriptions of the data that we had and it hallucinated a bunch of info from it.

- Model selection:
  - CNN strong candidate, studied in class, maybe ResNet but given the uncertainty of the data we'll have at this stage, we should gather some more info then decide what models we want to go with

---

## Submitted Abstract

**Habitat Suitability Mapping of Downy Brome (Bromus Tectorum) in the Municipal District of Ranchland No. 66, Alberta**

Downy Brome, an invasive annual grass designated as a noxious weed under the Alberta Weed Control Act, poses a growing threat to grassland ecosystems and rangeland management in southern Alberta. The species competes with native vegetation, contributes to increased wildfire risk, and can persist aggressively once established. This study aims to develop a habitat suitability map to predict the potential spatial distribution of Downy Brome across the Municipal District of Ranchland No. 66.

To achieve this, multiple modelling approaches will be applied and compared, including MaxEnt and deep learning models. Model performance will be evaluated to identify the most effective method for predicting Downy Brome occurrence. The predictions from individual models will also be integrated into an ensemble model to produce a final habitat suitability map with improved predictive reliability. The analysis will incorporate a range of environmental and land-use variables, including satellite imagery, land cover, disturbance layers, and climate-related factors derived from both locally provided and publicly available spatial datasets.

This research addresses a gap in spatial information on invasive species risk in this region. The resulting habitat suitability map will support the Municipal District of Ranchland No. 66 by helping identify areas most vulnerable to invasion, allowing land managers to prioritize monitoring and control efforts.

---

## 1. Key Ecological Characteristics of Downy Brome

Understanding the biology of *Bromus tectorum* informs predictor selection. The key characteristics and their modelling implications are summarized below.

| Characteristic | Summary | Predictor Implications |
|---|---|---|
| **Regulatory status** | Classified as a noxious weed under the Alberta Weed Control Act (2010). Landowners must control it. Some municipalities have elevated its status via bylaw. | — |
| **Climate preferences** | Cool-season annual. Germinates in fall/early spring, dies in hot dry weather. Precipitation timing (late fall/early spring moisture) matters more than total amount. | Mean annual temp, seasonal temp range, precipitation seasonality, aridity indices. Sources: WorldClim, ClimateNA. |
| **Soil preferences** | Strongly prefers coarse-textured (sandy/gravelly) soils. Tolerates calcareous and low-fertility soils. Does not establish well on acidic, nutrient-poor soils. | Soil texture, soil pH, organic matter. Source: AAFC CanSIS. |
| **Terrain & topography** | Associated with south/west-facing slopes (warmer, drier). Flat to moderate slopes also suitable. | Slope, aspect, elevation from DEM. Source: NRCan CDEM or USGS. |
| **Disturbance dependency** | Requires disturbance to establish — poor competitor against healthy perennial vegetation. Overgrazing, vehicle traffic, pipelines, and trails create conditions it exploits. | Distance to trails, pipelines; disturbance layers from MD-provided shapefiles; land use/land cover. |
| **Fire adaptation** | Accumulates dry flammable litter by summer. Strong post-fire colonizer; seeds can survive low-intensity burns. | Fire history (binary/time-since-fire) if available via Alberta Wildfire open data. |
| **Spread mechanisms** | Seeds spread by animals, vehicles, contaminated hay, human clothing. Off-road vehicles (quads, dirt bikes) are a documented dispersal vector. | Proximity to roads, trails, and access routes as dispersal corridor predictors. |

**Key sources:** Alberta Weed Control Act (2010); Alberta Invasive Species Council fact sheet; Gerling (2007) Agdex 641-15; County of Newell weed control program description. Broader *B. tectorum* literature (especially western US) should also be consulted.

---

## 2. Potential Analysis Workflows

**Overall structure:** One baseline model (MaxEnt) + two deep learning models → ensemble. Three models total, combined into a final habitat suitability map.

### 2.1 Baseline — MaxEnt

Presence-only SDM using maximum entropy modelling. Well-established benchmark for the deep learning comparison. Two implementation options: Java MaxEnt (most cited, standalone app) or `elapid` Python package (integrates into a Python pipeline, scikit-learn conventions, built-in spatial CV tools). Choose based on whether a fully Python-based pipeline is preferred.

### 2.2 Deep Learning Options

| Option | Input Data | Spatial Context | Compute Needs | Priority |
|---|---|---|---|---|
| **A — MLP** | Tabular (same as MaxEnt) | None (point-level) | CPU sufficient | **High** — direct comparison to MaxEnt |
| **B — CNN** | Multi-band raster patches | Yes (spatial neighbourhood) | GPU recommended | **High** — tests whether spatial context improves prediction |
| C — ViT | Multi-band raster patches | Yes (via self-attention) | GPU required | Lower — likely needs more data than available |
| D — CNN + temporal imagery | Sentinel-2 time series patches | Yes + temporal | GPU required | Lower — interesting given DB phenology but heavy data prep |
| E — Multimodal (MLP + CNN) | Tabular + raster patches | Both | GPU recommended | Lower — attempt only if A and B are already working |

**Literature note:** A 2024 bioRxiv preprint evaluating deep learning SDMs found that DL models matched but did not consistently surpass MaxEnt/Random Forest, and performed weaker for species with narrow ranges and few data points. Information leakage across splits can inflate CNN metrics — careful spatial CV design is required.

### 2.3 Ensemble Modelling

Combine the three models' predictions into a consensus output. Primary approach: **weighted averaging** (weight by AUC/TSS from cross-validation). Secondary: **committee averaging** (binary agreement count — intuitive for the MD audience). Other options include simple averaging, median, or stacking if time permits.

### 2.4 Final Map Outputs

- **Continuous suitability map** (0–1 raster) — primary deliverable
- **Binary map** (thresholded suitable/not-suitable) — document threshold method chosen
- **Risk-priority map** (optional) — overlay suitability with proximity to infrastructure/known infestations

**Production:** Generate all rasters in Python (rasterio + numpy). Use ArcGIS or QGIS for final cartographic layout and export.

---

## 3. Potential Environmental Predictor Layers

| Predictor | Type | Source |
|---|---|---|
| Slope | Continuous | Derived from DEM (NRCan / USGS) |
| Aspect | Categorical | Derived from DEM |
| Elevation | Continuous | DEM |
| Disturbance - Cabin Ridge (2021) | TODO | Provided dataset |
| Disturbance - Crown Land Designated Trails | TODO | Provided dataset |
| Disturbance - ERCB pipelines | TODO | Provided dataset |
| Disturbance - Harvest blocks 2022–2024 | TODO | Provided dataset |
| Disturbance - Low pressure lines | TODO | Provided dataset |
| Disturbance - Roads Reclaimed | TODO | Provided dataset |
| Mean annual precipitation | Continuous | WorldClim / ClimateNA |
| Precipitation seasonality | Continuous | WorldClim / ClimateNA |
| Mean annual temperature | Continuous | WorldClim / ClimateNA |
| Soil texture (sand content) | Continuous | AAFC CanSIS |
| Soil pH | Continuous | AAFC CanSIS |
| Roads reclaimed 2022–2024 | Binary/categorical | Provided dataset |
| Distance to water bodies | Continuous | Provided dataset |
| Distance to water course | Continuous | Provided dataset |
| Land cover / land use | Categorical | ABMI / AAFC |
| Fire history *(if available)* | Binary/time-since-fire | Alberta Wildfire open data |