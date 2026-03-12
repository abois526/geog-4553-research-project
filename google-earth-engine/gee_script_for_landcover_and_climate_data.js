/*--------------------------------------
/ SECTION: Script Header
/
/ NOTES:
/ - need to decide on source(s) for climate data, using terraclimate here but may
/   want to use something else like DAYMET (finer res), but need soil data separate 
/   in that case (looking more into this)
/
/ Exports to Google Drive:
/   1. AAFC 2022 Annual Crop Inventory (land cover)
/   2. TerraClimate 2022 — mean temperature (tmean)
/   3. TerraClimate 2022 — total precipitation (ppt)
/   4. TerraClimate 2022 — mean soil moisture (soil)
/
/ All layers:
/   - Clipped to boundary
/   - Reprojected to EPSG:26911 (UTM Zone 11N)
/   - Exported at 10 m resolution
/   - Exported as GeoTIFF to Google Drive
/-------------------------------------*/


/*--------------------------------------
/ SECTION: Step 1 - set up study area and export settings, view coverage
/-------------------------------------*/
// --- Option A: Using shapefile ---
// Can use the boundary for the MD
// updated to reflect our filepath 
var ranchlandBoundary = ee.FeatureCollection("projects/project-sds-473702/assets/MD_bound_zipped");
var studyArea = ranchlandBoundary.geometry();


// --- Option B: Approximate bounding box (use this to test while uploading your shapefile) ---
// rough bounding box, checked manually that it fits
// would need to clip to bounds later
var studyArea = ee.Geometry.Rectangle([-114.767597, 49.612230, -113.976582, 50.332934]);

// export settings
// UTM Zone 11N
var CRS = 'EPSG:26911';
// 10m res to match cost/distance disturbance layers
var SCALE = 10;
// Google Drive folder name, created automatically, may need update
var FOLDER = 'GEE_RanchlandExports';
var YEAR = 2022;


// for verifying coverage (checked and looks good)
Map.centerObject(studyArea, 9);
Map.addLayer(ee.Image().paint(ee.FeatureCollection([ee.Feature(studyArea)]), 0, 2),
  { palette: ['red'] }, 'Study Area Boundary');

print('Study area bounds:', studyArea.bounds());


/*--------------------------------------
/ SECTION: Step 2 - retrieve land cover data
/-------------------------------------*/
// Canadian-specific, annual product, covers all of Alberta.
// Class values and legend: https://www.agr.gc.ca/atlas/aci

var aafc = ee.ImageCollection('AAFC/ACI')
  .filterDate('2022-01-01', '2022-12-31')
  .first()                         // only one image per year
  .select('landcover')
  .clip(studyArea);

// Visualise in the map (optional — helps confirm coverage)
Map.addLayer(aafc, { min: 10, max: 255, palette: ['green', 'yellow', 'brown', 'blue'] },
  'AAFC Land Cover 2022', false);

// Export
Export.image.toDrive({
  image: aafc,
  description: 'AAFC_LandCover_2022',
  folder: FOLDER,
  fileNamePrefix: 'landcover_aafc_2022',
  region: studyArea,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

print('✓ AAFC land cover layer ready — check Tasks panel to start export');


/*--------------------------------------
/ SECTION: Step 3 - retrieve climate data
/-------------------------------------*/
// TerraClimate: ~4 km native resolution, monthly, global.
// NOTE: may want to use other method
// We aggregate to annual summaries (mean temp, total precip, mean soil moisture)
// to match your label year (2022).
//
// Variables used:
//   tmmx  — maximum temperature (×0.1 °C, so divide by 10)
//   tmmn  — minimum temperature (×0.1 °C, so divide by 10)
//   pr    — precipitation (mm/month)
//   soil  — soil moisture (mm, top ~1.5 m)
//
// For SDM, growing-season means (April–September) are more ecologically
// meaningful for Downy Brome. Both annual and growing-season versions
// are exported below — use whichever you prefer.

var terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
  .filterDate('2022-01-01', '2022-12-31')
  .filterBounds(studyArea);

// ── Helper: clip each image in the collection ──
var clipToStudyArea = function (img) { return img.clip(studyArea); };
var tc = terraclimate.map(clipToStudyArea);


// ── 4a. Mean Annual Temperature ──────────────────────────────────────────────
// Average of monthly mean temperatures (mean of tmmx and tmmn), scaled to °C

var tmean_annual = tc.select('tmmx').mean()
  .add(tc.select('tmmn').mean())
  .divide(2)        // average of max and min = mean temp
  .multiply(0.1)    // TerraClimate scale factor (units: 0.1 °C)
  .rename('tmean_annual_C');

Map.addLayer(tmean_annual, { min: -10, max: 20, palette: ['blue', 'white', 'red'] },
  'Mean Annual Temp 2022', false);

Export.image.toDrive({
  image: tmean_annual,
  description: 'TerraClimate_Tmean_Annual_2022',
  folder: FOLDER,
  fileNamePrefix: 'tmean_annual_2022',
  region: studyArea,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});


// ── 4b. Total Annual Precipitation ───────────────────────────────────────────
// Sum of monthly precipitation (mm/year)

var precip_annual = tc.select('pr').sum()
  .rename('precip_annual_mm');

Map.addLayer(precip_annual, { min: 200, max: 800, palette: ['white', 'blue'] },
  'Total Annual Precip 2022', false);

Export.image.toDrive({
  image: precip_annual,
  description: 'TerraClimate_Precip_Annual_2022',
  folder: FOLDER,
  fileNamePrefix: 'precip_annual_2022',
  region: studyArea,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});


// ── 4c. Mean Annual Soil Moisture ─────────────────────────────────────────────

var soilmoisture_annual = tc.select('soil').mean()
  .rename('soilmoisture_annual_mm');

Map.addLayer(soilmoisture_annual, { min: 0, max: 200, palette: ['red', 'yellow', 'green'] },
  'Mean Annual Soil Moisture 2022', false);

Export.image.toDrive({
  image: soilmoisture_annual,
  description: 'TerraClimate_SoilMoisture_Annual_2022',
  folder: FOLDER,
  fileNamePrefix: 'soilmoisture_annual_2022',
  region: studyArea,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

print('✓ TerraClimate layers ready — check Tasks panel to start exports');

// log that all exports are finished
print('All exports queued. Open the Tasks panel (top right) and click Run on each task.');