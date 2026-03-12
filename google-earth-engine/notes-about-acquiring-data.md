# Notes for Geog 4553 Project data acquisition 

## Checklist

[wip] presence / absence labels
- got tree coverage, just need to remove noted absences and disturbance areas then generate random points
  - need to verify and make sure it makes sense taking into consideration most of the noted presence data is noted along roads/trails, don't want models to arrive to weird conclusions

[X] google satellite embeddings for baseline model X vars

[X] cost distance raster for disturbance layers

[wip] land cover data
- can get from GEE => check when created, need 2022-2025
- look into:
  - AAFC Annual Crop Inventory
    - search for "AAFC/ACI"
    - 2022 should be available 
  - MODIS Land Cover (MCD12Q1)
    - should be available through 2023
    - likely coarser resolution (need to verify)
  - North America Land Cover (NALCMS)
    - may not be recent enough
- may want to start with AAFC, see if it works for the study area

[wip] climate data
  [ ] temperature
  [ ] rainfall
  [ ] soil moisture
- many datasets for climate data, think about which one to use
- commonly used:
  - for temp and rainfall:
    - ERA5-Land (should cover 2022)
    - TerraClimate (apparently pretty popular)
    - DAYMET (excludes soil moisture)
  - for soil moisture:
    - need to decide on tradeoff between soil resolution and different sets 

[ ] for training: value extraction to label locations
- Lan says should be easy

[ ] for prediction: generate entire map from the input data
- all rasters must be exactly matched (same resolution and coverage)

## Presence / Absence Data Workflow

For absence:
- completed:
  - projected everything to NAD 1983 UTM Zone 11N (some was, some wasn't)
  - created new shapefile with confirmed absences
  - combined disturbance lines and polygons 
  - clipped everything within the study boundary
  - added a 100m buffer to the disturbance lines and polygons to account for potential spread
  - get forested area polygon (possibly from the satellite embedding data, maybe other source)

- next:
  - use erase tool with forested polygon to get feature w/ only the forested areas that are away from disturbance features and noted absences 
  - merge all the absence zones
  - create random points constrained to merged absence polygon aiming for no more than about a 1:2 ratio of presence:absence

---

## Approx Bounds for the MD

North Bound:  50.332934, -114.219654
East Bound:   49.944776, -113.976582
South Bound:  49.612230, -114.341877
West Bound:   50.098303, -114.767597