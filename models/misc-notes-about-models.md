# Misc Notes

## Models

### For Baseline (MaxEnt)
- [Java - Maxent](https://biodiversityinformatics.amnh.org/open_source/maxent/)
    - original implementation
    - looks like theres a GUI available, should have user-friendly interface
    (Noting this article in case it may be useful, may want to find something more recent)[https://element84.com/geospatial/spatial-analysis/preparing-data-for-maxent-species-distribution-modeling-using-r/]
- [Python - elapid](https://earth-chris.github.io/elapid/)
    - SDM modelling tools for Python
- R libraries available like maxnet (features much of the Java functionality) and can find linked through the "Java - Maxent" page
- Found this [curated list of R packages for species distribution modelling](https://github.com/helixcn/sdm_r_packages) and noting it down here in case it might be useful to look through

### DL Models
- CNNs are often used in SDM
- early ideas and notes from searching around about CNN architectures used in SDMs, need to look into all this to verify and see what is/isn't a good fit for this project:
    - Inception-v3
    - VGG Architecture
        - VGG16
            - maybe a better balance of accuracy, speed, lower risk of overfitting
        - VGG19
            - can capture more complex features due to its added depth
    - TResNet
    - ResNet