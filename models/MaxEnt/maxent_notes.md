# MaxEnt Notes

## Info RE: logs

- performed a 5-fold run then ran it again after adjusting CBI implementation to be more accurate 
- had bad CBI metrics and one of the spatial folds was skipped entirely
  - ./data/outputs/logs/output-5-fold-run-1.log
  - ./data/outputs/logs/output-5-fold-run-2.log

- adjusted to 4 spatial folds to try to get better results by creating larger spatial blocks to try to mitigate issue of the one fold being skipped entirely
- produced better results overall but still running into that issue (only 1 point fell within the previously empty block)
- looks to be an issue with the data itself, not the model. it seems that there just isn't enough data throughout the study region and the clustering around the SE-ish region causes this. spatial thinning of points does not seem like it would improve results, so leaving it at 209 presence points
  - ./data/outputs/logs/output-4-fold.log