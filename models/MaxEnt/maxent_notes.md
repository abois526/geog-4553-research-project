# MaxEnt Notes

## Info RE: logs

- included some deprecated logs from earlier iterations of the code that live within `./data/outputs/logs/deprecated/`
  - one spatial fold was skipped entirely
  - adjusted to 4 spatial folds to try to get better results by creating larger spatial blocks to try to mitigate issue of the one fold being skipped entirely
  - produced better results overall but still running into that issue (only 1 point fell within the previously empty block)
  - wide variance in points per fold

- ended up deriving PCA components from the embeddings to aim for better results
  - latest version utilizes this and produced better overall results, still needs interpretation based upon weak CBI stats
  - looks to be an issue with the data itself, not the model. it seems that there just isn't enough data throughout the study region and the clustering around the SE-ish region causes this. spatial thinning of points helped, but still need to interpret the stats well because of the weak CBI stats (though better in one of the folds, so exported them to take a look at them)
  - the good logs live within `./data/outputs/logs/`
