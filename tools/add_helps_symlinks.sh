#!/bin/bash

# create directory for large data files

mkdir data

## Sharing files

# To save space on HELPS, we share large files via symbolic links. 

### COCO is special, so we add it separately.
ln -s /srv/sdb1/image_team/coco/ ./data/
### To make loading faster, the program caches the roidb files. To add the shared caches, run:
ln -s /srv/sdb1/image_team/roidb_cache/ ./data/cache
### We all need access to the same mixture datasets.
ln -s /srv/sdb1/image_team/mixtureDatasets/ ./data/
### We all need access to the same mixture datasets.
ln -s /srv/sdb1/image_team/annoAnalysis/ ./data/


exit 0
