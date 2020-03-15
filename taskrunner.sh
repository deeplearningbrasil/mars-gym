# bin/bash

#exec conda activate deep-reco-gym
source activate deep-reco-gym
exec python -m luigi "$@"