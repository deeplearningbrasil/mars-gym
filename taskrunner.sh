# bin/bash

source activate deep-reco-gym
exec python -m luigi "$@"