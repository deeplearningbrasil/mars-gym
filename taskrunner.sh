# bin/bash


export LC_ALL=C.UTF-8
export LANG=C.UTF-8

source activate deep-reco-gym
#exec env PYTHONPATH="." python -m luigi "$@"
PYTHONPATH="." python -m luigi "$@"

#cd /app/output
#python -m  http.server 8502 & 

#cd /app
#exec streamlit run tools/eval_viz/app.py 
