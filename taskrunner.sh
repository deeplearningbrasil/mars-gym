# bin/bash
source activate deep-reco-gym
exec env PYTHONPATH="." python -m luigi "$@"

cd /app/output
exec python -m  http.server 8502 & 

cd /app
exec streamlit run tools/eval_viz/app.py 