PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5000/rank"]}' --obs-batch-size 1 --val-size 0 --obs "Random"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MostPopular"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5002/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MostPopularPerUser"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5003/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MatrixFactorization"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5004/rank"]}' --obs-batch-size 1 --val-size 0 --obs "CVAE"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "CDAE"

# PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5006/rank"]}' --obs-batch-size 1 --val-size 0 --obs "AVAE"

# Embedding

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "window_reward": 500}' --obs-batch-size 1 --val-size 0 --obs "e-greedy"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_ucb --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "UCB"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_softmax --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "Softmax"


PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "e-greedy Contextual"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_softmax --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank", "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "Contextual Softmax"

PYTHONPATH="." luigi --module recommendation.task.data_preparation.new_ifood IndexDataset