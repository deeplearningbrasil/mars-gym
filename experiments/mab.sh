PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5000/rank"]}' --obs-batch-size 1 --val-size 0 --obs "RandomRankingRecommender"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MostPopularRankingRecommender"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5002/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MostPopularPerUserRankingRecommender"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5003/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MatrixFactorizationRankingRecommender"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5004/rank"]}' --obs-batch-size 1 --val-size 0 --obs "CVAERankingRecommender"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "CDAERankingRecommender"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5006/rank"]}' --obs-batch-size 1 --val-size 0 --obs "AVAERankingRecommender"

# Embedding

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "window_reward": 500}' --obs-batch-size 1 --val-size 0 --obs "e-greedy Ensamble"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank", "http://localhost:5002/rank", "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank", "http://localhost:5006/rank"]}' --obs-batch-size 1 --val-size 0 --obs "e-greedy Ensamble"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_ucb --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "UCB - Ensamble"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_softmax --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "window_reward": 500}' --obs-batch-size 1 --val-size 0 --obs "Softmax - Ensamble"

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank", "http://localhost:5006/rank"]}' --obs-batch-size 1 --val-size 0 --obs "Ensamble Contextual"


PYTHONPATH="." luigi --module recommendation.task.data_preparation.new_ifood IndexDataset