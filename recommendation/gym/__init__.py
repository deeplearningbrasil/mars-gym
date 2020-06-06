from gym.envs.registration import register

register(
    id='recsys-v0',
    entry_point='recommendation.gym.envs:RecSysEnv',
)
