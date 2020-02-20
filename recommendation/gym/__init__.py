from gym.envs.registration import register

register(
    id='ifood-recsys-v0',
    entry_point='recommendation.gym.envs:IFoodRecSysEnv',
)