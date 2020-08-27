import collections
from typing import Tuple, List, Any, Dict, Optional

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym import utils

ITEM_METADATA_KEY = "item_metadata"


class RecSysEnv(gym.Env, utils.EzPickle):
    metadata = {"render.modes": []}
    reward_range = [0.0, 1.0]

    def __init__(
        self,
        dataset: pd.DataFrame,
        item_column: str,
        number_of_items: int,
        available_items_column: Optional[str] = None,
        item_metadata: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__()
        self._dataset = dataset.copy()
        self._item_metadata = item_metadata
        self._item_column = item_column
        self._available_items_column = available_items_column

        self._number_of_items = (
            number_of_items  # number_of_itemsdataset[item_column].max() + 1
        )

        if available_items_column:
            assert isinstance(
                self._dataset[available_items_column].values[0], collections.Sequence
            )
            available_items = np.zeros(
                (len(self._dataset), self._number_of_items), dtype=np.int8
            )
            non_zero_indices = [
                (i, available_item)
                for i, available_items in enumerate(
                    self._dataset[available_items_column]
                )
                for available_item in available_items
            ]
            i, j = zip(*non_zero_indices)
            available_items[i, j] = 1
            self._dataset[available_items_column] = list(iter(available_items))

        self._obs_dataset: List[dict] = self._dataset.drop(
            columns=[item_column]
        ).to_dict("records")

        self.action_space = spaces.Discrete(self._number_of_items)

        observation_space = {
            key: self._convert_value_to_space(key, value)
            for key, value in self._obs_dataset[0].items()
        }
        if item_metadata is not None:
            observation_space[ITEM_METADATA_KEY] = spaces.Dict(
                {
                    key: spaces.Box(
                        value.min(), value.max(), shape=value.shape, dtype=value.dtype
                    )
                    for key, value in item_metadata.items()
                }
            )
        self.observation_space = spaces.Dict(observation_space)
        self._current_index = 0

    def _convert_value_to_space(self, key: str, value: Any) -> spaces.Space:
        if isinstance(value, list):
            value = np.array(value)

        if key == self._available_items_column:
            return spaces.MultiBinary(self._number_of_items)
        if isinstance(value, int):
            return spaces.Discrete(self._dataset[key].max() + 1)
        elif isinstance(value, float):
            return spaces.Box(
                self._dataset[key].min(), self._dataset[key].max(), shape=(1,)
            )
        elif isinstance(value, np.ndarray):
            if issubclass(value.dtype.type, np.integer):
                return spaces.MultiDiscrete(
                    [np.max(self._dataset[key].max()) + 1] * len(value)
                )
            elif issubclass(value.dtype.type, np.floating):
                return spaces.Box(
                    self._dataset[key].min(),
                    self._dataset[key].max(),
                    shape=value.shape,
                )
        raise ValueError(
            "Unkown type in the observation space for {}:{}".format(key, value)
        )

    def _compute_stats(self, action: float) -> dict:
        # TODO: Choose which batch metrics to return
        return {}

    def _compute_reward(self, action: int) -> float:

        return float(
            self._dataset.iloc[self._current_index][self._item_column] == action
        )

    def _get_next_ob(self) -> dict:
        ob = self._obs_dataset[self._current_index].copy()
        if self._item_metadata is not None:
            ob[ITEM_METADATA_KEY] = self._item_metadata
        else:
            ob[ITEM_METADATA_KEY] = None
        return ob

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        reward = self._compute_reward(action)
        info = self._compute_stats(action)

        self._current_index += 1

        done = (self._current_index + 1) == len(self._dataset)
        next_ob = self._get_next_ob() if not done else None

        return next_ob, reward, done, info

    def reset(self) -> dict:
        self._current_index = 0
        return self._get_next_ob()

    def render(self, mode="human"):
        pass

    def close(self):
        pass
