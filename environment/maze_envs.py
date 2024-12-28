
from .base_env import BaseMazeEnv

class MazeEnv(BaseMazeEnv):
    def __init__(
            self,
            width=10,
            height=10,
            num_keys=3,
            cell_size=20,
            num_obstacles=5,
            peek_distance=1,
            distance_type="manhattan",
            new_layout_on_reset=True,
    ):
        super().__init__(width,
                         height,
                         num_keys,
                         cell_size,
                         num_obstacles,
                         peek_distance,
                         distance_type,
                         new_layout_on_reset)

    def get_observation(self):
        return (
            tuple(self.agent_pos),
            tuple(self.goal_pos),
            self.goal_distance
        )

class MazeEnvKeys(BaseMazeEnv):
    def __init__(
            self,
            width=10,
            height=10,
            num_keys=3,
            cell_size=20,
            num_obstacles=5,
            peek_distance=1,
            distance_type="manhattan",
            new_layout_on_reset=True,
    ):
        super().__init__(width,
                         height,
                         num_keys,
                         cell_size,
                         num_obstacles,
                         peek_distance,
                         distance_type,
                         new_layout_on_reset)

    def get_observation(self):
        return (
            tuple(self.agent_pos),
            tuple(self.goal_pos),
            self.goal_distance,
            self.get_nearest_key()
        )