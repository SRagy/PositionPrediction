import torch
import types
from sim_utils import arc_trace

class Simulator:
    """A toy simulator for a marine vessel.

    """
    def __init__(self, time_resolution=10, 
                 wobbliness=torch.tensor(0.1),
                 oomphness=torch.tensor(0.1),
                 ):
        """_summary_

        Args:
            time_resolution (int, optional): _description_. Defaults to 10.
            initial_position (_type_, optional): _description_. Defaults to torch.tensor([0,0]).
            initial_orientation (_type_, optional): _description_. Defaults to torch.tensor(0).
            initial_rudder_angle (_type_, optional): _description_. Defaults to torch.tensor(0).
        """
        self.time_resolution = time_resolution
        self.position = initial_position
        self.orientation = initial_orientation
        self.rudder_angle = initial_rudder_angle

    def time_step_sim(self):
        self

    def forward(self):
        self

    def __call__(self, inputs):
        self.forward(inputs)