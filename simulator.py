import torch
import types

class Simulator:
    """A toy simulator for a marine vessel.

    """
    def __init__(self, time_resolution=10, initial_position=torch.tensor([0,0]), 
                 initial_orientation=torch.tensor(0), initial_rudder_angle=torch.tensor(0)):
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