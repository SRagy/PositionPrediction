import torch
from torch.distributions import Bernoulli, Normal
from torch import Tensor
from sim_utils import arc_trace

pi = torch.tensor(torch.pi)

class Simulator:
    """A toy simulator for a marine vessel. Callable.

    """
    def __init__(self, 
                 time_per_action: int = 10, 
                 total_time: int = 1000,
                 wobbliness: Tensor = torch.tensor(0.1),
                 jerkiness: Tensor = torch.tensor(0.1),
                 ):
        """
        Args:
            time_per_action (int, optional): The interval during which rudder_angle and velocity
            are fixed, after which we stochastically sample new ones. Defaults to 10.
            total_time (int, optional): The total simulation time. Defaults to 1000.
            wobbliness (Tensor, optional): The probability of a change in rudder_angle.
            Defaults to torch.tensor(0.1).
            jerkiness (Tensor, optional): The probability of a change in velocity. Defaults to torch.tensor(0.1).
        """
        assert total_time % time_per_action == 0, "time_per_action must be a factor of total_time"

        self.time_per_action = time_per_action
        self.total_time = total_time
        self.wobbliness_dist = Bernoulli(wobbliness)
        self.jerkiness_dist = Bernoulli(jerkiness)

    def sample_new_angle(self, current_angle: Tensor):
        """A function to sample a new rudder direction. Changes angle with p(wobbliness).
        If angle is changed, there is an 80% change it resets to neutral and a 20% chance
        it samples from a Gaussian and adds this to the current angle.

        Args:
            current_angle (Tensor): a scalar torch tensor representing start angle
        """
        change_angle = self.wobbliness_dist.sample()
        # Ordinarily if the angle is changed, we want to straighten out.
        reset_angle = Bernoulli(0.8).sample()
        
        if change_angle and reset_angle:
            new_angle = torch.tensor(0.)
        elif change_angle and not reset_angle:
            # If a change occurs, and it isn't a reset, then we sample a new angle
            new_angle_delta = Normal(0,0.03).sample()
            new_angle = current_angle + new_angle_delta
        else: # No change.
            new_angle = current_angle

        

        # Constrain rudder angle to be between -pi/5 and pi/5 (i.e. +/- 36 degrees.)
        new_angle = max(-pi/5, min(new_angle, pi/5))

        return new_angle


    def sample_new_speed(self, current_speed: Tensor):
        """_summary_

        Args:
            current_speed (Tensor): speed at the start of the present step.

        Returns:
            Tensor: updated speed.
        """

        change_speed = self.jerkiness_dist.sample()

        if change_speed:
            new_speed_delta = Normal(0,1).sample()
        else:
            new_speed_delta = 0

        return current_speed + new_speed_delta


    def forward(self, 
                position: Tensor = torch.tensor([0.,0.]), 
                speed: Tensor = torch.tensor(7.), # 7m/s about 25km/h or 13.5 knots.
                orientation: Tensor = torch.tensor(0.), 
                rudder_angle: Tensor = torch.tensor(0.)):
        """Simulates the trajectory.

        Args:
            position (Tensor, optional): position at initialisation. Defaults to torch.tensor([0.,0.]).
            speed (Tensor, optional): signed speed. Defaults to torch.tensor(7.).
            orientation (Tensor, optional): the direction the ship is facing. Defaults to torch.tensor(0.).
            rudder_angle (Tensor, optional): Defaults to torch.tensor(0.).

        Returns:
            Tuple(Tensor, Tensor, Tensor): position, orientation, trajectory
        """

        trajectory = position.reshape(1,2) # To enable concatenation

        for i in range(self.total_time//self.time_per_action):
            positions, orientation = arc_trace(orientation, position, speed, rudder_angle, self.time_per_action)
            speed = self.sample_new_speed(speed)
            rudder_angle = self.sample_new_angle(rudder_angle)
            trajectory = torch.cat([trajectory, positions])
            position = positions[-1]

        return position, orientation, trajectory


    def __call__(self, 
                position: Tensor = torch.tensor([0.,0.]), 
                speed: Tensor = torch.tensor(7.), # 7m/s about 25km/h or 13.5 knots.
                orientation: Tensor = torch.tensor(0.), 
                rudder_angle: Tensor = torch.tensor(0.)):
        """Simulates the trajectory

        Args:
            position (Tensor, optional): position at initialisation. Defaults to torch.tensor([0.,0.]).
            speed (Tensor, optional): signed speed. Defaults to torch.tensor(7.).
            orientation (Tensor, optional): the direction the ship is facing. Defaults to torch.tensor(0.).
            rudder_angle (Tensor, optional): Defaults to torch.tensor(0.).

        Returns:
            Tuple(Tensor, Tensor, Tensor): position, orientation, trajectory
        """
        return self.forward(position, speed, orientation, rudder_angle)