import torch
from torch import Tensor, cos, sin
from typing import List, Tuple

def arc_trace(orientation: Tensor, 
              velocity: Tensor, 
              rudder_angle: Tensor, 
              trajectory: List[Tensor],
              time_step: int = 10,
              record_resolution: int = 1) -> Tuple(Tensor, Tensor):
    """A function for tracing an arc-segment of a ship's trajectory.
    Toy model which simply assumes that the arc radius depends purely on rudder angle.
    
    Args:
        orientation (Tensor): the ship's angle relative to coordinate axes. 
        velocity (Tensor): the ship's velocity for the present timestep.
        rudder_angle (Tensor): the rudder angle relative to the ship's orientation.
        trajectory (List[Tensor]): the trajectory so far, as a list of position coordinates.
        time_step (float): amount of time over which to integrate trajectory.
        record_resolution (float): the time resolution with which the position of ship is recorded.
    """
    assert torch.abs(rudder_angle) <= torch.pi/4, "abs(rudder_angle) must be <= pi/4"
    assert time_step%record_resolution==0, "record_resolution must divide time_step without remainder"
    
    # assume a 100 unit arc_radius at 45 degree rudder
    arc_radius = 100/torch.tan(rudder_angle)
    for i in range(time_step//record_resolution):
    total_distance  = velocity*time_step

def arc_trace_centered(velocity: Tensor, 
              rudder_angle: Tensor, 
              time_steps: int = 10,
              record_resolution: int = 1) -> List[Tensor]:
    """A function for tracing an arc-segment of a ship's trajectory.
    Toy model which simply assumes that the arc radius depends purely on rudder angle.
    Centers the coordinate frame on the ship, and rotates it so the ships starts
    with a vertical orientation.
    
    Args:
        velocity (Tensor): the ship's velocity for the present timestep.
        rudder_angle (Tensor): the rudder angle relative to the ship's orientation.
        time_step (float): amount of time over which to integrate trajectory.
        record_resolution (float): the time resolution with which the position of ship is recorded.
    """
    assert torch.abs(torch.tensor(rudder_angle)) <= torch.pi/4, "abs(rudder_angle) must be <= pi/4"
    assert time_steps%record_resolution==0, "record_resolution must be a factor of time_steps"
    
    # assume a 100 unit arc_radius at 45 degree rudder
    arc_radius = 100/torch.tan(rudder_angle)
    total_steps = time_steps//record_resolution
    position_deltas = []
    distances_moved = velocity*record_resolution*torch.arange(1,total_steps+1)
    if rudder_angle==0:
        zero_x = torch.zeros_like(distances_moved)
        return torch.stack([zero_x, distances_moved], dim=1)
    else:
        subtended_angles = distances_moved/arc_radius
        new_positions_x = arc_radius*sin(subtended_angles)
        new_positions_y = arc_radius*cos(subtended_angles)
        position_deltas = torch.stack([new_positions_x, new_positions_y],dim=1)

        return position_deltas    
    

    






