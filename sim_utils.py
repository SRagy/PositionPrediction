import torch
from torch import Tensor, cos, sin
from typing import List, Tuple

def arc_trace(orientation: Tensor, 
              position: Tensor,
              velocity: Tensor, 
              rudder_angle: Tensor, 
              time_steps: int = 10,
              record_resolution: int = 1) -> Tuple[Tensor, Tensor]:
    """A function for tracing an arc-segment of a ship's trajectory.
    Toy model which assumes the ship always travels along circular arcs where
    the arc radius depends purely on rudder angle. When the rudder is straight
    takes the limiting case of travelling in a straight line.
    
    Args:
        orientation (Tensor): the ship's angle relative to coordinate axes. 
        velocity (Tensor): the ship's velocity for the present timestep.
        rudder_angle (Tensor): the rudder angle relative to the ship's orientation.
        trajectory (List[Tensor]): the trajectory so far, as a list of position coordinates.
        time_step (float): amount of time over which to integrate trajectory.
        record_resolution (float): the time resolution with which the position of ship is recorded.

    Returns:
        A tuple containing 
        1. The new points in the ships trajectory, after time_step has elapsed. The number of points
        recorded depends on record_resolution.
        2. The new orientation at the end of the arc of the trajectory.
    """
    position_deltas, orientation_delta = arc_trace_centered(velocity, rudder_angle, time_steps, record_resolution)
    # assume a 100 unit arc_radius at 45 degree rudder
    rotation_matrix = torch.tensor([[cos(orientation), -sin(orientation)],[sin(orientation), cos(orientation)]])
    rotated_position_deltas = rotation_matrix*position_deltas

    new_positions = position + rotated_position_deltas
    new_orientation = orientation + orientation_delta

    # trajectory = torch.cat([trajectory, new_positions])

    return new_positions, new_orientation

def arc_trace_centered(velocity: Tensor, 
              rudder_angle: Tensor, 
              time_steps: int = 10,
              record_resolution: int = 1) -> Tuple[Tensor, Tensor]:
    """A function for tracing an arc-segment of a ship's trajectory.
    Toy model which assumes the ship always travels along circular arcs where
    the arc radius depends purely on rudder angle. When the rudder is straight
    takes the limiting case of travelling in a straight line.

    Assumes a coordinate frame which is centred on the ship, and rotated so that the
    ship is initialised with a vertical orientation.
    
    
    Args:
        velocity (Tensor): the ship's velocity for the present timestep.
        rudder_angle (Tensor): the rudder angle relative to the ship's orientation.
        time_step (float): amount of time over which to integrate trajectory.
        record_resolution (float): the time resolution with which the position of ship is recorded.

    Returns:
        A tuple containing 
        1. The new points in the ships trajectory.
        2. The new orientation.
    """
    assert torch.abs(torch.tensor(rudder_angle)) <= torch.pi/5, "abs(rudder_angle) must be <= pi/5"
    assert time_steps%record_resolution==0, "record_resolution must be a factor of time_steps"
    
    # assume a 100 unit turn_radius at 45 degree rudder
    turn_radius = 100/torch.tan(rudder_angle)
    total_steps = time_steps//record_resolution
    position_deltas = []
    distances_moved = velocity*record_resolution*torch.arange(1,total_steps+1)
    if rudder_angle==0:
        zero_x = torch.zeros_like(distances_moved)
        return torch.stack([zero_x, distances_moved], dim=1)
    else:
        subtended_angles = distances_moved/turn_radius
        new_positions_x = turn_radius*cos(subtended_angles)-turn_radius
        new_positions_y = turn_radius*sin(subtended_angles)
        position_deltas = torch.stack([new_positions_x, new_positions_y],dim=1)

        return position_deltas, subtended_angles[-1]