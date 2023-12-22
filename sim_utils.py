import torch
from torch import Tensor, cos, sin
from typing import List, Tuple

def arc_trace_centered(speed: Tensor, 
              rudder_angle: Tensor, 
              time_steps: int = 10,
              record_resolution: int = 1) -> Tuple[Tensor, Tensor]:
    """A function for tracing an arc-segment of a ship's trajectory.
    Assumes a coordinate frame which is initially centred on the ship, and rotated so that the
    ship is starts with a vertical orientation.

    Toy model which assumes the ship always travels along circular arcs where
    the arc radius depends purely on rudder angle. When the rudder is straight
    takes the limiting case of travelling in a straight line.

    Args:
        speed (Tensor): the ship's speed for the present timestep. Negative values indicate reversing.
        rudder_angle (Tensor): the rudder angle relative to the ship's orientation.
        time_step (float): amount of time over which to integrate trajectory.
        record_resolution (float): the time resolution with which the position of ship is recorded.

    Returns:
        A tuple containing 
        1. The n new points in the ships trajectory, in an nx2 tensor.
        2. The new orientation as a scalar torch.tensor.
    """
    assert torch.abs(rudder_angle) <= torch.pi/5, "abs(rudder_angle) must be <= pi/5"
    assert time_steps%record_resolution==0, "record_resolution must be a factor of time_steps"
    
    # assume a 100 unit turn_radius at 45 degree rudder
    turn_radius = 100/torch.tan(rudder_angle)
    total_steps = time_steps//record_resolution
    position_deltas = []
    distances_moved = speed*record_resolution*torch.arange(1,total_steps+1)
    if rudder_angle==0:
        zero_x = torch.zeros_like(distances_moved)
        return torch.stack([zero_x, distances_moved], dim=1), torch.tensor(0.)
    else:
        subtended_angles = distances_moved/turn_radius
        new_positions_x = turn_radius*cos(subtended_angles)-turn_radius
        new_positions_y = turn_radius*sin(subtended_angles)
        position_deltas = torch.stack([new_positions_x, new_positions_y],dim=1)

        return position_deltas, subtended_angles[-1]
    
def translate_and_rotate_arc(init_position, init_orientation, position_deltas, orientation_delta):
    """Given an arc in space, 

    Args:
        init_position (_type_): _description_
        init_orientation (_type_): _description_
        position_deltas (_type_): _description_
        orientation_delta (_type_): _description_

    Returns:
        _type_: _description_
    """
    rotation_matrix = torch.tensor([[cos(init_orientation), -sin(init_orientation)],
                                    [sin(init_orientation), cos(init_orientation)]])
    
    # Multiply each row vector by the rotation matrix so e.g. pos[0] -> rot@pos[0]
    rotated_position_deltas = torch.einsum('ij,kj -> ki', rotation_matrix, position_deltas)

    new_positions = init_position + rotated_position_deltas
    new_orientation = init_orientation + orientation_delta

    return new_positions, new_orientation


def arc_trace(orientation: Tensor, 
              position: Tensor,
              speed: Tensor, 
              rudder_angle: Tensor, 
              time_steps: int = 10,
              record_resolution: int = 1) -> Tuple[Tensor, Tensor]:
    """
    Generalisation of arc_trace_centered to account for uncentered and rotated coordinate axes.

    Traces an arc-segment of a ship's trajectory.
    Toy model which assumes the ship always travels along circular arcs where
    the arc radius depends purely on rudder angle. When the rudder is straight
    takes the limiting case of travelling in a straight line. 
    
    Args:
        orientation (Tensor): the ship's angle relative to coordinate axes. 
        speed (Tensor): the ship's speed for the present timestep.
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
    position_deltas, orientation_delta = arc_trace_centered(speed, rudder_angle, 
                                                            time_steps, record_resolution)

    new_positions, new_orientation = translate_and_rotate_arc(position, orientation, 
                                                              position_deltas, orientation_delta)

    return new_positions, new_orientation
