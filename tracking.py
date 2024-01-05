import torch
from torch import Tensor
from nflows.flows import Flow
import warnings




class Tracker:
    """Tracker class. Given a trained density estimator, searches for object. Assumes scans in small
    square regions with size determined by scan_radius. The area_of_interest refers to the full search
    region, not just the immediate scan area. If initialised with a set of points to check, then will
    scan around these points. Otherwise, it is recommended to instead initialise a sampling tracker or
    grid tracker.

    Attributes:
        density_estimator: A normalising flow.
        scan_radius: Half the length of the square in which to scan.
        area_of_interest (Tensor, optional): Area in which to search - points outside are ignored.
        points_to_check (Tensor, optional): Points around which to check
    Methods:
        simulate_tracking: Given the true location, simulate how a tracker would behave.

    """
    def __init__(self, density_estimator: Flow,
                 scan_radius: int,
                 area_of_interest: Tensor = None,
                 points_to_check: Tensor = None):
        """Initialises Base Tracker class

        Args:
            density_estimator (Flow): A density estimator for trajectory end points
            scan_radius (int): 'radius' of scanning area; half side-length of square scanning region.
            area_of_interest (Tensor, optional): Area to search in. Defaults to None.
            points_to_check (Tensor, optional): Points to check. Defaults to None.
        """
        self.density_estimator = density_estimator
        self.scan_radius = scan_radius
        self.area_of_interest = area_of_interest
        self.points_to_check = points_to_check


    def _rank_points(self, points: Tensor):
        """Ranks points by likelihood. Most likely first.

        Args:
            points (Tensor): Batch of n points to rank.

        Returns:
            Tensor: Batch of n points ranked by likelihood.
        """
        densities = self.density_estimator.log_prob(points)
        density_rank_index = densities.sort().indices

        return points[density_rank_index]
    
    def _points_in_area(self, area: Tensor, points: Tensor):
        """ Given an area and one or more points, returns any points which land
        in the area.

        Args:
            area (Tensor): Tensor defining a square's boundaries: [[left, right], [bottom, top]].
            points (Tensor): Either an m-dimensional point or an nxm batch of points.
        Returns:
            Tensor: The set of points which fall in the given area.
        """

        # to handle case of a single point
        is_single_point = len(points.shape) == 1
        if is_single_point:
            mask_x = area[0,0] < points[0] < area[0,1]
            mask_y = area[1,0] < points[1] < area[1,1]
        else:
            mask_x = torch.logical_and(area[0,0] < points[:,0], points[:,0] < area[0,1])
            mask_y = torch.logical_and(area[1,0] < points[:,1], points[:,1] < area[1,1])

        mask = torch.logical_and(mask_x, mask_y)

        if not is_single_point and not all(mask):
            warnings.warn("not all points to check are in area of interest, some will be ignored")
       
        return points[mask]

    def _generate_scan_area(self, point: Tensor):
        """Generates an area of self.scan_radius centred on the given point 

        Args:
            point (Tensor): A point about which to scan.

        Returns:
            Tensor: A square's boundaries, [[left, right], [bottom, top]]
        """
        left_and_bottom = point - self.scan_radius
        right_and_top = point + self.scan_radius

        return torch.stack([left_and_bottom, right_and_top], dim=1)
    

    def simulate_tracking(self, true_point: Tensor):
        assert self.points_to_check is not None, "Assign points to check"

        if self.area_of_interest is None:
            points = self.points_to_check
        else:
            points = self._points_in_area(self.area_of_interest, self.points_to_check)

        sorted_points = self._rank_points(points)
        for i in range(len(sorted_points)):
            area = self._generate_scan_area(sorted_points[i])
            if self._points_in_area(area, true_point).nelement() > 0:
                # break if returned tensor isn't empty, i.e. match found.
                break
        else: # if vessel is not found at any of the points to check
            return sorted_points, False


        return sorted_points[:i+1], True

                    
class SamplingTracker(Tracker):
    """Tracker which checks in region around sampled points. May be useful for cases where
    the path is quite predictable and we don't want to evaluate a whole grid of log_probs.

    Attributes:
        density_estimator: A normalising flow.
        scan_radius: Half the length of the square in which to scan.
        area_of_interest (Tensor): Area in which to search - points outside here are ignored.
        points_to_check (Tensor): Points around which to check
    Methods:
        simulate_tracking: Given the true location, simulate how a tracker would behave.


    """
    def __init__(self, density_estimator: Flow, 
                 scan_radius: int, 
                 area_of_interest: Tensor = None, 
                 num_samples: int = 10000):
        """Init function for sampling tracker

        Args:
            density_estimator (Flow): A density estimator for trajectory end points
            scan_radius (int): 'radius' of scanning area; half side-length of square scanning region.
            area_of_interest (Tensor, optional): are in which to search. Defaults to None.
            num_samples (int): the number of points to sample for checking. Defaults to 1000.
        """
        points_to_check = density_estimator.sample(num_samples).detach()
        return super().__init__(density_estimator, scan_radius, area_of_interest, points_to_check)

class GridTracker(Tracker):
    """Tracker which establishes a grid of points in the area of interest, and scans around these.
    If an area_of_interest is given, then the grid is generated within this area, with grid-size
    matching scan_radius as closely as possible (subject to the constraint of full coverage).

    If no area_of_interest is given, then one will be defined as multiple of the minimum bounding
    box of all points in points_to_check. If neither is given, points will be generated from the
    density estimator.

    Attributes:
        density_estimator: A normalising flow.
        scan_radius: Half the length of the square in which to scan.
        area_of_interest (Tensor): Area in which to search - points outside here are ignored.
        points_to_check (Tensor): Points around which to check
    Methods:
        simulate_tracking: Given the true location, simulate how tracker would behave.
    """
    def __init__(self, density_estimator: Flow,
                 scan_radius: int,
                 area_of_interest: Tensor = None,
                 points_to_check: Tensor = None):
        """Initialises Tracker class

        Args:
            density_estimator (Flow): A density estimator for trajectory end points
            scan_radius (int): 
            area_of_interest (Tensor, optional): area_of_interest. Defaults to None.
            points_to_check (Tensor, optional): Points around which we'd like to scan. Defaults to None.
        """
        if area_of_interest is None:
            if points_to_check is None:
                points_to_check = density_estimator.sample(10000).detach()
                area_of_interest = self._generate_AOI_from_points(points_to_check)
            else:
                area_of_interest = self._generate_AOI_from_points(points_to_check)

        new_points_to_check = self._generate_points_from_grid_centres(scan_radius, area_of_interest)
        super().__init__(density_estimator, scan_radius, area_of_interest, new_points_to_check)
    
    
    def _generate_points_from_grid_centres(self, scan_radius = None, area_of_interest: Tensor = None):
        """Generates a grid, defined by the points at the centre of the grid squares.
        The grid squares are chosen to be approximately the of side-length 2*self.scan_radius

        Args:
            area_of_interest (Tensor, optional): Area in which to draw the grid.
        """
        
        if area_of_interest is None:
            area_of_interest = self.area_of_interest

        if scan_radius is None:
            scan_radius = self.scan_radius


        left, right = area_of_interest[0]
        top, bottom = area_of_interest[1]

        # The minimum number of steps to cross the whole space
        horiz_steps = torch.ceil((right-left)/(2*scan_radius))
        vert_steps= torch.ceil((top-bottom)/(2*scan_radius))

        # Adjusted step lengths, to ensure we can fit the final step in each direction.
        # Will be <= scan_radius (there are probably better solutions here). 
        h_step_length = (right-left)/horiz_steps
        v_step_length = (right-left)/vert_steps

        horiz_start = left + h_step_length #centre of leftmost column
        vert_start = bottom + v_step_length #centre of bottom row

        horizontal_centres = torch.arange(horiz_start, right, h_step_length)
        vertical_centres = torch.arange(vert_start, top, v_step_length)

        return torch.cartesian_prod(horizontal_centres, vertical_centres)

    def _generate_AOI_from_points(self, points: Tensor, mult_factor: float = 1.05):
        """Generates area of interest as function of the minimum enclosing box of the points.
        The multiplication factor grows or shrinks the minimum enclosing box.

        Args:
            points (Tensor): The points to be enclosed by the bounding box
            mult_factor (float, optional): Multiplies size of bounding box. Defaults to 1.
        Returns:
            Tensor: _description_
        """
        axes_max = points.max(dim=0).values
        axes_min = points.min(dim=0).values


        mid_point = (axes_max + axes_min)/2
        dist_from_mid = mult_factor* (axes_max - axes_min)/2

        right_and_top = mid_point+dist_from_mid
        left_and_bottom = mid_point - dist_from_mid

        return torch.stack([left_and_bottom, right_and_top], dim=1)


# class HybridTracker(GridTracker, SamplingTracker):
#     raise NotImplementedError