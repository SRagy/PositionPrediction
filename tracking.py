import torch
from torch import Tensor
from nflows.flows import Flow
import warnings




class Tracker:
    """Tracker class. Given a trained density estimator, searches for object. Assumes scans in small
    square regions with size determined by scan_radius. The area_of_interest refers to the full search
    region, not just the immediate scan area. If initialised with a set of points to check, then will
    scan around these points. Otherwise, it is recommended to initialise as a sampling tracker or
    grid tracker. The grid tracker provides full coverage of the area of interest.

    Attributes:
        density_estimator: A normalising flow.
        scan_radius: Half the length of the square in which to scan.
        area_of_interest (Tensor, optional): Area in which to search - points outside here are ignored.
        points_to_check (Tensor, optional): Points around which to check
    Methods:
        init_sampling_tracker: generates points from sampling density estimator, then inits with these
        init_grid_tracker: generates points on a grid within the area of interest, then inits with these
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

    @classmethod
    def init_sampling_tracker(cls, 
                              density_estimator: Flow, 
                              scan_radius:int, 
                              area_of_interest: Tensor = None, 
                              num_samples: int = 1000):
        """Initialises tracker where sampling 

        Args:
            density_estimator (Flow): _description_
            scan_radius (int): _description_
            area_of_interest (Tensor, optional): _description_. Defaults to None.
            num_samples (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        points_to_check = density_estimator.sample(num_samples)
        return cls(density_estimator, scan_radius, area_of_interest, points_to_check)

    @classmethod
    def init_grid_tracker(cls, 
                          density_estimator: Flow, 
                          scan_radius:int, 
                          area_of_interest: Tensor = None, 
                          points_to_check: Tensor = None):
        """_summary_

        Args:
            density_estimator (Flow): _description_
            scan_radius (int): _description_
            area_of_interest (Tensor, optional): _description_. Defaults to None.
            points_to_check (Tensor, optional): _description_. Defaults to None.
        """
        

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
            mask_x = area[0,0] < points[:,0] < area[0,1]
            mask_y = area[1,0] < points[:,1] < area[1,1]

        mask = mask_x and mask_y

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
    

    def _generate_AOI_from_points(self, points: Tensor, mult_factor: float = 1.3):
        """Generates area of interest as function of the minimum enclosing box of the points.
        The multiplication factor grows or shrinks the minimum enclosing box.

        Args:
            points (Tensor): The points to be enclosed by the bounding box
            mult_factor (float, optional): Multiplies size of bounding box. Defaults to 1.
        Returns:
            Tensor: _description_
        """
        axes_max = points.max(dim=0)
        axes_min = points.min(dim=0)


        mid_point = (axes_max + axes_min)/2
        dist_from_mid = mult_factor* (axes_max - axes_min)/2

        right_and_top = mid_point+dist_from_mid
        left_and_bottom = mid_point - dist_from_mid

        return torch.stack([left_and_bottom, right_and_top], dim=1)
    
    def _generate_points_from_grid_centres(self, area_of_interest: Tensor = None):
        """Generates a grid, defined by the points at the centre of the grid squares.
        The grid squares are 

        Args:
            area_of_interest (Tensor, optional): Area in which to draw the grid. Defaults to None.
        """
        
        if area_of_interest is None:
            area_of_interest = self.area_of_interest

        left, right = area_of_interest[0]
        top, bottom = area_of_interest[1]

        # The minimum number of steps to cross the whole space
        horiz_steps = torch.ceil((right-left)/(2*self.scan_radius))
        vert_steps= torch.ceil((top-bottom)/(2*self.scan_radius))

        # Adjusted step lengths, to ensure we can fit the final step in each direction.
        # Will be <= scan_radius (there are probably better solutions). 
        h_step_length = (right-left)/horiz_steps
        v_step_length = (right-left)/vert_steps

        horiz_start = left + h_step_length #centre of leftmost column
        vert_start = bottom + v_step_length #centre of bottom row

        horizontal_centres = torch.arange(horiz_start, right, h_step_length)
        vertical_centres = torch.arange(vert_start, top, v_step_length)

        self.h_step_length = h_step_length
        self.v_step_length = v_step_length
        self.grid = torch.cartesian_prod(horizontal_centres, vertical_centres)


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
                 num_samples: int = 1000):
        """Initialises Tracker class

        Args:
            density_estimator (Flow): A density estimator for trajectory end points
            scan_radius (int): 'radius' of scanning area; half side-length of square scanning region.
            area_of_interest (Tensor, optional): area_of_interest. Defaults to None.
            num_samples (int): _description_. Defaults to 1000.
        """
        super().init_sampling_tracker(density_estimator, scan_radius, area_of_interest, num_samples)

class GridTracker(Tracker):
    def __init__(self, density_estimator: Flow,
                 scan_radius: int,
                 area_of_interest: Tensor,
                 points_to_check: Tensor):
        """Initialises Tracker class

        Args:
            density_estimator (Flow): A density estimator for trajectory end points
            scan_radius (int): 
            area_of_interest (Tensor, optional): area_of_interest. Defaults to None.
            points_to_check (Tensor, optional): _description_. Defaults to None.
        """
        super().__init__(density_estimator, scan_radius, area_of_interest)
        self._generate_grid_centres()
    
    def _generate_AOI_from_points(self, points: Tensor, mult_factor: float = 1.3):
        """_summary_

        Args:
            points (Tensor): The points to be enclosed by the bounding box
            mult_factor (float, optional): Multiplies size of minimal bounding box. Defaults to 1.
        Returns:
            Tensor: _description_
        """
        axes_max = points.max(dim=0)
        axes_min = points.min(dim=0)


        mid_point = (axes_max + axes_min)/2
        dist_from_mid = mult_factor* (axes_max - axes_min)/2

        right_and_top = mid_point+dist_from_mid
        left_and_bottom = mid_point - dist_from_mid

        return torch.stack([left_and_bottom, right_and_top], dim=1)
    
    def _generate_grid_centres(self, 
                               area_of_interest: Tensor = None):
        
        if area_of_interest is None:
            area_of_interest = self.area_of_interest

        left, right = area_of_interest[0]
        top, bottom = area_of_interest[1]

        # The minimum number of steps to cross the whole space
        horiz_steps = torch.ceil((right-left)/(2*self.scan_radius))
        vert_steps= torch.ceil((top-bottom)/(2*self.scan_radius))

        # Adjusted step lengths, to ensure we can fit the final step in each direction.
        # Will be <= scan_radius (there are probably better solutions). 
        h_step_length = (right-left)/horiz_steps
        v_step_length = (right-left)/vert_steps

        horiz_start = left + h_step_length #centre of leftmost column
        vert_start = bottom + v_step_length.scan_radius #centre of bottom row

        horizontal_centres = torch.arange(horiz_start, right, h_step_length)
        vertical_centres = torch.arange(vert_start, top, v_step_length)

        self.h_step_length = h_step_length
        self.v_step_length = v_step_length
        self.grid = torch.cartesian_prod(horizontal_centres, vertical_centres)

        
    def track(self, type = 'sampled', ):
        pass

# class HybridTracker(GridTracker, SamplingTracker):
#     raise NotImplementedError