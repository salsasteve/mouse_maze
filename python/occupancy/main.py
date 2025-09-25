import numpy as np
import matplotlib.pyplot as plt
import json
import heapq
from math import sqrt
from skimage.draw import line as bresenham_line # Using a library for Bresenham's is cleaner

# --- Core Classes ---

class OccupancyMap:
    """Manages the 2D grid map created from LiDAR data."""
    def __init__(self, resolution=2.0, width=800, height=800, max_range=100.0):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.grid = np.full((height, width), 0.5, dtype=np.float32)  # Unknown
        self.origin_x = width // 2
        self.origin_y = height // 2
        self.max_range = max_range

    def world_to_grid(self, x, y):
        """Converts world coordinates (a single point or arrays) to grid coordinates."""
        grid_x = np.int32(x / self.resolution) + self.origin_x
        grid_y = np.int32(y / self.resolution) + self.origin_y
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Converts grid coordinates to world coordinates."""
        x = (grid_x - self.origin_x) * self.resolution
        y = (grid_y - self.origin_y) * self.resolution
        return x, y
        
    def _is_within_bounds(self, x, y):
        """Checks if grid coordinates are within the map bounds."""
        return (0 <= x) & (x < self.width) & (0 <= y) & (y < self.height)

    def update_with_lidar(self, lidar_data):
            """
            Processes all LiDAR readings to update the occupancy map.
            This version is vectorized for performance.
            """
            print(f"Processing {len(lidar_data)} LiDAR readings...")
            
            robot_positions = [reading['agent_position'] for reading in lidar_data]
            
            for reading in lidar_data:
                robot_x, robot_y = reading['agent_position']
                robot_rot = reading['agent_rotation']
                distances = np.array(reading['distances'])
                
                # --- Vectorized Ray Casting ---
                num_rays = len(distances)
                angle_increment = 2 * np.pi / num_rays
                ray_angles = robot_rot + (np.arange(num_rays) * angle_increment) - np.pi
                valid_mask = distances <= self.max_range
                valid_distances = distances[valid_mask]
                valid_angles = ray_angles[valid_mask]
                end_x = robot_x + valid_distances * np.cos(valid_angles)
                end_y = robot_y + valid_distances * np.sin(valid_angles)

                # --- Update Grid ---
                start_gx, start_gy = self.world_to_grid(robot_x, robot_y)
                end_gx, end_gy = self.world_to_grid(end_x, end_y)
                valid_ends = self._is_within_bounds(end_gx, end_gy)

                # 4. Mark endpoints with thicker walls
                for gx, gy in zip(end_gx[valid_ends], end_gy[valid_ends]):
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = gx + dx, gy + dy
                            if self._is_within_bounds(nx, ny):
                                self.grid[ny, nx] = 1.0 # Mark as Occupied

                # 5. Mark rays as free space
                for ex, ey in zip(end_gx, end_gy):
                    rr, cc = bresenham_line(start_gy, start_gx, ey, ex)
                    free_rr, free_cc = rr[:-1], cc[:-1]
                    valid_line = self._is_within_bounds(free_cc, free_rr)
                    self.grid[free_rr[valid_line], free_cc[valid_line]] = 0.0

            # 6. Ensure the entire robot path is a wide, free corridor
            robot_positions_arr = np.array(robot_positions)
            robot_gx, robot_gy = self.world_to_grid(
                robot_positions_arr[:, 0], robot_positions_arr[:, 1]
            )
            valid_robot_pos = self._is_within_bounds(robot_gx, robot_gy)
            
            print("Carving a safe corridor for the robot's path...")
            for gx, gy in zip(robot_gx[valid_robot_pos], robot_gy[valid_robot_pos]):
                # Clear a 5x5 block around the robot's position
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = gx + dx, gy + dy
                        if self._is_within_bounds(nx, ny):
                            self.grid[ny, nx] = 0.0 # Mark as Free
            
            print("Map update complete.")
            occupied_cells = np.sum(self.grid == 1.0)
            print(f"Sanity Check: Found {occupied_cells} occupied (wall) cells.")

class AStarPathfinder:
    """Finds the shortest path on the occupancy map using A*."""
    def __init__(self, occupancy_map, occupied_threshold=0.7):
        self.map = occupancy_map
        self.occupied_threshold = occupied_threshold

    def _heuristic(self, a, b):
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _get_neighbors(self, node):
        x, y = node
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if self.map._is_within_bounds(nx, ny) and self.map.grid[ny, nx] < self.occupied_threshold:
                    cost = sqrt(dx**2 + dy**2)
                    yield (nx, ny), cost

    def find_path(self, start_world, goal_world):
        start = self.map.world_to_grid(*start_world)
        goal = self.map.world_to_grid(*goal_world)
        
        if not self.map._is_within_bounds(*start) or self.map.grid[start[1], start[0]] >= self.occupied_threshold:
            print(f"Error: Start position {start_world} is invalid or occupied.")
            return None
        if not self.map._is_within_bounds(*goal) or self.map.grid[goal[1], goal[0]] >= self.occupied_threshold:
            print(f"Error: Goal position {goal_world} is invalid or occupied.")
            return None

        print(f"Finding path from grid {start} to {goal}...")
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(self.map.grid_to_world(*current))
                    current = came_from[current]
                # FIX: Add the start position to complete the path
                path.append(self.map.grid_to_world(*start))
                return path[::-1] # Return reversed path

            for neighbor, cost in self._get_neighbors(current):
                tentative_g = g_score.get(current, float('inf')) + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("No path found!")
        return None

# --- Visualization & Main Execution ---

def visualize_results(occupancy_map, path=None, start_pos=None, goal_pos=None):
    """A simplified function to visualize the map and the final path."""
    plt.figure(figsize=(10, 10))
    
    cmap = plt.cm.colors.ListedColormap(['white', 'gray', 'black'])
    bounds = [-0.1, 0.1, 0.6, 1.1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(occupancy_map.grid, cmap=cmap, norm=norm, origin='lower')
    plt.title('Occupancy Map & A* Path')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')

    if path:
        path_grid = [occupancy_map.world_to_grid(x, y) for x, y in path]
        path_x, path_y = zip(*path_grid)
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')

    if start_pos:
        start_grid = occupancy_map.world_to_grid(*start_pos)
        # FIX: Swapped to Red for Start
        plt.plot(start_grid[0], start_grid[1], 'ro', markersize=10, label='Start')
    if goal_pos:
        goal_grid = occupancy_map.world_to_grid(*goal_pos)
        # FIX: Swapped to Green for Goal
        plt.plot(goal_grid[0], goal_grid[1], 'go', markersize=10, label='Goal')

    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

def main():
    """Main function to run the process."""
    try:
        with open("/Users/solz/Documents/rust/mouse_maze/python/occupancy/lidar_data_1758831298.json", 'r') as f:
            lidar_data = json.load(f)
    except FileNotFoundError:
        print("Error: lidar_data.json not found. Please place it in the same directory.")
        return

    # 1. Initialize and build the map
    occupancy_map = OccupancyMap(resolution=2.0, width=800, height=800)
    occupancy_map.update_with_lidar(lidar_data)
    
    # 2. Create the pathfinder
    pathfinder = AStarPathfinder(occupancy_map)
    
    # 3. Define start and goal, then find the path
    # Using positions from the trajectory is a good starting point
    start_pos = tuple(lidar_data[0]['agent_position'])
    goal_pos = tuple(lidar_data[-1]['agent_position'])
    
    path = pathfinder.find_path(start_pos, goal_pos)
    
    # 4. Visualize the final result
    if path:
        print(f"Path found with {len(path)} waypoints.")
        visualize_results(occupancy_map, path, start_pos, goal_pos)

if __name__ == "__main__":
    # It's recommended to install scikit-image for the bresenham line function:
    # pip install scikit-image
    main()