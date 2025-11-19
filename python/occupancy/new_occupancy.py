import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import cv2  # Used for OpenCV drawing functions
from pathfinding.core.grid import Grid as PFGrid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement

# --- Core Classes ---


class OccupancyMap:
    """Manages the 2D grid map created from LiDAR data using OpenCV."""

    def __init__(self, resolution=2.0, width=800, height=800, max_range=100.0):
        self.resolution = resolution
        self.width = width
        self.height = height
        # 0 = Free, 127 = Unknown, 255 = Occupied
        self.grid = np.full((height, width), 127, dtype=np.uint8)
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

    def _is_within_bounds_vec(self, x, y):
        """Vectorized check if grid coordinates are within the map bounds."""
        return (0 <= x) & (x < self.width) & (0 <= y) & (y < self.height)

    def update_with_lidar(self, lidar_df: pl.DataFrame):
        """
        Processes all LiDAR readings from a Polars DataFrame to update the map
        using vectorized numpy and OpenCV drawing functions.
        """
        print(f"Processing {len(lidar_df)} LiDAR readings...")

        # Correctly get (N, 2) array of robot positions
        robot_positions = np.array(lidar_df["agent_position"].to_list())

        for reading in lidar_df.iter_rows(named=True):
            robot_x, robot_y = reading["agent_position"]
            robot_rot = reading["agent_rotation"]
            distances = np.array(reading["distances"])

            # --- Vectorized Ray Casting ---
            num_rays = len(distances)
            angle_increment = 2 * np.pi / num_rays
            ray_angles = robot_rot + (np.arange(num_rays) * angle_increment) - np.pi
            valid_mask = distances <= self.max_range
            valid_distances = distances[valid_mask]
            valid_angles = ray_angles[valid_mask]
            end_x = robot_x + valid_distances * np.cos(valid_angles)
            end_y = robot_y + valid_distances * np.sin(valid_angles)

            # --- Simplified Grid Update using OpenCV ---
            start_gx, start_gy = self.world_to_grid(robot_x, robot_y)
            end_gx_vec, end_gy_vec = self.world_to_grid(end_x, end_y)
            valid_ends_mask = self._is_within_bounds_vec(end_gx_vec, end_gy_vec)

            # 1. Mark rays as free space (more concise)
            start_point = (start_gx, start_gy)
            for ex, ey in zip(end_gx_vec, end_gy_vec):
                cv2.line(
                    self.grid, start_point, (ex, ey), color=0, thickness=1
                )  # 0 = Free

            # 2. Mark endpoints (more concise)
            for gx, gy in zip(end_gx_vec[valid_ends_mask], end_gy_vec[valid_ends_mask]):
                cv2.circle(
                    self.grid, (gx, gy), radius=1, color=255, thickness=-1
                )  # 255 = Occupied

        # 3. Ensure the entire robot path is a wide, free corridor
        print("Carving a safe corridor for the robot's path...")
        robot_gx, robot_gy = self.world_to_grid(
            robot_positions[:, 0], robot_positions[:, 1]
        )
        valid_robot_pos = self._is_within_bounds_vec(robot_gx, robot_gy)

        # Use cv2.circle to clear a radius around each robot position
        for gx, gy in zip(robot_gx[valid_robot_pos], robot_gy[valid_robot_pos]):
            cv2.circle(self.grid, (gx, gy), radius=2, color=0, thickness=-1)  # 0 = Free

        print("Map update complete.")
        occupied_cells = np.sum(self.grid == 255)
        print(f"Sanity Check: Found {occupied_cells} occupied (wall) cells.")


# --- Pathfinding Function (Replaces AStarPathfinder class) ---


def find_astar_path(occupancy_map, start_world, goal_world, occupied_threshold=200):
    """
    Finds the shortest path on the occupancy map using the 'python-pathfinding' library.
    """
    # 1. Create walkability grid from the map
    # 1 = Walkable (free or unknown), 0 = Blocked (occupied)
    walkable_matrix = (occupancy_map.grid < occupied_threshold).astype(int)
    pf_grid = PFGrid(matrix=walkable_matrix)

    # 2. Convert world to grid coordinates
    start_gx, start_gy = occupancy_map.world_to_grid(*start_world)
    goal_gx, goal_gy = occupancy_map.world_to_grid(*goal_world)

    # 3. Check if start/goal are valid
    if not pf_grid.node(start_gx, start_gy).walkable:
        print(
            f"Error: Start position {start_world} at grid ({start_gx}, {start_gy}) is blocked."
        )
        return None
    if not pf_grid.node(goal_gx, goal_gy).walkable:
        print(
            f"Error: Goal position {goal_world} at grid ({goal_gx}, {goal_gy}) is blocked."
        )
        return None

    print(
        f"Finding path from grid ({start_gx}, {start_gy}) to ({goal_gx}, {goal_gy})..."
    )

    # 4. Create nodes and finder
    start_node = pf_grid.node(start_gx, start_gy)
    goal_node = pf_grid.node(goal_gx, goal_gy)
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

    # 5. Run the finder
    path_nodes, runs = finder.find_path(start_node, goal_node, pf_grid)

    # 6. Convert path back to world coordinates
    if not path_nodes:
        print("No path found!")
        return None

    # Use a list comprehension for a concise conversion
    return [occupancy_map.grid_to_world(x, y) for x, y in path_nodes]


# --- Visualization & Main Execution ---


def visualize_results(occupancy_map, path=None, start_pos=None, goal_pos=None):
    """Visualize the map and path, updated for the 0-255 grid scale."""
    plt.figure(figsize=(10, 10))

    cmap = plt.cm.colors.ListedColormap(["white", "gray", "black"])
    bounds = [-1, 64, 192, 256]  # Bins around 0, 127, 255
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(occupancy_map.grid, cmap=cmap, norm=norm, origin="lower")
    plt.title("Occupancy Map & A* Path (OpenCV + Pathfinding Lib)")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")

    if path:
        path_grid = [occupancy_map.world_to_grid(x, y) for x, y in path]
        path_x, path_y = zip(*path_grid)
        plt.plot(path_x, path_y, "b-", linewidth=2, label="Path")

    if start_pos:
        start_grid = occupancy_map.world_to_grid(*start_pos)
        plt.plot(
            start_grid[0], start_grid[1], "ro", markersize=10, label="Start"
        )  # Red
    if goal_pos:
        goal_grid = occupancy_map.world_to_grid(*goal_pos)
        plt.plot(goal_grid[0], goal_grid[1], "go", markersize=10, label="Goal")  # Green

    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


def main():
    """Main function to run the process."""
    try:
        file_path = "/Users/solz/Documents/rust/mouse_maze/output_data/lidar/lidar_data_1761096364.json"
        lidar_df = pl.read_json(file_path)
    except Exception as e:
        print(f"Error: Could not load '{file_path}'. {e}")
        return

    # 1. Initialize and build the map
    occupancy_map = OccupancyMap(resolution=2.0, width=800, height=800)
    occupancy_map.update_with_lidar(lidar_df)

    # 2. Define start and goal from the Polars DataFrame
    start_pos = tuple(lidar_df.item(0, "agent_position"))
    goal_pos = tuple(lidar_df.item(-1, "agent_position"))

    # 3. Find the path using the new function
    path = find_astar_path(occupancy_map, start_pos, goal_pos)

    # 4. Visualize the final result
    if path:
        print(f"Path found with {len(path)} waypoints.")
    visualize_results(occupancy_map, path, start_pos, goal_pos)


if __name__ == "__main__":
    # You will need to install the new libraries:
    # pip install polars opencv-python python-pathfinding scikit-image matplotlib
    main()
