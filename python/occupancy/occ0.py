import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import cv2
from pathfinding.core.grid import Grid as PFGrid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
from matplotlib.animation import FuncAnimation, FFMpegWriter

ANIMATION_PATH = "/Users/solz/Documents/rust/mouse_maze/report/debug_path_finding/lidar_scan_animation.mp4"
DEFAULT_OCCUPANCY_WIDTH = 200
DEFAULT_OCCUPANCY_HEIGHT = 200


class OccupancyMap:
    """Manages the 2D grid map created from LiDAR data using OpenCV."""

    def __init__(
        self,
        resolution=2.0,
        width=DEFAULT_OCCUPANCY_WIDTH,
        height=DEFAULT_OCCUPANCY_HEIGHT,
        max_range=100.0,
    ):
        self.resolution = resolution
        self.width = width
        self.height = height
        # 0 = Free, 127 = Unknown, 255 = Occupied
        self.grid = np.full((height, width), 127, dtype=np.uint8)
        self.origin_x = width // 2
        self.origin_y = height // 2
        # Use a slightly smaller max_range internally for reliable < checks
        self.max_range = max_range
        self._max_range_threshold = max_range - 0.1

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

        # Store robot positions for corridor carving
        robot_positions = np.array(lidar_df["agent_position"].to_list())

        for reading in lidar_df.iter_rows(named=True):
            robot_x, robot_y = reading["agent_position"]
            distances = np.array(reading["distances"])
            ray_angles = np.array(reading["angles"])

            valid_mask = distances <= self.max_range
            valid_distances = distances[valid_mask]
            valid_angles = ray_angles[valid_mask]

            end_x = robot_x + valid_distances * np.cos(valid_angles)
            end_y = robot_y + valid_distances * np.sin(valid_angles)

            start_gx, start_gy = self.world_to_grid(robot_x, robot_y)
            end_gx_vec, end_gy_vec = self.world_to_grid(end_x, end_y)

            # 1. Mark rays as free space NON-DESTRUCTIVELY
            start_point = (start_gx, start_gy)
            temp_free_mask = np.zeros(self.grid.shape, dtype=np.uint8)
            
            # Draw all free lines onto the temp mask
            for ex, ey in zip(end_gx_vec, end_gy_vec):
                cv2.line(temp_free_mask, start_point, (ex, ey), color=1, thickness=1)
            
            # Apply "free" mask only to unknown cells
            unknown_cells = (self.grid == 127)
            free_mask = (temp_free_mask == 1)
            self.grid[unknown_cells & free_mask] = 0  # Set to FREE

            # 2. Mark obstacles (ONLY FOR *HITS*)
            hit_mask = valid_distances < self._max_range_threshold
            hit_gx_vec = end_gx_vec[hit_mask]
            hit_gy_vec = end_gy_vec[hit_mask]
            valid_hits_mask = self._is_within_bounds_vec(hit_gx_vec, hit_gy_vec)

            for gx, gy in zip(hit_gx_vec[valid_hits_mask], hit_gy_vec[valid_hits_mask]):
                cv2.circle(
                    self.grid, (gx, gy), radius=1, color=255, thickness=-1
                )  # 255 = Occupied

        # 3. Carve safe corridor along robot's path
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
        free_cells = np.sum(self.grid == 0)
        unknown_cells = np.sum(self.grid == 127)
        print(f"Occupied: {occupied_cells}, Free: {free_cells}, Unknown: {unknown_cells}")


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
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

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


def get_lidar_data(file_path):
    """Loads LiDAR data from a JSON file into a Polars DataFrame."""
    lidar_df = pl.read_json(file_path)
    print(f"Loaded LiDAR data with {len(lidar_df)} readings.")
    return lidar_df


def get_start_goal_positions(lidar_df):
    """Extracts start and goal positions from the LiDAR DataFrame."""
    start_pos = tuple(lidar_df.item(0, "agent_position"))
    goal_pos = tuple(lidar_df.item(-1, "agent_position"))
    return start_pos, goal_pos


def build_occupancy_map(lidar_df):
    """Builds the occupancy map from LiDAR data."""
    occupancy_map = OccupancyMap()
    occupancy_map.update_with_lidar(lidar_df)
    return occupancy_map


def animate_lidar_scan(
    lidar_df,
    resolution=2.0,
    width=DEFAULT_OCCUPANCY_WIDTH,
    height=DEFAULT_OCCUPANCY_HEIGHT,
    save_path=None,
    fps=10,
    frame_skip=1,
):
    """
    Creates an animated visualization of LiDAR scanning process.

    Args:
        lidar_df: Polars DataFrame with LiDAR readings
        resolution: Grid resolution
        width, height: Grid dimensions
        save_path: If provided, saves video to this path (e.g., 'lidar_scan.mp4')
        fps: Frames per second for the video
        frame_skip: Only animate every Nth reading (1 = all readings)
    """
    # Initialize map
    occupancy_map = OccupancyMap(resolution=resolution, width=width, height=height)

    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Color map setup
    cmap = plt.cm.colors.ListedColormap(["white", "gray", "black"])
    bounds = [-1, 64, 192, 256]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Initial plot
    im = ax1.imshow(occupancy_map.grid, cmap=cmap, norm=norm, origin="lower")
    ax1.set_title("Occupancy Map (Building...)")
    ax1.set_xlabel("Grid X")
    ax1.set_ylabel("Grid Y")

    # For ray visualization
    ax2.set_xlim(-width // 2 * resolution, width // 2 * resolution)
    ax2.set_ylim(-height // 2 * resolution, height // 2 * resolution)
    ax2.set_title("Current LiDAR Scan")
    ax2.set_xlabel("World X")
    ax2.set_ylabel("World Y")
    ax2.grid(True, alpha=0.3)

    # Stats text
    stats_text = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Select readings to animate
    total_readings = len(lidar_df)
    readings_to_animate = list(range(0, total_readings, frame_skip))
    robot_positions = np.array(lidar_df["agent_position"].to_list())

    def update(frame_idx):
        reading_idx = readings_to_animate[frame_idx]
        reading = lidar_df.row(reading_idx, named=True)

        # Process single reading
        robot_x, robot_y = reading["agent_position"]
        distances = np.array(reading["distances"])

        # Update occupancy map
        ray_angles = np.array(reading["angles"])

        valid_mask = distances <= occupancy_map.max_range
        valid_distances = distances[valid_mask]
        valid_angles = ray_angles[valid_mask]

        end_x = robot_x + valid_distances * np.cos(valid_angles)
        end_y = robot_y + valid_distances * np.sin(valid_angles)

        start_gx, start_gy = occupancy_map.world_to_grid(robot_x, robot_y)
        end_gx_vec, end_gy_vec = occupancy_map.world_to_grid(end_x, end_y)

        # Draw free space
        start_point = (start_gx, start_gy)
        temp_free_mask = np.zeros(occupancy_map.grid.shape, dtype=np.uint8)

        # 2. Draw all free lines onto the temp mask
        for ex, ey in zip(end_gx_vec, end_gy_vec):
            cv2.line(temp_free_mask, start_point, (ex, ey), color=1, thickness=1)

        # 3. Apply this "free" mask NON-DESTRUCTIVELY
        unknown_cells = (occupancy_map.grid == 127)
        free_mask = (temp_free_mask == 1)
        occupancy_map.grid[unknown_cells & free_mask] = 0 # Set to FREE

        # 4. Mark obstacles (this IS destructive, which is correct)
        hit_mask = valid_distances < occupancy_map._max_range_threshold
        hit_gx_vec = end_gx_vec[hit_mask]
        hit_gy_vec = end_gy_vec[hit_mask]
        valid_hits_mask = occupancy_map._is_within_bounds_vec(hit_gx_vec, hit_gy_vec)

        for gx, gy in zip(hit_gx_vec[valid_hits_mask], hit_gy_vec[valid_hits_mask]):
            cv2.circle(occupancy_map.grid, (gx, gy), radius=1, color=255, thickness=-1)

        positions_so_far = robot_positions[:reading_idx + 1]
        robot_gx, robot_gy = occupancy_map.world_to_grid(
            positions_so_far[:, 0], positions_so_far[:, 1]
        )
        valid_robot_pos = occupancy_map._is_within_bounds_vec(robot_gx, robot_gy)
        
        for gx, gy in zip(robot_gx[valid_robot_pos], robot_gy[valid_robot_pos]):
            cv2.circle(occupancy_map.grid, (gx, gy), radius=2, color=0, thickness=-1)

        # Update map display
        im.set_array(occupancy_map.grid)

        # Clear and redraw current scan
        ax2.clear()
        ax2.set_xlim(-width // 2 * resolution, width // 2 * resolution)
        ax2.set_ylim(-height // 2 * resolution, height // 2 * resolution)
        ax2.set_title(f"LiDAR Scan #{reading_idx}")
        ax2.set_xlabel("World X")
        ax2.set_ylabel("World Y")
        ax2.grid(True, alpha=0.3)

        # Plot robot position
        ax2.plot(robot_x, robot_y, "ro", markersize=8, label="Robot")

        # Plot rays
        for angle, dist in zip(valid_angles, valid_distances):
            ex = robot_x + dist * np.cos(angle)
            ey = robot_y + dist * np.sin(angle)
            color = "red" if dist < occupancy_map._max_range_threshold else "blue"
            ax2.plot(
                [robot_x, ex], [robot_y, ey], color=color, alpha=0.3, linewidth=0.5
            )

        # Update stats
        occupied = np.sum(occupancy_map.grid == 255)
        free = np.sum(occupancy_map.grid == 0)
        stats_text.set_text(
            f"Reading: {reading_idx}/{total_readings}\n"
            f"Occupied cells: {occupied}\n"
            f"Free cells: {free}"
        )

        return [im, stats_text]

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(readings_to_animate),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    # Save or show
    if save_path:
        print(f"Saving animation to {save_path}...")
        writer = FFMpegWriter(
            fps=fps, metadata=dict(artist="LiDAR Scanner"), bitrate=1800
        )
        anim.save(save_path, writer=writer)
        print(f"Animation saved!")
    else:
        plt.tight_layout()
        plt.show()

    return occupancy_map


def main():
    DEBUG = False
    ANIMATE = False

    """Main function to run the process."""
    file_path = "/Users/solz/Documents/rust/mouse_maze/output_data/lidar/lidar_data_1761096364.json"

    try:
        lidar_df = get_lidar_data(file_path)
    except Exception as e:
        print(f"Error: Could not load '{file_path}'. {e}")
        return

    if DEBUG:
        lidar_to_visualize = lidar_df.slice(0, 100)  # More frames for animation
    else:
        lidar_to_visualize = lidar_df

    if ANIMATE:
        # Create animation (every 5th frame to speed it up)
        occupancy_map = animate_lidar_scan(
            lidar_df,
            save_path=ANIMATION_PATH,
            fps=10,
            frame_skip=5,  # Animate every 5th reading
        )
        print("Carving safe corridor for pathfinding...")
        robot_positions = np.array(lidar_to_visualize["agent_position"].to_list())
        robot_gx, robot_gy = occupancy_map.world_to_grid(
            robot_positions[:, 0], robot_positions[:, 1]
        )
        valid_robot_pos = occupancy_map._is_within_bounds_vec(robot_gx, robot_gy)
        
        for gx, gy in zip(robot_gx[valid_robot_pos], robot_gy[valid_robot_pos]):
            cv2.circle(occupancy_map.grid, (gx, gy), radius=2, color=0, thickness=-1)
    else:
        # Original static version
        occupancy_map = build_occupancy_map(lidar_to_visualize)

    # 2. Define start and goal from the Polars DataFrame
    start_pos, goal_pos = get_start_goal_positions(lidar_to_visualize)

    # 3. Manually clear Start and Goal for A*
    # This ensures the pathfinder can *start* and *end* # without erasing the rest of the map.
    print("Manually clearing start and goal positions for pathfinder.")
    start_gx, start_gy = occupancy_map.world_to_grid(*start_pos)
    goal_gx, goal_gy = occupancy_map.world_to_grid(*goal_pos)

    # Clear a small 3x3 area (radius=1) at start and goal
    cv2.circle(
        occupancy_map.grid, (start_gx, start_gy), radius=1, color=0, thickness=-1
    )
    cv2.circle(occupancy_map.grid, (goal_gx, goal_gy), radius=1, color=0, thickness=-1)
    # --- *** END NEW LOGIC *** ---

    # 4. Find the path using the new function
    path = find_astar_path(occupancy_map, start_pos, goal_pos)

    # 5. Visualize the final result
    if path:
        print(f"Path found with {len(path)} waypoints.")
    visualize_results(occupancy_map, path, start_pos, goal_pos)


if __name__ == "__main__":
    main()
