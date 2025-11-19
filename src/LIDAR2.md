
///
/// This structure captures a complete snapshot of an agent's state at a specific moment,
/// combining sensor data with positional information and actions taken.
///
/// ## Data Format
///
/// Each recording contains the following information:
///
/// ```json
/// {
///   "timestamp": 45.67,                    // When this reading was taken (seconds)
///   "agent_position": [100.5, 200.3],      // Agent's world position [x, y]
///   "agent_rotation": 1.57,                // Agent's rotation in radians (~90 degrees)
///   "agent_velocity": [5.0, -2.0],         // Agent's velocity [x, y] units/second
///   "distances": [25.0, 30.0, 15.0, ...],  // Distance to obstacles for each ray
///   "angles": [0.0, 1.57, 3.14, ...],      // Absolute angle of each ray in radians
///   "hit_points": [                        // World coordinates where each ray hit
///     [125.5, 200.3],                      // Hit point for ray 0
///     [130.5, 230.3],                      // Hit point for ray 1
///     ...
///   ],
///   "action_taken": "move_up"               // Action performed at this moment
/// }
/// ```
///
/// ## Field Descriptions
///
/// ### `timestamp`
/// - **Type**: `f64` (seconds)
/// - **Description**: Game time when this recording was captured
/// - **Use**: Temporal ordering of training samples
///
/// ### `agent_position`
/// - **Type**: `[f32, f32]` (world coordinates)
/// - **Description**: Agent's center position in the world
/// - **Use**: Spatial context for the sensor reading
///
/// ### `agent_rotation`
/// - **Type**: `f32` (radians)
/// - **Description**: Agent's facing direction (0 = right, π/2 = up, π = left, 3π/2 = down)
/// - **Use**: Understanding sensor orientation relative to world
///
/// ### `agent_velocity`
/// - **Type**: `[f32, f32]` (units per second)
/// - **Description**: Agent's current movement velocity [x, y]
/// - **Use**: Motion context for predicting future states
///
/// ### `distances`
/// - **Type**: `Vec<f32>` (world units)
/// - **Description**: Distance from agent center to nearest obstacle for each ray
/// - **Range**: `[sensor_radius + 2.0, lidar.range]`
/// - **Use**: Primary sensor input for obstacle detection
///
/// ### `angles`
/// - **Type**: `Vec<f32>` (radians)
/// - **Description**: Absolute world angle of each ray direction
/// - **Calculation**: `agent_rotation + ray_offset`
/// - **Use**: Understanding spatial relationship of sensor readings
///
/// ### `hit_points`
/// - **Type**: `Vec<[f32, f32]>` (world coordinates)
/// - **Description**: World position where each ray intersected an obstacle
/// - **Use**: Detailed spatial mapping, obstacle shape inference
///
/// ### `action_taken`
/// - **Type**: `Option<String>`
/// - **Description**: Action performed by human/AI at this moment
/// - **Values**: `"move_up"`, `"move_down"`, `"move_left"`, `"move_right"`, `"no_action"`
/// - **Use**: Supervised learning target for action prediction
///
/// ## Machine Learning Applications
///
/// This data format is designed for various ML tasks:
///
/// ### Supervised Learning
/// ```python
/// # Input: [distances, angles, position, rotation, velocity]
/// # Output: action_taken
/// model = train_action_predictor(recordings)
/// ```
///
/// ### Reinforcement Learning
/// ```python
/// # State: distances + position + velocity
/// # Action: movement command
/// # Reward: progress toward goal, collision penalty
/// env = LiDARMazeEnvironment(recordings)
/// ```
///
/// ### Imitation Learning
/// ```python
/// # Learn from human demonstrations
/// policy = imitate_human_actions(recordings)
/// ```
///
/// ## Data Quality
///
/// The system filters recordings to ensure quality:
/// - Only records when `distances` contain meaningful readings (> 6.1 units)
/// - Limits total recordings to prevent memory issues
/// - Includes action distribution statistics in exports
///
/// ## Export Format
///
/// Data is exported as pretty-printed JSON with statistics:
/// ```
/// Exported 1500 LiDAR recordings to lidar_data_1640995200.json
/// Action distribution: {"move_up": 400, "move_right": 350, "no_action": 750}
/// Sample distances: [45.2, 67.8, 23.1, 89.4, 12.7]
/// ```