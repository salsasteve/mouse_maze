Here is a 4-slide summary of the system breakdown.

-----

### **Slide 1: System Overview & Architecture**

  * **Project:** A modular Bevy application to simulate a LiDAR-equipped agent ("mouse") in a procedurally generated maze for data collection.
  * **Architecture:** Built on Bevy's **Plugin** system. Key modules include:
      * `MazeMakerPlugin`: Generates the environment.
      * `MousePlugin`: Spawns and controls the agent.
      * `LiDARPlugin`: Simulates the 360° sensor.
      * `HandOnWallPlugin`: Contains the agent's AI/control logic.
      * `GameIntegrationPlugin`: Handles data recording and export.
  * **Core Data Flow:**
    1.  **Startup:** `MazeMakerPlugin` generates the maze, tilemap, and a single compound physics collider for all walls.
    2.  **Update (10 Hz):** `LiDARPlugin` triggers a scan.
    3.  **Scan:** 72 physics raycasts are performed to get distances.
    4.  **Record:** The scan data + agent's position, rotation, and action (e.g., "move\_up") are saved to a `LiDARRecording` struct.
    5.  **Export:** Pressing 'R' saves all recorded data to a JSON file.

-----

### **Slide 2: Environment Generation (Maze & Physics)**

  * **Maze Algorithm:**
      * Uses the `knossos` crate to generate a 10x10 abstract maze via **Recursive Backtracking**.
      * This pattern is formatted into a 21x21 ASCII grid (using `#` for walls, `     ` for passages).
  * **Tilemap Rendering:**
      * `bevy_ecs_tilemap` spawns all visual tiles.
      * It correctly flips the Y-axis, as the maze string's origin (top-left) differs from Bevy's (bottom-left).
  * **Physics Optimization:**
      * Instead of creating hundreds of individual colliders (one per wall tile), the system creates **one static `RigidBody`**.
      * This single entity holds a **`Collider::compound()`** shape, which contains all 200+ wall rectangles. This is much more performant for the physics engine (`bevy_xpbd`).

-----

### **Slide 3: Sensor Simulation (LiDAR)**

  * **Configuration:**
      * **Rays:** 72 (one every 5 degrees).
      * **Range:** 100.0 units.
      * **Frequency:** 10 Hz (scans every 0.1 seconds).
  * **Ray Casting Algorithm:**
      * For each of the 72 rays, it calculates the world-space angle based on the mouse's rotation.
      * It uses `spatial_query.cast_ray` to find the distance to the nearest wall collider.
      * **Key Detail:** The ray *origin* is offset by 4.0 units from the mouse's center to avoid self-collisions.
  * **Visualization:**
      * Uses Bevy's `Gizmos` to draw the scan results.
      * **Front Ray (Index 36):** Drawn in **neon purple**.
      * **Other Hits:** Drawn in **red** (brighter for closer hits).
      * **Misses:** Drawn as dim green dots at max range.

-----

### **Slide 4: Data Recording & Output**

  * **Data Structure:** A `LiDARRecording` struct is captured at 10 Hz. It contains a complete snapshot of the agent's state:
      * `timestamp`
      * `agent_position` (Vec2)
      * `agent_rotation` (f32)
      * `agent_velocity` (Vec2)
      * `distances` (Vec\<f32\>)
      * `angles` (Vec\<f32\>)
      * `hit_points` (Vec\<Vec2\>)
      * `action_taken` (String, e.g., "move\_up", "no\_action")
  * **Export:**
      * All recordings are pushed into a `Vec<LiDARRecording>` resource.
      * When the **'R' key** is pressed, this entire vector is serialized using `serde_json` and saved to a timestamped `.json` file.
  * **Data Analysis Finding:**
      * The sample JSON file (`...196.json`) only contains `action_taken` fields.
      * **Cause:** The `is_scan_meaningful()` function filters out all scans where no ray is longer than 6.1 units.
      * **Solution:** This threshold is likely too high, discarding all data from the start. It should be lowered or removed to ensure data is recorded.