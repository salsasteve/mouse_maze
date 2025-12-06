use super::{BrainMode, BrainState};
use crate::{
    lidar::LiDARScanEvent,
    maze_maker::MazeData,
    mouse::{Direction, Mouse, MouseCommands, MovementState, TILE_SIZE},
};
use bevy::prelude::*;

#[derive(Resource)]
pub struct LeftHandOnWallConfig {
    pub wall_threshold: f32,
    pub left_wall_distance: f32,
    pub goal_distance: f32, // Add this
}

impl Default for LeftHandOnWallConfig {
    fn default() -> Self {
        Self {
            wall_threshold: TILE_SIZE * 0.6,
            left_wall_distance: TILE_SIZE * 0.6,
            goal_distance: TILE_SIZE * 0.5, // ~8 units
        }
    }
}

pub fn execute_left_hand_on_wall(
    brain_state: Res<BrainState>,
    mut lidar_events: EventReader<LiDARScanEvent>,
    mut mouse_query: Query<(&Transform, &MovementState, &mut MouseCommands), With<Mouse>>,
    maze_data: Option<Res<MazeData>>, // Add this
    config: Res<LeftHandOnWallConfig>,
) {
    if brain_state.mode != BrainMode::LeftHandOnWall {
        return;
    }

    for event in lidar_events.read() {
        if let Ok((transform, movement_state, mut commands)) = mouse_query.get_mut(event.entity) {
            if movement_state.is_moving() {
                continue;
            }

            let pos = transform.translation.truncate();

            // CHECK GOAL - early return if reached
            if let Some(maze) = &maze_data {
                let dist = pos.distance(maze.goal_world_pos());
                if dist < config.goal_distance {
                    info!("🎉 [AI] GOAL REACHED! Distance: {:.1}", dist);
                    return;
                }
                info!("  🎯 Goal: {:.1} units away", dist);
            }

            let rotation = transform.rotation.to_euler(EulerRot::ZYX).0;
            let current_dir = get_current_direction(rotation);
            let (front, left, right, back) = get_directional_distances(&event.scan, rotation);

            // Debug output (rest unchanged)
            info!(
                "[AI CONFIG] wall_threshold: {:.1}, left_wall_distance: {:.1}, TILE_SIZE: {:.1}",
                config.wall_threshold, config.left_wall_distance, TILE_SIZE
            );
            info!(
                "[AI DEBUG] At ({:.1}, {:.1}) facing {:?}",
                pos.x, pos.y, current_dir
            );
            info!(
                "  Front: {:.1}, Left: {:.1}, Right: {:.1}, Back: {:.1}",
                front, left, right, back
            );
            info!(
                "  Can move Up: {:.1}",
                get_distance_in_direction(Direction::Up, current_dir, front, left, right, back)
            );
            info!(
                "  Can move Down: {:.1}",
                get_distance_in_direction(Direction::Down, current_dir, front, left, right, back)
            );
            info!(
                "  Can move Left: {:.1}",
                get_distance_in_direction(Direction::Left, current_dir, front, left, right, back)
            );
            info!(
                "  Can move Right: {:.1}",
                get_distance_in_direction(Direction::Right, current_dir, front, left, right, back)
            );

            // Navigation logic (unchanged)
            let front_blocked = front < config.wall_threshold;
            let left_blocked = left < config.wall_threshold;
            let right_blocked = right < config.wall_threshold;
            let back_open = back > config.wall_threshold;

            let next_move = if front_blocked && left_blocked && right_blocked && back_open {
                info!(
                    "[AI] DEAD END! Walls on 3 sides (F:{:.1}, L:{:.1}, R:{:.1}), back open ({:.1}) - turning around",
                    front, left, right, back
                );
                turn_around(current_dir)
            } else if left > config.left_wall_distance {
                info!(
                    "[AI] No left wall (dist: {:.1} > {:.1}) - turning left to find wall",
                    left, config.left_wall_distance
                );
                turn_left(current_dir)
            } else if front < config.wall_threshold {
                info!(
                    "[AI] Wall ahead (dist: {:.1} < {:.1}) - turning right",
                    front, config.wall_threshold
                );
                turn_right(current_dir)
            } else {
                info!(
                    "[AI] Following left wall (left: {:.1}, front: {:.1}) - moving forward",
                    left, front
                );
                current_dir
            };

            commands.move_in(next_move);
        }
    }
}

fn get_distance_in_direction(
    target_dir: Direction,
    current_dir: Direction,
    front: f32,
    left: f32,
    right: f32,
    back: f32,
) -> f32 {
    // Calculate relative position
    let diff = (target_dir as i32 - current_dir as i32).rem_euclid(4);
    match diff {
        0 => front, // Same direction
        1 => right, // 90° clockwise
        2 => back,  // 180°
        3 => left,  // 90° counter-clockwise
        _ => unreachable!(),
    }
}

fn get_directional_distances(
    scan: &crate::lidar::LiDARScan,
    mouse_rotation: f32,
) -> (f32, f32, f32, f32) {
    let mut front = f32::INFINITY;
    let mut left = f32::INFINITY;
    let mut right = f32::INFINITY;
    let mut back = f32::INFINITY;

    for (&distance, &angle) in scan.distances.iter().zip(&scan.angles) {
        let relative = (angle - mouse_rotation).rem_euclid(std::f32::consts::TAU);

        match relative {
            // Front: -45° to +45° (0° ± 45°)
            r if r < std::f32::consts::FRAC_PI_4 || r > 7.0 * std::f32::consts::FRAC_PI_4 => {
                front = front.min(distance);
            }
            // Left: 45° to 135° (90° ± 45°)
            r if r >= std::f32::consts::FRAC_PI_4 && r <= 3.0 * std::f32::consts::FRAC_PI_4 => {
                left = left.min(distance);
            }
            // Back: 135° to 225° (180° ± 45°)
            r if r > 3.0 * std::f32::consts::FRAC_PI_4 && r < 5.0 * std::f32::consts::FRAC_PI_4 => {
                back = back.min(distance);
            }
            // Right: 225° to 315° (270° ± 45°)
            _ => {
                right = right.min(distance);
            }
        }
    }

    (front, left, right, back)
}

fn get_current_direction(rotation: f32) -> Direction {
    let normalized = rotation.rem_euclid(std::f32::consts::TAU);

    if normalized < std::f32::consts::FRAC_PI_4 || normalized >= 7.0 * std::f32::consts::FRAC_PI_4 {
        Direction::Right
    } else if normalized < 3.0 * std::f32::consts::FRAC_PI_4 {
        Direction::Up
    } else if normalized < 5.0 * std::f32::consts::FRAC_PI_4 {
        Direction::Left
    } else {
        Direction::Down
    }
}

fn turn_left(current: Direction) -> Direction {
    match current {
        Direction::Up => Direction::Left,
        Direction::Left => Direction::Down,
        Direction::Down => Direction::Right,
        Direction::Right => Direction::Up,
    }
}

fn turn_right(current: Direction) -> Direction {
    match current {
        Direction::Up => Direction::Right,
        Direction::Right => Direction::Down,
        Direction::Down => Direction::Left,
        Direction::Left => Direction::Up,
    }
}

fn turn_around(current: Direction) -> Direction {
    match current {
        Direction::Up => Direction::Down,
        Direction::Down => Direction::Up,
        Direction::Left => Direction::Right,
        Direction::Right => Direction::Left,
    }
}

/*
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LEFT-HAND-ON-WALL ALGORITHM DOCUMENTATION                 ║
╔══════════════════════════════════════════════════════════════════════════════╗

OVERVIEW:
─────────
The left-hand-on-wall algorithm is a classic maze-solving technique that works by
keeping the left hand in contact with a wall and following it continuously. This
guarantees finding the exit in any maze with walls connected to the entrance,
though the path may not be optimal.

CORE PRINCIPLE:
───────────────
1. Always prefer turning left (to maintain wall contact)
2. If left turn is blocked, go straight
3. If straight is blocked, turn right
4. If all three are blocked, turn around (dead end)

HOW IT WORKS IN THIS IMPLEMENTATION:
────────────────────────────────────
This system uses LiDAR sensor data to detect walls and make navigation decisions.
The mouse continuously scans its environment in a 360° radius and processes the
distance readings to determine which direction to move.

KEY COMPONENTS:
───────────────
1. LiDARScanEvent: Provides 360° distance measurements
2. LeftHandOnWallConfig: Configurable thresholds for wall detection
3. Direction enum: Up, Down, Left, Right (absolute world coordinates)
4. Mouse rotation: Tracked to determine current facing direction

CONFIGURATION PARAMETERS:
─────────────────────────
• wall_threshold (default: TILE_SIZE * 0.6 ≈ 9.6 units)
  - Minimum distance to consider a path "open"
  - If distance < threshold, the path is blocked

• left_wall_distance (default: TILE_SIZE * 0.6 ≈ 9.6 units)
  - Expected distance to maintain from left wall
  - If distance > threshold, mouse turns left to find wall

• goal_distance (default: TILE_SIZE * 0.5 ≈ 8.0 units)
  - Distance at which goal is considered "reached"

SENSOR DATA PROCESSING:
──────────────────────
The LiDAR scan provides distances at multiple angles. These are grouped into
four directional quadrants relative to the mouse's current facing:

• Front:  -45° to +45° (0° ± 45°)
• Left:   +45° to +135° (90° ± 45°)
• Back:   +135° to +225° (180° ± 45°)
• Right:  +225° to +315° (270° ± 45°)

For each quadrant, we take the MINIMUM distance to get the closest obstacle.

DECISION LOGIC:
───────────────
The algorithm follows this priority order:

1. GOAL CHECK (highest priority)
   - If distance to goal < goal_distance → STOP (success!)

2. DEAD END DETECTION
   - If front, left, AND right are all blocked, but back is open → Turn around

3. LEFT WALL LOST
   - If left wall distance > left_wall_distance → Turn left
   - This ensures we maintain contact with the left wall

4. WALL AHEAD
   - If front distance < wall_threshold → Turn right
   - Can't go forward, so follow the wall by turning right

5. DEFAULT: FOLLOW WALL
   - Move straight ahead while maintaining left wall proximity

COORDINATE SYSTEM:
──────────────────
• World coordinates: Fixed reference frame
  - Direction::Right = 0° (positive X-axis)
  - Direction::Up = 90° (positive Y-axis)
  - Direction::Left = 180° (negative X-axis)
  - Direction::Down = 270° (negative Y-axis)

• Mouse rotation: Angle in radians from world X-axis
• Relative angles: Calculated from mouse's current facing direction

WHY IT WORKS:
─────────────
The left-hand-on-wall algorithm is guaranteed to solve any "simply connected" maze
(where all walls are connected to the outer boundary). By consistently following
the left wall, you will eventually trace the entire perimeter of the maze and
find the exit.

LIMITATIONS:
────────────
• May not find the shortest path (explores unnecessarily)
• Fails in mazes with disconnected walls or islands
• Can loop indefinitely if the goal is in an isolated section

BEVY ECS INTEGRATION:
────────────────────
This system runs in Bevy's Update schedule and:
1. Reads LiDARScanEvent from the event stream
2. Queries Mouse entities with Transform, MovementState, and MouseCommands
3. Only processes when BrainMode::LeftHandOnWall is active
4. Only makes decisions when the mouse is stationary (not currently moving)

╔══════════════════════════════════════════════════════════════════════════════╗
║                              ALGORITHM FLOWCHART                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

                           ┌─────────────────────┐
                           │  LiDAR Scan Event   │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │  Is mouse moving?   │
                           └──────────┬──────────┘
                                      │
                         Yes ◄────────┼────────► No
                          │           │           │
                          │           │           ▼
                          │           │  ┌─────────────────────┐
                          │           │  │ Get current position│
                          │           │  │   and rotation      │
                          │           │  └──────────┬──────────┘
                          │           │             │
                          │           │             ▼
                          │           │  ┌─────────────────────┐
                          │           │  │ Check goal distance │
                          │           │  └──────────┬──────────┘
                          │           │             │
                          │           │       ┌─────┴─────┐
                          │           │       │           │
                          │           │  < goal_dist   >= goal_dist
                          │           │       │           │
                          │           │       ▼           ▼
                          │           │  ┌─────────┐  ┌─────────────────────┐
                          │           │  │  STOP!  │  │ Process LiDAR scan  │
                          │           │  │ Success │  │ Get F/L/R/B dists   │
                          │           │  └─────────┘  └──────────┬──────────┘
                          │           │                           │
                          │           │                           ▼
                          │           │              ┌────────────────────────┐
                          │           │              │ F, L, R blocked AND    │
                          │           │              │    back open?          │
                          │           │              └────────┬───────────────┘
                          │           │                       │
                          │           │                  ┌────┴────┐
                          │           │                  │         │
                          │           │                 Yes       No
                          │           │                  │         │
                          │           │                  ▼         ▼
                          │           │          ┌──────────┐  ┌──────────────────┐
                          │           │          │Turn 180° │  │ Left dist >      │
                          │           │          │ (DEAD END)  │ left_wall_dist?  │
                          │           │          └─────┬────┘  └────────┬─────────┘
                          │           │                │                 │
                          │           │                │            ┌────┴────┐
                          │           │                │            │         │
                          │           │                │           Yes       No
                          │           │                │            │         │
                          │           │                │            ▼         ▼
                          │           │                │    ┌────────────┐  ┌─────────────────┐
                          │           │                │    │ Turn left  │  │ Front dist <    │
                          │           │                │    │(find wall) │  │ wall_threshold? │
                          │           │                │    └─────┬──────┘  └────────┬────────┘
                          │           │                │          │                  │
                          │           │                │          │             ┌────┴────┐
                          │           │                │          │             │         │
                          │           │                │          │            Yes       No
                          │           │                │          │             │         │
                          │           │                │          │             ▼         ▼
                          │           │                │          │     ┌────────────┐  ┌──────────┐
                          │           │                │          │     │Turn right  │  │ Go       │
                          │           │                │          │     │(avoid wall)│  │ straight │
                          │           │                │          │     └─────┬──────┘  └────┬─────┘
                          │           │                │          │           │              │
                          │           │                └──────────┴───────────┴──────────────┘
                          │           │                                       │
                          │           │                                       ▼
                          │           │                           ┌───────────────────────┐
                          │           │                           │ Issue move command    │
                          │           │                           │  to MouseCommands     │
                          │           │                           └───────────────────────┘
                          │           │
                          └───────────┴────► Wait for next frame
                                      │
                                      │
                                      ▼
                                  (Repeat)

LEGEND:
───────
F = Front distance    L = Left distance
R = Right distance    B = Back distance

NOTE: The algorithm only makes decisions when the mouse is stationary. Once a
      move command is issued, the system waits until the movement completes
      before processing the next LiDAR scan.

╚══════════════════════════════════════════════════════════════════════════════╝
*/
