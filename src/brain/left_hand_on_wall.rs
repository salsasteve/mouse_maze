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
