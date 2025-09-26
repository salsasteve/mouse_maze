//! # LiDAR Simulation Module
//!
//! This module provides a complete LiDAR (Light Detection and Ranging) simulation system
//! for agents in a 2D environment. It's designed for machine learning and reinforcement
//! learning applications where agents need spatial awareness.
//!
//! ## Overview
//!
//! The LiDAR system works by casting rays in multiple directions around an agent and
//! recording the distances to obstacles. This creates a "sensor reading" that can be
//! used for:
//! - Obstacle avoidance
//! - Navigation
//! - Machine learning training data
//! - Spatial reasoning
//!
//! ## Core Components
//!
//! - [`LiDAR`]: Configuration component that defines sensor parameters
//! - [`LiDARScan`]: Stores the results of a sensor reading
//! - [`LiDARRecording`]: Complete data snapshot for ML training
//! - [`LiDARConfig`]: Global configuration for visualization and recording
//!
//! ## Usage Example
//!
//! ```rust
//! use bevy::prelude::*;
//! use crate::lidar::{LiDAR, add_lidar_to_entity};
//!
//! fn spawn_agent_with_lidar(mut commands: Commands) {
//!     let agent = commands.spawn((
//!         // Your agent components...
//!         Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
//!     )).id();
//!     
//!     // Add LiDAR with custom configuration
//!     let lidar_config = LiDAR {
//!         range: 150.0,           // 150 unit detection range
//!         num_rays: 36,           // 36 rays (10-degree increments)
//!         angle_range: std::f32::consts::PI, // 180-degree field of view
//!         update_frequency: 20.0, // 20 Hz update rate
//!         sensor_radius: 5.0,     // 5-unit sensor radius
//!         ..Default::default()
//!     };
//!     
//!     add_lidar_to_entity(&mut commands, agent, lidar_config);
//! }
//! ```
//!
//! ## Data Recording
//!
//! The system automatically records LiDAR data with keyboard actions for training ML models.
//! Press 'S' to export recorded data to a JSON file.

use avian2d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

pub struct LiDARPlugin;

impl Plugin for LiDARPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LiDARDataRecorder>()
            .init_resource::<LiDARConfig>()
            .add_systems(
                Update,
                (
                    simulate_lidar,
                    visualize_lidar,
                    // debug_raycast_simple,
                    record_lidar_data_with_actions,
                    export_data_on_keypress,
                )
                    .chain(),
            );
    }
}

/// Component that enables LiDAR simulation on an entity
#[derive(Component, Debug, Clone)]
pub struct LiDAR {
    pub range: f32,
    pub num_rays: usize,
    pub angle_range: f32,      // Total angle range in radians
    pub update_frequency: f32, // Hz - how often to update readings
    pub last_update: f32,      // Internal timer
    pub sensor_radius: f32,    // Distance from center to sensor origin
}

impl Default for LiDAR {
    fn default() -> Self {
        Self {
            range: 100.0,                       // 500 units
            num_rays: 72,                       // 5-degree increments
            angle_range: std::f32::consts::TAU, // Full 360 degrees
            update_frequency: 10.0,             // 10 Hz
            last_update: 0.0,
            sensor_radius: 4.0, // Start at center
        }
    }
}

/// Component to store the current LiDAR reading on an entity
#[derive(Component, Debug, Clone)]
pub struct LiDARScan {
    pub distances: Vec<f32>,
    pub angles: Vec<f32>,
    pub timestamp: f64,
    pub hit_points: Vec<Vec2>, // World positions of ray hits
}

/// Global LiDAR configuration
#[derive(Resource, Debug, Clone)]
pub struct LiDARConfig {
    pub visualization_enabled: bool,
    pub recording_enabled: bool,
    pub ray_hit_color: Color,
    pub ray_miss_color: Color,
    pub max_recordings: usize,
}

impl Default for LiDARConfig {
    fn default() -> Self {
        Self {
            visualization_enabled: true,
            recording_enabled: true,
            ray_hit_color: Color::srgba(1.0, 0.2, 0.2, 0.8), // Semi-transparent red
            ray_miss_color: Color::srgba(0.2, 1.0, 0.2, 0.3), // Very transparent green
            max_recordings: 10000,
        }
    }
}

/// Stores recorded LiDAR data for export
#[derive(Resource, Default, Debug)]
pub struct LiDARDataRecorder {
    pub recordings: Vec<LiDARRecording>,
}

/// Individual LiDAR data point for ML/RL training
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiDARRecording {
    /// Game time when this recording was captured (seconds since start)
    pub timestamp: f64,

    /// Agent's world position [x, y] in game units
    pub agent_position: Vec2,

    /// Agent's rotation in radians (0 = facing right, π/2 = facing up)
    pub agent_rotation: f32,

    /// Agent's current velocity [x, y] in units per second
    pub agent_velocity: Vec2,

    /// Distance to nearest obstacle for each ray, in world units
    /// Length matches the number of rays configured in LiDAR
    pub distances: Vec<f32>,

    /// Absolute angle of each ray in world coordinates (radians)
    /// Length matches distances array
    pub angles: Vec<f32>,

    /// World coordinates [x, y] where each ray hit an obstacle
    /// Length matches distances array
    pub hit_points: Vec<Vec2>,

    /// Action taken by human/AI at this moment
    /// None if no action system is active
    pub action_taken: Option<String>,
}

/// Check if LiDAR should update based on frequency
fn update_lidar(lidar: &mut LiDAR, delta_time: f32) -> bool {
    lidar.last_update += delta_time;
    if lidar.last_update >= (1.0 / lidar.update_frequency) {
        lidar.last_update = 0.0;
        true
    } else {
        false
    }
}

/// Calculate ray angle for a given ray index
fn calculate_ray_angle(
    ray_index: usize,
    total_rays: usize,
    base_rotation: f32,
    angle_range: f32,
) -> f32 {
    base_rotation + (ray_index as f32 / total_rays as f32) * angle_range - (angle_range / 2.0)
}

/// Cast a single ray and return the hit information
fn cast_single_ray(
    origin: Vec2,
    angle: f32,
    lidar: &LiDAR,
    spatial_query: &SpatialQuery,
    entity: Entity,
) -> (f32, Vec2) {
    // Direction vector from angle
    let direction = Dir2::new(Vec2::new(angle.cos(), angle.sin())).unwrap();
    // This is to avoid self-collision. Its say 2 units beyond the sensor radius.
    let ray_start_offset = lidar.sensor_radius + 2.0;
    // Ray origin is offset from entity center by sensor radius in the direction of the ray
    let ray_origin = origin + direction.as_vec2() * ray_start_offset;
    let effective_range = lidar.range - ray_start_offset;

    // Exclude the entity itself from raycast hits
    let filter = SpatialQueryFilter::default().with_excluded_entities([entity]);

    // Perform the raycast
    // A raycast is a line from origin in direction for effective_range distance
    if let Some(hit) = spatial_query.cast_ray(ray_origin, direction, effective_range, true, &filter)
    {
        let total_distance = hit.distance + ray_start_offset;
        let hit_point = origin + direction.as_vec2() * total_distance;
        (total_distance, hit_point)
    } else {
        let miss_point = origin + direction.as_vec2() * lidar.range;
        (lidar.range, miss_point)
    }
}

/// Perform full LiDAR scan for an entity
fn perform_lidar_scan(
    entity: Entity,
    transform: &Transform,
    lidar: &LiDAR,
    spatial_query: &SpatialQuery,
    timestamp: f64,
) -> LiDARScan {
    let position = transform.translation.truncate();
    let rotation = transform.rotation.to_euler(EulerRot::ZYX).0;

    let mut distances = Vec::with_capacity(lidar.num_rays);
    let mut angles = Vec::with_capacity(lidar.num_rays);
    let mut hit_points = Vec::with_capacity(lidar.num_rays);

    for i in 0..lidar.num_rays {
        let angle = calculate_ray_angle(i, lidar.num_rays, rotation, lidar.angle_range);
        let (distance, hit_point) = cast_single_ray(position, angle, lidar, spatial_query, entity);

        distances.push(distance);
        angles.push(angle);
        hit_points.push(hit_point);
    }

    LiDARScan {
        distances,
        angles,
        hit_points,
        timestamp,
    }
}

/// Main LiDAR simulation system
fn simulate_lidar(
    mut lidar_query: Query<(Entity, &Transform, &mut LiDAR, &mut LiDARScan)>,
    spatial_query: SpatialQuery,
    time: Res<Time>,
) {
    for (entity, transform, mut lidar, mut scan) in lidar_query.iter_mut() {
        if !update_lidar(&mut lidar, time.delta_secs()) {
            continue;
        }

        *scan = perform_lidar_scan(
            entity,
            transform,
            &lidar,
            &spatial_query,
            time.elapsed_secs_f64(),
        );
    }
}

/// Visualize LiDAR rays using Bevy's gizmo system
fn visualize_lidar(
    mut gizmos: Gizmos,
    lidar_query: Query<(&Transform, &LiDAR, &LiDARScan)>,
    config: Res<LiDARConfig>,
) {
    if !config.visualization_enabled {
        return;
    }

    for (transform, lidar, scan) in lidar_query.iter() {
        draw_lidar_rays(&mut gizmos, transform, lidar, scan);
        draw_lidar_sensor(&mut gizmos, transform, lidar, scan);
        draw_range_circle(&mut gizmos, transform, lidar);
    }
}

/// Draw all LiDAR rays with appropriate colors
fn draw_lidar_rays(gizmos: &mut Gizmos, transform: &Transform, lidar: &LiDAR, scan: &LiDARScan) {
    let position = transform.translation.truncate();

    for (&distance, &hit_point) in scan.distances.iter().zip(scan.hit_points.iter()) {
        let is_hit = distance < lidar.range;

        if is_hit {
            let intensity = 1.0 - (distance / lidar.range).min(1.0);
            let color = Color::srgba(1.0, 0.1, 0.1, 0.3 + intensity * 0.7);
            gizmos.line_2d(position, hit_point, color);
            gizmos.circle_2d(hit_point, 2.0, Color::srgb(1.0, 0.0, 0.0));
        } else {
            let color = Color::srgba(0.1, 0.8, 0.1, 0.1);
            gizmos.line_2d(position, hit_point, color);
        }
    }
}

/// Draw the LiDAR sensor with pulsing effect
fn draw_lidar_sensor(gizmos: &mut Gizmos, transform: &Transform, lidar: &LiDAR, scan: &LiDARScan) {
    let position = transform.translation.truncate();
    let pulse = (scan.timestamp as f32 * 2.0).sin() * 0.3 + 0.7;
    gizmos.circle_2d(
        position,
        lidar.sensor_radius + 2.0,
        Color::srgba(0.5, 0.5, 1.0, pulse),
    );
}

/// Draw the maximum range circle
fn draw_range_circle(gizmos: &mut Gizmos, transform: &Transform, lidar: &LiDAR) {
    let position = transform.translation.truncate();
    gizmos.circle_2d(position, lidar.range, Color::srgba(0.3, 0.3, 0.8, 0.1));
}

/// Determine current action from keyboard input
fn get_current_action(keyboard_input: &Res<ButtonInput<KeyCode>>) -> Option<String> {
    if keyboard_input.pressed(KeyCode::ArrowUp) {
        Some("move_up".to_string())
    } else if keyboard_input.pressed(KeyCode::ArrowDown) {
        Some("move_down".to_string())
    } else if keyboard_input.pressed(KeyCode::ArrowLeft) {
        Some("move_left".to_string())
    } else if keyboard_input.pressed(KeyCode::ArrowRight) {
        Some("move_right".to_string())
    } else {
        Some("no_action".to_string())
    }
}

/// Check if LiDAR scan data is meaningful (not all zeros)
fn is_scan_meaningful(scan: &LiDARScan) -> bool {
    scan.distances.iter().any(|&d| d > 6.1)
}

/// Create a LiDAR recording from current state
fn create_recording(
    transform: &Transform,
    scan: &LiDARScan,
    velocity: Option<&LinearVelocity>,
    action: Option<String>,
) -> LiDARRecording {
    LiDARRecording {
        timestamp: scan.timestamp,
        agent_position: transform.translation.truncate(),
        agent_rotation: transform.rotation.to_euler(EulerRot::ZYX).0,
        agent_velocity: velocity.map_or(Vec2::ZERO, |v| v.0),
        distances: scan.distances.clone(),
        angles: scan.angles.clone(),
        hit_points: scan.hit_points.clone(),
        action_taken: action,
    }
}

/// Record LiDAR data with action information
fn record_lidar_data_with_actions(
    mut recorder: ResMut<LiDARDataRecorder>,
    lidar_query: Query<(&Transform, &LiDARScan, Option<&LinearVelocity>)>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    config: Res<LiDARConfig>,
) {
    if !config.recording_enabled {
        return;
    }

    let action = get_current_action(&keyboard_input);

    for (transform, scan, velocity) in lidar_query.iter() {
        if !is_scan_meaningful(scan) {
            continue;
        }

        // Limit recordings to prevent memory issues
        if recorder.recordings.len() >= config.max_recordings {
            recorder.recordings.remove(0);
        }

        let recording = create_recording(transform, scan, velocity, action.clone());
        recorder.recordings.push(recording);
    }
}

/// Export data when 'S' key is pressed
fn export_data_on_keypress(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    recorder: Res<LiDARDataRecorder>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyS) {
        export_lidar_data(&recorder);
    }
}

/// Export LiDAR data to JSON file with statistics
pub fn export_lidar_data(recorder: &LiDARDataRecorder) {
    let filename = format!(
        "output_data/lidar/lidar_data_{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    let total_recordings = recorder.recordings.len();
    let actions = calculate_action_distribution(&recorder.recordings);

    match serde_json::to_string_pretty(&recorder.recordings) {
        Ok(json_data) => {
            if let Err(e) = write_data_to_file(&filename, &json_data) {
                error!("Failed to write data: {}", e);
            } else {
                log_export_statistics(total_recordings, &filename, &actions, &recorder.recordings);
            }
        }
        Err(e) => error!("Failed to serialize LiDAR data: {}", e),
    }
}

/// Calculate action distribution statistics
fn calculate_action_distribution(
    recordings: &[LiDARRecording],
) -> std::collections::HashMap<String, usize> {
    recordings
        .iter()
        .filter_map(|r| r.action_taken.as_ref())
        .fold(std::collections::HashMap::new(), |mut acc, action| {
            *acc.entry(action.clone()).or_insert(0) += 1;
            acc
        })
}

/// Write JSON data to file
fn write_data_to_file(filename: &str, data: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(filename)?;
    file.write_all(data.as_bytes())?;
    Ok(())
}

/// Log export statistics
fn log_export_statistics(
    total: usize,
    filename: &str,
    actions: &std::collections::HashMap<String, usize>,
    recordings: &[LiDARRecording],
) {
    info!("Exported {} LiDAR recordings to {}", total, filename);
    info!("Action distribution: {:?}", actions);

    if let Some(first) = recordings.first() {
        info!(
            "Sample distances: {:?}",
            &first.distances[..5.min(first.distances.len())]
        );
    }
}

/// Helper function to add LiDAR to an entity
pub fn add_lidar_to_entity(commands: &mut Commands, entity: Entity, lidar_config: LiDAR) {
    commands.entity(entity).insert((
        lidar_config,
        LiDARScan {
            distances: Vec::new(),
            angles: Vec::new(),
            timestamp: 0.0,
            hit_points: Vec::new(),
        },
    ));
}

#[allow(dead_code)]
fn debug_raycast_simple(
    lidar_query: Query<(&Transform, &LiDAR)>, // Changed from Mouse query
    spatial_query: SpatialQuery,
    mut gizmos: Gizmos,
) {
    for (transform, lidar) in lidar_query.iter() {
        let pos = transform.translation.truncate();
        let direction = Dir2::new(Vec2::new(1.0, 0.0)).unwrap();

        // Use configurable sensor radius
        let ray_start_offset = lidar.sensor_radius + 2.0;
        let ray_origin = pos + direction.as_vec2() * ray_start_offset;

        // Test single ray
        if let Some(hit) = spatial_query.cast_ray(
            ray_origin,
            direction,
            100.0 - ray_start_offset,
            true,
            &SpatialQueryFilter::default(),
        ) {
            let total_distance = hit.distance + ray_start_offset;
            info!(
                "Ray hit at distance: {:.2} to entity: {:?}",
                total_distance, hit.entity
            );
            let hit_pos = pos + direction.as_vec2() * total_distance;
            gizmos.line_2d(pos, hit_pos, Color::srgb(1.0, 0.0, 0.0));
        } else {
            info!("Ray missed - no obstacles detected");
            gizmos.line_2d(
                pos,
                pos + direction.as_vec2() * 100.0,
                Color::srgb(0.0, 1.0, 0.0),
            );
        }

        // Show the ray start point
        gizmos.circle_2d(ray_origin, 1.0, Color::srgb(1.0, 1.0, 0.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;
    use std::f32::consts::PI;

    #[test]
    fn test_lidar_default_values() {
        let lidar = LiDAR::default();

        assert_eq!(lidar.range, 100.0);
        assert_eq!(lidar.num_rays, 72);
        assert_eq!(lidar.angle_range, std::f32::consts::TAU);
        assert_eq!(lidar.update_frequency, 10.0);
        assert_eq!(lidar.last_update, 0.0);
        assert_eq!(lidar.sensor_radius, 4.0);
    }

    #[test]
    fn test_update_lidar_timing() {
        let mut lidar = LiDAR {
            update_frequency: 10.0, // 10 Hz = 0.1 second intervals
            last_update: 0.0,
            ..Default::default()
        };

        // Should not update immediately
        assert!(!update_lidar(&mut lidar, 0.05)); // 0.05 seconds passed
        assert_eq!(lidar.last_update, 0.05);

        // Should update after enough time
        assert!(update_lidar(&mut lidar, 0.06)); // Total 0.11 seconds
        assert_eq!(lidar.last_update, 0.0); // Reset after update
    }

    #[test]
    fn test_calculate_ray_angle() {
        let base_rotation = 0.0;
        let angle_range = PI; // 180 degrees
        let total_rays = 4;

        // First ray should be at -PI/4 (leftmost)
        let angle0 = calculate_ray_angle(0, total_rays, base_rotation, angle_range);
        assert!((angle0 - (-PI / 2.0)).abs() < 0.001);

        // Last ray should be at PI/4 (rightmost)
        let angle3 = calculate_ray_angle(3, total_rays, base_rotation, angle_range);
        assert!((angle3 - (PI / 4.0)).abs() < 0.001);

        // Middle ray should be at -PI/8
        let angle1 = calculate_ray_angle(1, total_rays, base_rotation, angle_range);
        assert!((angle1 - (-PI / 4.0)).abs() < 0.001);
    }

    #[test]
    fn test_calculate_ray_angle_with_rotation() {
        let base_rotation = PI / 2.0; // 90 degrees
        let angle_range = PI; // 180 degrees  
        let total_rays = 2;

        let angle0 = calculate_ray_angle(0, total_rays, base_rotation, angle_range);
        let angle1 = calculate_ray_angle(1, total_rays, base_rotation, angle_range);

        // With 90° base rotation and 180° range:
        // angle0 should be 90° - 90° = 0°
        // angle1 should be 90° + 0° = 90°
        assert!((angle0 - 0.0).abs() < 0.001);
        assert!((angle1 - PI / 2.0).abs() < 0.001);
    }

    #[test]
    fn test_get_current_action() {
        // This test would need to mock ButtonInput, but shows the concept
        // In a real test, you'd inject the input state

        // Pseudo-test showing the logic:
        // assert_eq!(get_current_action(&up_pressed), Some("move_up".to_string()));
        // assert_eq!(get_current_action(&no_keys), Some("no_action".to_string()));
    }

    #[test]
    fn test_is_scan_meaningful() {
        // Empty scan should not be meaningful
        let empty_scan = LiDARScan {
            distances: vec![],
            angles: vec![],
            timestamp: 0.0,
            hit_points: vec![],
        };
        assert!(!is_scan_meaningful(&empty_scan));

        // Scan with all short distances (inside sensor radius) not meaningful
        let close_scan = LiDARScan {
            distances: vec![2.0, 3.0, 4.0],
            angles: vec![0.0, 1.0, 2.0],
            timestamp: 0.0,
            hit_points: vec![Vec2::ZERO; 3],
        };
        assert!(!is_scan_meaningful(&close_scan));

        // Scan with at least one meaningful distance
        let meaningful_scan = LiDARScan {
            distances: vec![2.0, 10.0, 4.0], // 10.0 > 6.1
            angles: vec![0.0, 1.0, 2.0],
            timestamp: 0.0,
            hit_points: vec![Vec2::ZERO; 3],
        };
        assert!(is_scan_meaningful(&meaningful_scan));
    }

    #[test]
    fn test_create_recording() {
        let transform = Transform::from_translation(Vec3::new(10.0, 20.0, 0.0))
            .with_rotation(Quat::from_rotation_z(PI / 4.0));

        let scan = LiDARScan {
            distances: vec![15.0, 25.0],
            angles: vec![0.0, PI / 2.0],
            timestamp: 123.456,
            hit_points: vec![Vec2::new(25.0, 20.0), Vec2::new(10.0, 45.0)],
        };

        let velocity = LinearVelocity(Vec2::new(5.0, -3.0));
        let action = Some("move_up".to_string());

        let recording = create_recording(&transform, &scan, Some(&velocity), action);

        assert_eq!(recording.timestamp, 123.456);
        assert_eq!(recording.agent_position, Vec2::new(10.0, 20.0));
        assert!((recording.agent_rotation - PI / 4.0).abs() < 0.001);
        assert_eq!(recording.agent_velocity, Vec2::new(5.0, -3.0));
        assert_eq!(recording.distances, vec![15.0, 25.0]);
        assert_eq!(recording.angles, vec![0.0, PI / 2.0]);
        assert_eq!(recording.action_taken, Some("move_up".to_string()));
    }

    #[test]
    fn test_calculate_action_distribution() {
        let recordings = vec![
            LiDARRecording {
                action_taken: Some("move_up".to_string()),
                ..create_dummy_recording()
            },
            LiDARRecording {
                action_taken: Some("move_up".to_string()),
                ..create_dummy_recording()
            },
            LiDARRecording {
                action_taken: Some("move_left".to_string()),
                ..create_dummy_recording()
            },
            LiDARRecording {
                action_taken: None,
                ..create_dummy_recording()
            },
        ];

        let distribution = calculate_action_distribution(&recordings);

        assert_eq!(distribution.get("move_up"), Some(&2));
        assert_eq!(distribution.get("move_left"), Some(&1));
        assert_eq!(distribution.get("none"), None); // None actions not counted
    }

    #[test]
    fn test_lidar_config_defaults() {
        let config = LiDARConfig::default();

        assert!(config.visualization_enabled);
        assert!(config.recording_enabled);
        assert_eq!(config.max_recordings, 10000);
    }

    // Helper function for tests
    fn create_dummy_recording() -> LiDARRecording {
        LiDARRecording {
            timestamp: 0.0,
            agent_position: Vec2::ZERO,
            agent_rotation: 0.0,
            agent_velocity: Vec2::ZERO,
            distances: vec![],
            angles: vec![],
            hit_points: vec![],
            action_taken: None,
        }
    }

    #[test]
    fn test_write_data_to_file() {
        let test_data = r#"{"test": "data"}"#;
        let filename = "test_output.json";

        // Test successful write
        let result = write_data_to_file(filename, test_data);
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(filename);
    }

    // Integration test for the ray angle calculation across full circle
    #[test]
    fn test_full_circle_ray_distribution() {
        let num_rays = 8;
        let angle_range = std::f32::consts::TAU; // Full circle
        let base_rotation = 0.0;

        let mut angles = Vec::new();
        for i in 0..num_rays {
            angles.push(calculate_ray_angle(i, num_rays, base_rotation, angle_range));
        }

        // First ray should be at -PI (leftmost)
        assert!((angles[0] + PI).abs() < 0.001);

        // Should be evenly distributed
        let expected_step = std::f32::consts::TAU / num_rays as f32;
        for i in 1..num_rays {
            let expected_angle = angles[0] + (i as f32 * expected_step);
            assert!((angles[i] - expected_angle).abs() < 0.001);
        }
    }
}
