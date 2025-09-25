use avian2d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

use crate::mouse::Mouse;


// ============================================================================
// PLUGIN DEFINITION
// ============================================================================

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
                ).chain(), // Run in sequence for consistent data
            );
    }
}

// ============================================================================
// COMPONENTS
// ============================================================================

/// Component that enables LiDAR simulation on an entity
#[derive(Component, Debug, Clone)]
pub struct LiDAR {
    pub range: f32,
    pub num_rays: usize,
    pub angle_range: f32, // Total angle range in radians
    pub update_frequency: f32, // Hz - how often to update readings
    pub last_update: f32, // Internal timer
}

impl Default for LiDAR {
    fn default() -> Self {
        Self {
            range: 100.0, // 500 units
            num_rays: 72, // 5-degree increments
            angle_range: std::f32::consts::TAU, // Full 360 degrees
            update_frequency: 10.0, // 10 Hz
            last_update: 0.0,
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

// ============================================================================
// RESOURCES
// ============================================================================

/// Global LiDAR configuration
#[derive(Resource, Debug, Clone)]
pub struct LiDARConfig {
    pub visualization_enabled: bool,
    pub recording_enabled: bool,
    #[allow(dead_code)]
    pub ray_hit_color: Color,
    #[allow(dead_code)]
    pub ray_miss_color: Color,
    pub max_recordings: usize,
}

impl Default for LiDARConfig {
    fn default() -> Self {
        Self {
            visualization_enabled: true,
            recording_enabled: true,
            ray_hit_color: Color::srgba(1.0, 0.2, 0.2, 0.8),    // Semi-transparent red
            ray_miss_color: Color::srgba(0.2, 1.0, 0.2, 0.3),   // Very transparent green
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiDARRecording {
    pub timestamp: f64,
    pub agent_position: Vec2,
    pub agent_rotation: f32,
    pub agent_velocity: Vec2,
    pub distances: Vec<f32>,
    pub angles: Vec<f32>,
    pub hit_points: Vec<Vec2>,
    pub action_taken: Option<String>, // For recording what action was taken
}

// ============================================================================
// SYSTEMS
// ============================================================================

/// Main LiDAR simulation system - performs raycasting
fn simulate_lidar(
    mut lidar_query: Query<(Entity, &Transform, &mut LiDAR, &mut LiDARScan, Option<&LinearVelocity>)>,
    spatial_query: SpatialQuery,
    time: Res<Time>,
) {
    for (_entity, transform, mut lidar, mut scan, _velocity) in lidar_query.iter_mut() {
        // Check if it's time to update based on frequency
        lidar.last_update += time.delta_secs();
        if lidar.last_update < (1.0 / lidar.update_frequency) {
            continue;
        }
        lidar.last_update = 0.0;

        let position = transform.translation.truncate();
        let rotation = transform.rotation.to_euler(EulerRot::ZYX).0;
        
        let mut distances = Vec::with_capacity(lidar.num_rays);
        let mut angles = Vec::with_capacity(lidar.num_rays);
        let mut hit_points = Vec::with_capacity(lidar.num_rays);
        
        // Start rays outside the mouse collider
        let ray_start_offset = MOUSE_RADIUS + 2.0; // Start 2 units outside mouse
        
        // Cast rays in all directions
        for i in 0..lidar.num_rays {
            let angle = rotation + (i as f32 / lidar.num_rays as f32) * lidar.angle_range 
                       - (lidar.angle_range / 2.0);
            let direction = Dir2::new(Vec2::new(angle.cos(), angle.sin())).unwrap();
            
            // Start ray outside the mouse collider
            let ray_origin = position + direction.as_vec2() * ray_start_offset;
            let effective_range = lidar.range - ray_start_offset;
            
            // Perform raycast from outside the mouse
            if let Some(hit) = spatial_query.cast_ray(
                ray_origin,         // Start outside mouse
                direction,
                effective_range,    // Reduced range
                true,
                &SpatialQueryFilter::default(), // Use default filter
            ) {
                let total_distance = hit.distance + ray_start_offset;
                distances.push(total_distance);
                hit_points.push(position + direction.as_vec2() * total_distance);
            } else {
                distances.push(lidar.range);
                hit_points.push(position + direction.as_vec2() * lidar.range);
            }
            
            angles.push(angle);
        }
        
        // Update the scan component
        scan.distances = distances;
        scan.angles = angles;
        scan.hit_points = hit_points;
        scan.timestamp = time.elapsed_secs_f64();
    }
}

/// Enhanced visualizes LiDAR rays using Bevy's gizmo system
fn visualize_lidar(
    mut gizmos: Gizmos,
    lidar_query: Query<(&Transform, &LiDAR, &LiDARScan)>,
    config: Res<LiDARConfig>,
) {
    if !config.visualization_enabled {
        return;
    }

    for (transform, lidar, scan) in lidar_query.iter() {
        let position = transform.translation.truncate();
        
        // Draw rays with distance-based color intensity
        for (_i, (&distance, &hit_point)) in scan.distances.iter()
            .zip(scan.hit_points.iter())
            .enumerate() 
        {
            let is_hit = distance < lidar.range;
            
            if is_hit {
                // Draw hit rays with varying intensity based on distance
                let intensity = 1.0 - (distance / lidar.range).min(1.0);
                let color = Color::srgba(1.0, 0.1, 0.1, 0.3 + intensity * 0.7);
                gizmos.line_2d(position, hit_point, color);
                
                // Draw hit point
                gizmos.circle_2d(hit_point, 2.0, Color::srgb(1.0, 0.0, 0.0));
            } else {
                // Draw miss rays (max range) more subtly
                let color = Color::srgba(0.1, 0.8, 0.1, 0.1);
                gizmos.line_2d(position, hit_point, color);
            }
        }
        
        // Draw LiDAR sensor with pulsing effect
        let pulse = (scan.timestamp as f32 * 2.0).sin() * 0.3 + 0.7;
        gizmos.circle_2d(position, MOUSE_RADIUS + 2.0, Color::srgba(0.5, 0.5, 1.0, pulse));
        
        // Draw range circle (optional - shows max range)
        gizmos.circle_2d(position, lidar.range, Color::srgba(0.3, 0.3, 0.8, 0.1));
    }
}

fn record_lidar_data_with_actions(
    mut recorder: ResMut<LiDARDataRecorder>,
    lidar_query: Query<(&Transform, &LiDARScan, Option<&LinearVelocity>)>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    config: Res<LiDARConfig>,
) {
    if !config.recording_enabled {
        return;
    }

    // Determine current action
    let action = if keyboard_input.pressed(KeyCode::ArrowUp) {
        Some("move_up".to_string())
    } else if keyboard_input.pressed(KeyCode::ArrowDown) {
        Some("move_down".to_string())
    } else if keyboard_input.pressed(KeyCode::ArrowLeft) {
        Some("move_left".to_string())
    } else if keyboard_input.pressed(KeyCode::ArrowRight) {
        Some("move_right".to_string())
    } else {
        Some("no_action".to_string())
    };

    for (transform, scan, velocity) in lidar_query.iter() {
        // Only record if distances are meaningful (not all zeros)
        let meaningful_data = scan.distances.iter().any(|&d| d > 6.1);
        if !meaningful_data {
            continue; // Skip recordings with bad data
        }

        // Limit recordings to prevent memory issues
        if recorder.recordings.len() >= config.max_recordings {
            recorder.recordings.remove(0); // Remove oldest
        }

        let recording = LiDARRecording {
            timestamp: scan.timestamp,
            agent_position: transform.translation.truncate(),
            agent_rotation: transform.rotation.to_euler(EulerRot::ZYX).0,
            agent_velocity: velocity.map_or(Vec2::ZERO, |v| v.0),
            distances: scan.distances.clone(),
            angles: scan.angles.clone(),
            hit_points: scan.hit_points.clone(),
            action_taken: action.clone(),
        };

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

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Export LiDAR data to JSON file with statistics
pub fn export_lidar_data(recorder: &LiDARDataRecorder) {
    let filename = format!("lidar_data_{}.json", 
                          std::time::SystemTime::now()
                              .duration_since(std::time::UNIX_EPOCH)
                              .unwrap()
                              .as_secs());
    
    // Calculate statistics
    let total_recordings = recorder.recordings.len();
    let actions: std::collections::HashMap<String, usize> = recorder.recordings
        .iter()
        .filter_map(|r| r.action_taken.as_ref())
        .fold(std::collections::HashMap::new(), |mut acc, action| {
            *acc.entry(action.clone()).or_insert(0) += 1;
            acc
        });
    
    match serde_json::to_string_pretty(&recorder.recordings) {
        Ok(json_data) => {
            match File::create(&filename) {
                Ok(mut file) => {
                    if file.write_all(json_data.as_bytes()).is_ok() {
                        info!("Exported {} LiDAR recordings to {}", total_recordings, filename);
                        info!("Action distribution: {:?}", actions);
                        
                        if let Some(first) = recorder.recordings.first() {
                            info!("Sample distances: {:?}", &first.distances[..5.min(first.distances.len())]);
                        }
                    } else {
                        error!("Failed to write to file: {}", filename);
                    }
                }
                Err(e) => error!("Failed to create file {}: {}", filename, e),
            }
        }
        Err(e) => error!("Failed to serialize LiDAR data: {}", e),
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
fn debug_lidar_info(
    lidar_query: Query<(Entity, &Transform, &LiDAR, &LiDARScan)>,
    spatial_query: SpatialQuery,
) {
    for (entity, transform, lidar, scan) in lidar_query.iter() {
        let position = transform.translation.truncate();
        
        if !scan.distances.is_empty() {
            let hits = scan.distances.iter().filter(|&&d| d < lidar.range && d > 0.1).count();
            let zero_hits = scan.distances.iter().filter(|&&d| d < 0.1).count();
            let avg_distance = scan.distances.iter().sum::<f32>() / scan.distances.len() as f32;
            let min_distance = scan.distances.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
            
            // Print debug info occasionally
            if scan.timestamp as i32 % 2 == 0 { // Every 2 seconds
                info!("LiDAR Debug - Entity {:?}: {} real hits, {} zero hits, avg: {:.1}, min: {:.1}", 
                      entity, hits, zero_hits, avg_distance, min_distance);
                
                // Test a single ray manually
                let test_direction = Dir2::new(Vec2::new(1.0, 0.0)).unwrap();
                let filter = SpatialQueryFilter::default().with_excluded_entities([entity]);
                
                if let Some(hit) = spatial_query.cast_ray(
                    position,
                    test_direction,
                    lidar.range,
                    true,
                    &filter,
                ) {
                    info!("Manual test ray hit at distance: {:.2}, entity: {:?}", hit.distance, hit.entity);
                } else {
                    info!("Manual test ray missed (range: {})", lidar.range);
                }
            }
        }
    }
}

#[allow(dead_code)]
fn debug_raycast_simple(
    mouse_query: Query<&Transform, With<Mouse>>,
    spatial_query: SpatialQuery,
    mut gizmos: Gizmos,
) {
    for transform in mouse_query.iter() {
        let pos = transform.translation.truncate();
        let direction = Dir2::new(Vec2::new(1.0, 0.0)).unwrap();
        
        // Start ray outside the mouse
        let ray_start_offset = MOUSE_RADIUS + 2.0;
        let ray_origin = pos + direction.as_vec2() * ray_start_offset;
        
        // Test single ray
        if let Some(hit) = spatial_query.cast_ray(
            ray_origin,     // Start outside mouse
            direction,
            100.0 - ray_start_offset,
            true,
            &SpatialQueryFilter::default(),
        ) {
            let total_distance = hit.distance + ray_start_offset;
            info!("Ray hit at distance: {:.2} to entity: {:?}", total_distance, hit.entity);
            let hit_pos = pos + direction.as_vec2() * total_distance;
            gizmos.line_2d(pos, hit_pos, Color::srgb(1.0, 0.0, 0.0));
        } else {
            info!("Ray missed - no obstacles detected");
            gizmos.line_2d(pos, pos + direction.as_vec2() * 100.0, Color::srgb(0.0, 1.0, 0.0));
        }
        
        // Show the ray start point
        gizmos.circle_2d(ray_origin, 1.0, Color::srgb(1.0, 1.0, 0.0));
    }
}

// Import MOUSE_RADIUS from main - you'll need to make this available
const MOUSE_RADIUS: f32 = 4.0; // Temporary - import this properly