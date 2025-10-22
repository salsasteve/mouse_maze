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
            .add_event::<LiDARScanEvent>()
            .add_systems(
                Update,
                (
                    simulate_lidar,
                    visualize_lidar,
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
            sensor_radius: 2.0, // Start at center
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

#[derive(Event, Debug, Clone)]
pub struct LiDARScanEvent {
    pub entity: Entity,
    pub scan: LiDARScan,
    pub transform: Transform,
    pub velocity: Option<Vec2>,
}

/// Global configuration
#[derive(Resource, Debug, Clone)]
pub struct LiDARConfig {
    pub visualization_enabled: bool,
    pub recording_enabled: bool,
    pub max_recordings: usize,
}

impl Default for LiDARConfig {
    fn default() -> Self {
        Self {
            visualization_enabled: true,
            recording_enabled: true,
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
// Replace your calculate_ray_angle function:
fn calculate_ray_angle(
    ray_index: usize,
    total_rays: usize,
    base_rotation: f32,
    angle_range: f32,
) -> f32 {
    // For full 360° coverage, distribute rays evenly around the circle
    let angle_step = angle_range / total_rays as f32;
    base_rotation + (ray_index as f32 * angle_step) - (angle_range / 2.0)
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
    mut scan_events: EventWriter<LiDARScanEvent>,
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

        scan_events.write(LiDARScanEvent {
            entity,
            scan: scan.clone(),
            transform: *transform,
            velocity: None,
        });
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

    for (i, (&distance, &hit_point)) in scan.distances.iter().zip(scan.hit_points.iter()).enumerate() {
        let is_hit = distance < lidar.range;
        let is_front_ray = i == lidar.num_rays / 2; // Front ray is the center ray

        if is_hit {
            let line_color = if is_front_ray {
                // Neon purple for front ray
                Color::srgba(0.8, 0.2, 1.0, 0.9)
            } else {
                // Original red coloring for other rays
                let intensity = 1.0 - (distance / lidar.range).min(1.0);
                Color::srgba(1.0, 0.1, 0.1, 0.3 + intensity * 0.7)
            };
            
            gizmos.line_2d(position, hit_point, line_color);
            
            // Make hit point solid and match the line color
            let hit_point_color = if is_front_ray {
                Color::srgb(0.8, 0.2, 1.0) // Solid neon purple (same as line but no alpha)
            } else {
                Color::srgb(1.0, 0.1, 0.1) // Solid red (same as line but no alpha)
            };
            
            gizmos.circle_2d(hit_point, 2.0, hit_point_color);
        } else {
            let line_color = if is_front_ray {
                // Dimmer purple for front ray when no hit
                Color::srgba(0.6, 0.1, 0.8, 0.3)
            } else {
                // Original green for missed rays
                Color::srgba(0.1, 0.8, 0.1, 0.1)
            };
            
            gizmos.line_2d(position, hit_point, line_color);
            // No hit points drawn for missed rays
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
    if keyboard_input.just_pressed(KeyCode::KeyR) {
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