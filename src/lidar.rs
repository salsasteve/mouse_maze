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
                    record_lidar_data,
                    export_data_on_keypress,
                )
                    .chain(),
            );
    }
}

#[derive(Component, Debug, Clone)]
pub struct LiDAR {
    pub range: f32,
    pub num_rays: usize,
    pub update_frequency: f32,
    pub last_update: f32,
}

impl Default for LiDAR {
    fn default() -> Self {
        Self {
            range: 100.0,
            num_rays: 36,           // 10-degree increments for 360°
            update_frequency: 10.0, // 10 Hz
            last_update: 0.0,
        }
    }
}

#[derive(Component, Debug, Clone)]
pub struct LiDARScan {
    pub distances: Vec<f32>,
    pub angles: Vec<f32>, // Absolute world angles
    pub timestamp: f64,
}

#[derive(Event, Debug, Clone)]
pub struct LiDARScanEvent {
    pub entity: Entity,
    pub scan: LiDARScan,
}

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

#[derive(Resource, Default, Debug)]
pub struct LiDARDataRecorder {
    pub recordings: Vec<LiDARRecording>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiDARRecording {
    pub timestamp: f64,
    pub position: Vec2,
    pub rotation: f32,
    pub velocity: Vec2,
    pub distances: Vec<f32>,
    pub angles: Vec<f32>,
    pub action: Option<String>,
}

fn should_update(lidar: &mut LiDAR, delta: f32) -> bool {
    lidar.last_update += delta;
    if lidar.last_update >= 1.0 / lidar.update_frequency {
        lidar.last_update = 0.0;
        true
    } else {
        false
    }
}

fn calculate_ray_angle(ray_index: usize, total_rays: usize, base_rotation: f32) -> f32 {
    let angle_step = std::f32::consts::TAU / total_rays as f32;
    base_rotation + (ray_index as f32 * angle_step)
}

fn cast_ray(
    origin: Vec2,
    angle: f32,
    range: f32,
    spatial_query: &SpatialQuery,
    entity: Entity,
) -> f32 {
    let direction = Dir2::new(Vec2::new(angle.cos(), angle.sin())).unwrap();
    let filter = SpatialQueryFilter::default().with_excluded_entities([entity]);

    if let Some(hit) = spatial_query.cast_ray(origin, direction, range, true, &filter) {
        hit.distance
    } else {
        range
    }
}

fn perform_scan(
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

    for i in 0..lidar.num_rays {
        let angle = calculate_ray_angle(i, lidar.num_rays, rotation);
        let distance = cast_ray(position, angle, lidar.range, spatial_query, entity);

        distances.push(distance);
        angles.push(angle);
    }

    LiDARScan {
        distances,
        angles,
        timestamp,
    }
}

fn simulate_lidar(
    mut lidar_query: Query<(Entity, &Transform, &mut LiDAR, &mut LiDARScan)>,
    spatial_query: SpatialQuery,
    time: Res<Time>,
    mut events: EventWriter<LiDARScanEvent>,
) {
    for (entity, transform, mut lidar, mut scan) in lidar_query.iter_mut() {
        if !should_update(&mut lidar, time.delta_secs()) {
            continue;
        }

        *scan = perform_scan(
            entity,
            transform,
            &lidar,
            &spatial_query,
            time.elapsed_secs_f64(),
        );

        events.write(LiDARScanEvent {
            entity,
            scan: scan.clone(),
        });
    }
}

fn visualize_lidar(
    mut gizmos: Gizmos,
    query: Query<(&Transform, &LiDAR, &LiDARScan)>,
    config: Res<LiDARConfig>,
) {
    if !config.visualization_enabled {
        return;
    }

    for (transform, lidar, scan) in query.iter() {
        let origin = transform.translation.truncate();

        // Draw range circle
        gizmos.circle_2d(origin, lidar.range, Color::srgba(0.3, 0.3, 0.8, 0.1));

        // Draw rays
        for (&distance, &angle) in scan.distances.iter().zip(scan.angles.iter()) {
            let end = origin + Vec2::new(angle.cos(), angle.sin()) * distance;
            let is_hit = distance < lidar.range;

            let color = if is_hit {
                let intensity = 1.0 - (distance / lidar.range);
                Color::srgba(1.0, 0.1, 0.1, 0.3 + intensity * 0.7)
            } else {
                Color::srgba(0.1, 0.8, 0.1, 0.1)
            };

            gizmos.line_2d(origin, end, color);

            if is_hit {
                gizmos.circle_2d(end, 2.0, Color::srgb(1.0, 0.1, 0.1));
            }
        }

        // Draw sensor center
        gizmos.circle_2d(origin, 3.0, Color::srgba(0.5, 0.5, 1.0, 0.8));
    }
}

fn get_action(keyboard: &Res<ButtonInput<KeyCode>>) -> Option<String> {
    if keyboard.pressed(KeyCode::ArrowUp) {
        Some("up".to_string())
    } else if keyboard.pressed(KeyCode::ArrowDown) {
        Some("down".to_string())
    } else if keyboard.pressed(KeyCode::ArrowLeft) {
        Some("left".to_string())
    } else if keyboard.pressed(KeyCode::ArrowRight) {
        Some("right".to_string())
    } else {
        None
    }
}

fn record_lidar_data(
    mut recorder: ResMut<LiDARDataRecorder>,
    query: Query<(&Transform, &LiDARScan, Option<&LinearVelocity>)>,
    keyboard: Res<ButtonInput<KeyCode>>,
    config: Res<LiDARConfig>,
) {
    if !config.recording_enabled {
        return;
    }

    let action = get_action(&keyboard);

    for (transform, scan, velocity) in query.iter() {
        if recorder.recordings.len() >= config.max_recordings {
            recorder.recordings.remove(0);
        }

        recorder.recordings.push(LiDARRecording {
            timestamp: scan.timestamp,
            position: transform.translation.truncate(),
            rotation: transform.rotation.to_euler(EulerRot::ZYX).0,
            velocity: velocity.map_or(Vec2::ZERO, |v| v.0),
            distances: scan.distances.clone(),
            angles: scan.angles.clone(),
            action: action.clone(),
        });
    }
}

fn export_data_on_keypress(keyboard: Res<ButtonInput<KeyCode>>, recorder: Res<LiDARDataRecorder>) {
    if keyboard.just_pressed(KeyCode::KeyR) {
        export_lidar_data(&recorder);
    }
}

pub fn export_lidar_data(recorder: &LiDARDataRecorder) {
    let filename = format!(
        "output_data/lidar/lidar_data_{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    match serde_json::to_string_pretty(&recorder.recordings) {
        Ok(json) => {
            if let Err(e) = File::create(&filename).and_then(|mut f| f.write_all(json.as_bytes())) {
                error!("Failed to write data: {}", e);
            } else {
                info!(
                    "Exported {} recordings to {}",
                    recorder.recordings.len(),
                    filename
                );
            }
        }
        Err(e) => error!("Failed to serialize: {}", e),
    }
}

pub fn add_lidar(commands: &mut Commands, entity: Entity, config: LiDAR) {
    commands.entity(entity).insert((
        config,
        LiDARScan {
            distances: Vec::new(),
            angles: Vec::new(),
            timestamp: 0.0,
        },
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_ray_angle() {
        // 4 rays at 0° rotation should be at 0°, 90°, 180°, 270°
        assert!((calculate_ray_angle(0, 4, 0.0) - 0.0).abs() < 0.01);
        assert!((calculate_ray_angle(1, 4, 0.0) - std::f32::consts::FRAC_PI_2).abs() < 0.01);
        assert!((calculate_ray_angle(2, 4, 0.0) - std::f32::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_should_update() {
        let mut lidar = LiDAR {
            update_frequency: 10.0,
            last_update: 0.0,
            ..Default::default()
        };

        // Should not update after small delta
        assert!(!should_update(&mut lidar, 0.05));

        // Should update after accumulating enough time
        assert!(!should_update(&mut lidar, 0.04));
        assert!(should_update(&mut lidar, 0.02)); // Total 0.11s > 0.1s

        // Timer should reset
        assert_eq!(lidar.last_update, 0.0);
    }

    #[test]
    fn test_lidar_default() {
        let lidar = LiDAR::default();
        assert_eq!(lidar.range, 100.0);
        assert_eq!(lidar.num_rays, 36);
        assert_eq!(lidar.update_frequency, 10.0);
    }
}
