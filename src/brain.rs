use crate::{lidar::LiDARScanEvent, mouse::Mouse};
use avian2d::prelude::LinearVelocity;
use bevy::prelude::*;

pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BrainState>()
            .add_systems(Update, (toggle_brain_mode, simple_brain));
    }
}

#[derive(Resource, Default)]
pub struct BrainState {
    pub ai_enabled: bool,
    pub log_counter: u32,
}

fn toggle_brain_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut brain_state: ResMut<BrainState>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        brain_state.ai_enabled = !brain_state.ai_enabled;
        info!("AI Mode: {}", if brain_state.ai_enabled { "ON" } else { "OFF" });
    }
}

fn simple_brain(
    mut lidar_events: EventReader<LiDARScanEvent>,
    mut mouse_query: Query<(&mut LinearVelocity, &Transform), With<Mouse>>,
    mut brain_state: ResMut<BrainState>,
) {
    if !brain_state.ai_enabled { return; }

    const SPEED: f32 = 200.0;
    const OBSTACLE_DISTANCE: f32 = 30.0;

    for event in lidar_events.read() {
        if let Ok((mut velocity, transform)) = mouse_query.get_mut(event.entity) {
            // Find closest obstacle in front (±45 degrees)
            let rotation = transform.rotation.to_euler(EulerRot::ZYX).0;
            let front_distance = event.scan.distances.iter()
                .zip(&event.scan.angles)
                .filter(|(_, angle)| {
                    let relative = (*angle - rotation).rem_euclid(std::f32::consts::TAU);
                    relative < std::f32::consts::FRAC_PI_4 || relative > 7.0 * std::f32::consts::FRAC_PI_4
                })
                .map(|(dist, _)| *dist)
                .fold(f32::INFINITY, f32::min);

            // Simple logging every 60 frames (~1 second at 60fps)
            brain_state.log_counter += 1;
            if brain_state.log_counter % 60 == 0 {
                let min_dist = event.scan.distances.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let avg_dist = event.scan.distances.iter().sum::<f32>() / event.scan.distances.len() as f32;
                info!("LiDAR: front={:.1}, min={:.1}, avg={:.1}", front_distance, min_dist, avg_dist);
            }

            // Simple behavior: move forward or turn left
            if front_distance < OBSTACLE_DISTANCE {
                // Turn left by moving perpendicular to current direction
                let turn_direction = Vec2::new(-rotation.sin(), rotation.cos());
                velocity.0 = turn_direction * SPEED * 0.5;
            } else {
                // Move forward
                let forward_direction = Vec2::new(rotation.cos(), rotation.sin());
                velocity.0 = forward_direction * SPEED;
            }
        }
    }
}