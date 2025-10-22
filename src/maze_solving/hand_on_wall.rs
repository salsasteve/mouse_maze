use crate::{lidar::LiDARScanEvent, mouse::Mouse};
use avian2d::prelude::{AngularVelocity, LinearVelocity};
use bevy::prelude::*;

#[derive(Resource)]
pub struct HandOnWallConfig {
    pub speed: f32,
    pub turn_speed: f32,
    pub wall_distance: f32,
    pub obstacle_threshold: f32,
}

impl Default for HandOnWallConfig {
    fn default() -> Self {
        Self {
            speed: 100.0,
            turn_speed: 2.0,
            wall_distance: 30.0,
            obstacle_threshold: 8.0,
        }
    }
}

pub fn hand_on_wall_logic(
    lidar_events: &mut EventReader<LiDARScanEvent>,
    mouse_query: &mut Query<(&mut LinearVelocity, &mut AngularVelocity, &Transform), With<Mouse>>,
    config: &HandOnWallConfig,
) {
    for event in lidar_events.read() {
        if let Ok((mut linear_vel, mut angular_vel, transform)) = mouse_query.get_mut(event.entity) {
            let rotation = transform.rotation.to_euler(EulerRot::ZYX).0;
            let (front, right) = get_front_right_distances(&event.scan, rotation);
            
            // **ADAPTIVE THRESHOLDS** based on corridor width
            let corridor_width = right * 2.0; // Estimate corridor width
            let adaptive_threshold = (corridor_width * 0.3).max(5.0).min(15.0); // 30% of corridor width, clamped
            let adaptive_wall_distance = (corridor_width * 0.4).max(6.0).min(20.0);
            
            info!("F: {:.1}, R: {:.1}, Corridor: {:.1}, Thresh: {:.1}, Rot: {:.1}°", 
                  front, right, corridor_width, adaptive_threshold, rotation.to_degrees());
            
            // **ADAPTIVE 3-RULE HAND-ON-WALL**
            
            if front < adaptive_threshold {
                // Rule 1: Obstacle in front -> Turn left
                info!("Turn left (obstacle)");
                linear_vel.0 = Vec2::ZERO;
                angular_vel.0 = config.turn_speed;
            } 
            else if right > adaptive_wall_distance * 1.8 {
                // Rule 2: No wall on right -> Turn right (find wall)
                info!("Turn right (find wall)");
                linear_vel.0 = Vec2::ZERO;
                angular_vel.0 = -config.turn_speed;
            } 
            else {
                // Rule 3: Path clear -> Move forward
                info!("Move forward");
                let forward = Vec2::new(rotation.cos(), rotation.sin());
                linear_vel.0 = forward * config.speed;
                angular_vel.0 = 0.0;
            }
        }
    }
}

fn get_front_right_distances(scan: &crate::lidar::LiDARScan, mouse_rotation: f32) -> (f32, f32) {
    let mut front_min = f32::INFINITY;
    let mut right_min = f32::INFINITY;
    
    for (&distance, &angle) in scan.distances.iter().zip(&scan.angles) {
        let relative = (angle - mouse_rotation).rem_euclid(std::f32::consts::TAU);
        
        // Front: ±30 degrees
        if relative < std::f32::consts::FRAC_PI_6 || relative > 11.0 * std::f32::consts::FRAC_PI_6 {
            front_min = front_min.min(distance);
        }
        // Right: 30-150 degrees  
        else if relative >= std::f32::consts::FRAC_PI_6 && relative <= 5.0 * std::f32::consts::FRAC_PI_6 {
            right_min = right_min.min(distance);
        }
    }
    
    (front_min, right_min)
}

pub struct HandOnWallPlugin;

impl Plugin for HandOnWallPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HandOnWallConfig>();
    }
}