use crate::{lidar::LiDARScanEvent, mouse::Mouse, maze_solving::hand_on_wall::{hand_on_wall_logic,HandOnWallConfig}};
use avian2d::prelude::{AngularVelocity, LinearVelocity};
use bevy::prelude::*;

pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BrainState>()
            .add_systems(Update, (toggle_brain_mode, brain_system));
    }
}

#[derive(Resource, Default)]
pub struct BrainState {
    pub mode: BrainMode,
    pub log_counter: u32,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum BrainMode {
    #[default]
    Manual,
    Simple,
    HandOnWall,
}

impl BrainMode {
    fn next(self) -> Self {
        match self {
            BrainMode::Manual => BrainMode::Simple,
            BrainMode::Simple => BrainMode::HandOnWall,
            BrainMode::HandOnWall => BrainMode::Manual,
        }
    }
}

fn toggle_brain_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut brain_state: ResMut<BrainState>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        let previous_mode = brain_state.mode;
        brain_state.mode = brain_state.mode.next();
        info!("Brain Mode changed: {:?} -> {:?}", previous_mode, brain_state.mode);
    }
}

fn brain_system(
    mut lidar_events: EventReader<LiDARScanEvent>,
    mut mouse_query: Query<(&mut LinearVelocity, &mut AngularVelocity, &Transform), With<Mouse>>,
    mut brain_state: ResMut<BrainState>,
    config: ResMut<HandOnWallConfig>,  // Change from ResMut to Res
) {
    match brain_state.mode {
        BrainMode::Manual => {
             visualize_lidar(&mut lidar_events, &mouse_query, &mut brain_state);
        },
        BrainMode::Simple => simple_brain(&mut lidar_events, &mut mouse_query, &mut brain_state),
        BrainMode::HandOnWall => {
            hand_on_wall_logic(
                &mut lidar_events, 
                &mut mouse_query, 
                &config
            );
        }
    }
}

fn simple_brain(
    lidar_events: &mut EventReader<LiDARScanEvent>,
    mouse_query: &mut Query<(&mut LinearVelocity, &mut AngularVelocity, &Transform), With<Mouse>>, // Add AngularVelocity
    brain_state: &mut BrainState,
) {
    const SPEED: f32 = 200.0;
    const OBSTACLE_DISTANCE: f32 = 30.0;

    for event in lidar_events.read() {
        if let Ok((mut velocity, mut _angular_velocity, transform)) = mouse_query.get_mut(event.entity) {
            let rotation = transform.rotation.to_euler(EulerRot::ZYX).0;
            
            // Ensure we have scan data
            if event.scan.distances.is_empty() {
                continue;
            }
            
            let front_distance = event.scan.distances.iter()
                .zip(&event.scan.angles)
                .filter(|(_, angle)| {
                    let relative = (*angle - rotation).rem_euclid(std::f32::consts::TAU);
                    relative < std::f32::consts::FRAC_PI_4 || relative > 7.0 * std::f32::consts::FRAC_PI_4
                })
                .map(|(dist, _)| *dist)
                .fold(f32::INFINITY, f32::min);

            brain_state.log_counter += 1;
            if brain_state.log_counter % 60 == 0 {
                info!("Simple Brain - Front: {:.1}", front_distance);
            }

            // Your existing movement logic
            velocity.0 = if front_distance < OBSTACLE_DISTANCE {
                Vec2::new(-rotation.sin(), rotation.cos()) * SPEED * 0.5
            } else {
                Vec2::new(rotation.cos(), rotation.sin()) * SPEED
            };
            
            // Note: I'm using _angular_velocity to indicate it's unused in simple_brain
            // You can set it to 0.0 if needed: _angular_velocity.0 = 0.0;
        }
    }
}

fn visualize_lidar(
    lidar_events: &mut EventReader<LiDARScanEvent>,
    mouse_query: &Query<(&mut LinearVelocity, &mut AngularVelocity, &Transform), With<Mouse>>,
    brain_state: &mut BrainState,
) {
    for event in lidar_events.read() {
        if let Ok((_, _, transform)) = mouse_query.get(event.entity) {
            let rotation = transform.rotation.to_euler(EulerRot::ZYX).0;
            let position = transform.translation.truncate();
            
            brain_state.log_counter += 1;
            
            // Show LiDAR data every 30 frames (about twice per second)
            if brain_state.log_counter % 30 == 0 {
                info!("=== MANUAL MODE LIDAR DEBUG ===");
                info!("Mouse position: ({:.1}, {:.1}), rotation: {:.1}°", 
                      position.x, position.y, rotation.to_degrees());
                
                // Calculate directional distances
                let (front, right, back, left) = get_all_distances(&event.scan, rotation);
                
                info!("Directional distances:");
                info!("  Front: {:.1}", front);
                info!("  Right: {:.1}", right);
                info!("  Back:  {:.1}", back);
                info!("  Left:  {:.1}", left);
                
                // Show some individual rays for debugging
                info!("Sample LiDAR rays:");
                for (i, (&distance, &angle)) in event.scan.distances.iter()
                    .zip(&event.scan.angles)
                    .enumerate()
                    .take(8) // Show first 8 rays
                {
                    let relative = (angle - rotation).rem_euclid(std::f32::consts::TAU);
                    info!("  Ray {}: angle={:.0}°, relative={:.0}°, dist={:.1}", 
                          i, angle.to_degrees(), relative.to_degrees(), distance);
                }
                
                info!("Total rays: {}", event.scan.distances.len());
                info!("=== END LIDAR DEBUG ===\n");
            }
        }
    }
}

fn get_all_distances(scan: &crate::lidar::LiDARScan, mouse_rotation: f32) -> (f32, f32, f32, f32) {
    let mut front = f32::INFINITY;
    let mut right = f32::INFINITY;
    let mut back = f32::INFINITY;
    let mut left = f32::INFINITY;
    
    // Debug counters
    let mut front_count = 0;
    let mut right_count = 0;
    let mut back_count = 0;
    let mut left_count = 0;
    
    for (&distance, &angle) in scan.distances.iter().zip(&scan.angles) {
        // Convert to relative angle (0 to 360 degrees)
        let mut relative = (angle - mouse_rotation).to_degrees();
        while relative < 0.0 { relative += 360.0; }
        while relative >= 360.0 { relative -= 360.0; }
        
        // FIXED angle ranges
        if (relative >= 315.0 && relative <= 360.0) || (relative >= 0.0 && relative < 45.0) {
            // Front: 315° to 45°
            front = front.min(distance);
            front_count += 1;
        } else if relative >= 45.0 && relative < 135.0 {
            // LEFT: 45° to 135°
            left = left.min(distance);
            left_count += 1;
        } else if relative >= 135.0 && relative < 225.0 {
            // Back: 135° to 225°
            back = back.min(distance);
            back_count += 1;
        } else if relative >= 225.0 && relative < 315.0 {
            // RIGHT: 225° to 315°
            right = right.min(distance);
            right_count += 1;
        }
    }
    
    // Debug output every few calls
    static mut DEBUG_COUNTER: u32 = 0;
    unsafe {
        DEBUG_COUNTER += 1;
        if DEBUG_COUNTER % 30 == 0 {
            info!("Ray distribution - Front: {} rays, Right: {} rays, Back: {} rays, Left: {} rays", 
                  front_count, right_count, back_count, left_count);
            info!("Mouse rotation: {:.1}°", mouse_rotation.to_degrees());
        }
    }
    
    (front, right, back, left)
}

fn create_fixed_directional_rays(base_rotation: f32) -> Vec<f32> {
    vec![
        base_rotation + 0.0,                                    // North (front)
        base_rotation + std::f32::consts::FRAC_PI_4,           // NorthEast  
        base_rotation + std::f32::consts::FRAC_PI_2,           // East (right)
        base_rotation + 3.0 * std::f32::consts::FRAC_PI_4,     // SouthEast
        base_rotation + std::f32::consts::PI,                  // South (back)
        base_rotation + 5.0 * std::f32::consts::FRAC_PI_4,     // SouthWest
        base_rotation + 3.0 * std::f32::consts::FRAC_PI_2,     // West (left)
        base_rotation + 7.0 * std::f32::consts::FRAC_PI_4,     // NorthWest
    ]
}