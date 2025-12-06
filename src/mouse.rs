use avian2d::prelude::*;
use bevy::prelude::*;

use crate::maze_maker::MazeData;

pub struct MousePlugin;

impl Plugin for MousePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera).add_systems(
            Update,
            (
                execute_mouse_commands,
                camera_follow_mouse,
                camera_zoom,
                add_front_indicator,
                debug_mouse_state, // Add this
            ),
        );
    }
}

pub const MOUSE_RADIUS: f32 = 5.0;
pub const TILE_SIZE: f32 = 16.0;
const MOVEMENT_SPEED: f32 = 200.0;

// Core mouse component
#[derive(Component)]
pub struct Mouse;

#[derive(Component)]
pub struct MouseFrontIndicator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    Right = 0,
    Up = 1,
    Left = 2,
    Down = 3,
}

impl Direction {
    pub fn to_vec2(self) -> Vec2 {
        match self {
            Self::Up => Vec2::Y,
            Self::Down => -Vec2::Y,
            Self::Left => -Vec2::X,
            Self::Right => Vec2::X,
        }
    }

    pub fn to_rotation(self) -> f32 {
        match self {
            Self::Up => std::f32::consts::FRAC_PI_2,
            Self::Down => -std::f32::consts::FRAC_PI_2,
            Self::Left => std::f32::consts::PI,
            Self::Right => 0.0,
        }
    }
}

// Command system - external systems queue commands here
#[derive(Component, Default)]
pub struct MouseCommands {
    pub move_direction: Option<Direction>,
}

impl MouseCommands {
    pub fn move_in(&mut self, direction: Direction) {
        self.move_direction = Some(direction);
    }

    pub fn clear(&mut self) {
        self.move_direction = None;
    }
}

// Internal state
#[derive(Component, Default)]
pub struct MovementState {
    pub target_position: Option<Vec2>,        // Make public
    pub current_direction: Option<Direction>, // Make public
}

impl MovementState {
    pub fn is_moving(&self) -> bool {
        self.target_position.is_some()
    }
}

// Execute queued commands
fn execute_mouse_commands(
    mut query: Query<
        (
            &mut Transform,
            &mut MovementState,
            &mut MouseCommands,
            &mut LinearVelocity,
        ),
        With<Mouse>,
    >,
    maze_data: Option<Res<MazeData>>,
) {
    for (mut transform, mut state, mut commands, mut velocity) in &mut query {
        if let Some(dir) = commands.move_direction {
            let current_pos = transform.translation.truncate();
            let target_offset = dir.to_vec2() * crate::mouse::TILE_SIZE;
            let target = current_pos + target_offset;

            // Check if target is a wall BEFORE moving
            if let Some(maze) = &maze_data {
                if maze.is_wall_at(target) {
                    warn!(
                        "[MOUSE] Blocked! Target ({:.1}, {:.1}) is a wall",
                        target.x, target.y
                    );
                    velocity.0 = Vec2::ZERO;
                    commands.clear();
                    state.target_position = None;
                    continue;
                }

                // Also check current position
                let current_tile = maze.tile_at(current_pos);
                info!(
                    "[MOUSE] Moving {:?} from tile {:?} at ({:.1}, {:.1})",
                    dir, current_tile, current_pos.x, current_pos.y
                );
            }

            state.target_position = Some(target);
            state.current_direction = Some(dir);
            commands.clear();
        }

        if let Some(target) = state.target_position {
            let current_pos = transform.translation.truncate();
            let distance = current_pos.distance(target);

            // ADD THIS HERE - Update rotation during movement
            if let Some(dir) = state.current_direction {
                transform.rotation = Quat::from_rotation_z(dir.to_rotation());
            }

            if distance < 0.5 {
                // Snap to exact position
                transform.translation = target.extend(transform.translation.z);
                velocity.0 = Vec2::ZERO;
                info!("[MOUSE] Arrived at ({:.1}, {:.1})", target.x, target.y);
                state.target_position = None;
                state.current_direction = None;
            } else {
                // Move with debugging
                let direction = (target - current_pos).normalize();
                velocity.0 = direction * MOVEMENT_SPEED;

                // Debug: Check if we're hitting a wall
                if let Some(maze) = &maze_data {
                    if maze.is_wall_at(current_pos) {
                        error!(
                            "[MOUSE] ERROR: Inside wall at ({:.1}, {:.1})!",
                            current_pos.x, current_pos.y
                        );
                        velocity.0 = Vec2::ZERO;
                        state.target_position = None;
                    }
                }
            }
        } else {
            velocity.0 = Vec2::ZERO;
        }
    }
}

// Camera
#[derive(Component)]
pub struct FollowCamera;

fn spawn_camera(mut commands: Commands) {
    commands.spawn((Camera2d, FollowCamera));
}

fn camera_follow_mouse(
    mouse: Query<&Transform, With<Mouse>>,
    mut camera: Query<&mut Transform, (With<FollowCamera>, Without<Mouse>)>,
) {
    if let (Ok(mouse_t), Ok(mut camera_t)) = (mouse.single(), camera.single_mut()) {
        camera_t.translation.x = mouse_t.translation.x;
        camera_t.translation.y = mouse_t.translation.y;
    }
}

fn camera_zoom(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera: Query<&mut Projection, With<FollowCamera>>,
) {
    if let Ok(mut proj) = camera.single_mut() {
        if let Projection::Orthographic(ortho) = proj.as_mut() {
            if keyboard.pressed(KeyCode::KeyZ) {
                ortho.scale = (ortho.scale - 0.1).max(0.1);
            }
            if keyboard.pressed(KeyCode::KeyX) {
                ortho.scale = (ortho.scale + 0.1).min(5.0);
            }
        }
    }
}

// Mouse bundle
#[derive(Bundle)]
pub struct MouseBundle {
    pub mouse: Mouse,
    pub commands: MouseCommands,
    pub state: MovementState,
    pub rigid_body: RigidBody,
    pub collider: Collider,
    pub linear_velocity: LinearVelocity,
    pub angular_velocity: AngularVelocity,
    pub mesh: Mesh2d,
    pub material: MeshMaterial2d<ColorMaterial>,
    pub transform: Transform,
    pub friction: Friction,
    pub restitution: Restitution,
    pub locked_axes: LockedAxes,
}

impl MouseBundle {
    pub fn new(
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<ColorMaterial>,
        position: Vec3,
    ) -> Self {
        Self {
            mouse: Mouse,
            commands: MouseCommands::default(),
            state: MovementState::default(),
            rigid_body: RigidBody::Dynamic,
            collider: Collider::circle(MOUSE_RADIUS),
            linear_velocity: LinearVelocity::default(),
            angular_velocity: AngularVelocity::default(),
            mesh: Mesh2d(meshes.add(Circle::new(MOUSE_RADIUS))),
            material: MeshMaterial2d(materials.add(Color::srgb(1.0, 0.0, 0.0))),
            transform: Transform::from_translation(position)
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
            friction: Friction::ZERO.with_combine_rule(CoefficientCombine::Min),
            restitution: Restitution::ZERO.with_combine_rule(CoefficientCombine::Min),
            locked_axes: LockedAxes::ROTATION_LOCKED,
        }
    }
}

// Front indicator
fn add_front_indicator(
    mut commands: Commands,
    mice: Query<Entity, (With<Mouse>, Without<Children>)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for entity in &mice {
        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                MouseFrontIndicator,
                Mesh2d(meshes.add(create_semicircle_mesh(MOUSE_RADIUS))),
                MeshMaterial2d(materials.add(Color::BLACK)),
                Transform::from_xyz(0.0, 0.0, 0.1),
            ));
        });
    }
}

fn create_semicircle_mesh(radius: f32) -> Mesh {
    use bevy::render::mesh::{Indices, PrimitiveTopology};
    let segments = 32;
    let mut positions = vec![[0.0, 0.0, 0.0]];
    let mut indices = Vec::new();

    for i in 0..=segments {
        let angle =
            -std::f32::consts::FRAC_PI_2 + (i as f32 / segments as f32) * std::f32::consts::PI;
        positions.push([radius * angle.cos(), radius * angle.sin(), 0.0]);
    }

    for i in 0..segments {
        indices.extend_from_slice(&[0, i + 1, i + 2]);
    }

    Mesh::new(PrimitiveTopology::TriangleList, Default::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_indices(Indices::U32(indices))
}

fn debug_mouse_state(
    query: Query<
        (
            &Transform,
            &LinearVelocity,
            &AngularVelocity,
            &MouseCommands,
            &MovementState,
        ),
        With<Mouse>,
    >,
    maze_data: Option<Res<MazeData>>,
    time: Res<Time>,
) {
    // Only log every 0.5 seconds to avoid spam
    if time.elapsed_secs() % 0.5 < 0.016 {
        for (transform, vel, ang_vel, commands, state) in &query {
            let pos = transform.translation;
            info!(
                "[DEBUG] Mouse State:\n  Pos: ({:.1}, {:.1})\n  Vel: ({:.1}, {:.1})\n  AngVel: {:.1}\n  Command: {:?}\n  Target: {:?}",
                pos.x,
                pos.y,
                vel.0.x,
                vel.0.y,
                ang_vel.0,
                commands.move_direction,
                state.target_position
            );

            if let Some(maze) = &maze_data {
                if let Some(tile) = maze.world_to_tile(pos.truncate()) {
                    let tile_type = maze.tile_at(pos.truncate());
                    info!("  Current Tile: {:?} (type: {:?})", tile, tile_type);
                }
            }
        }
    }
}
