use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_ecs_tilemap::TilemapPlugin;

mod maze_maker;
mod lidar;

use maze_maker::{MazeMakerPlugin, MazeData};
use lidar::{LiDARPlugin, LiDAR, add_lidar_to_entity}; 

pub struct MousePlugin;

impl Plugin for MousePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_mouse.after(maze_maker::setup_maze_system));
        app.add_systems(Update, (mouse_manual_control, camera_follow_mouse));
    }
}

const MOUSE_RADIUS: f32 = 4.0;
const MOUSE_SPEED: f32 = 200.0;

#[derive(Component)]
pub struct Mouse;

#[derive(Component)]
pub struct FollowCamera;

fn spawn_mouse(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    maze_data: Res<MazeData>,
) {
    debug!("Spawning mouse...");

    let mouse: Circle = Circle::new(MOUSE_RADIUS).into();
    let color = Color::srgb(1., 0., 0.);

    let mesh: Handle<Mesh> = meshes.add(mouse);
    let material = materials.add(color);
    let mouse_spawn_position = if let Some(start_tile_pos) = maze_data.start_pos {
        maze_maker::tile_pos_to_world_pos(start_tile_pos, &maze_data)
    } else {
        warn!("No start position found in maze data");
        Vec3::new(0.0, 0.0, 0.1)
    };

    let mouse_entity = commands.spawn((
        Mouse,
        RigidBody::Dynamic,
        Collider::circle(MOUSE_RADIUS),
        Mesh2d(mesh),
        MeshMaterial2d(material),
        Transform::from_translation(mouse_spawn_position),
        Friction::ZERO.with_combine_rule(CoefficientCombine::Min),
        Restitution::ZERO.with_combine_rule(CoefficientCombine::Min),
    )).id();

    add_lidar_to_entity(&mut commands, mouse_entity, LiDAR::default());
}

fn mouse_manual_control(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut mouse_query: Query<&mut LinearVelocity, With<Mouse>>,
) {
    for mut velocity in mouse_query.iter_mut() {
        let mut direction = Vec2::ZERO;

        if keyboard_input.pressed(KeyCode::ArrowUp) {
            direction.y += 1.0;
        }
        if keyboard_input.pressed(KeyCode::ArrowDown) {
            direction.y -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::ArrowLeft) {
            direction.x -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::ArrowRight) {
            direction.x += 1.0;
        }

        if direction.length_squared() > 0.0 {
            direction = direction.normalize();
            velocity.0 = direction * MOUSE_SPEED;
        } else {
            velocity.0 = Vec2::ZERO;
        }
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((Camera2d::default(), FollowCamera));
}

fn camera_follow_mouse(
    mouse_query: Query<&Transform, (With<Mouse>, Without<FollowCamera>)>,
    mut camera_query: Query<&mut Transform, (With<FollowCamera>, Without<Mouse>)>,
) {
    if let Ok(mouse_transform) = mouse_query.single() {
        if let Ok(mut camera_transform) = camera_query.single_mut() {
            camera_transform.translation.x = mouse_transform.translation.x;
            camera_transform.translation.y = mouse_transform.translation.y;
        }
    }
}

fn main() {
    let mut app = App::new();

    // Add all plugins first
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Bevy Maze Example".into(),
                    ..default()
                }),
                ..default()
            })
            .set(ImagePlugin::default_nearest()),
        PhysicsPlugins::default(),
    ));


    app.add_plugins(TilemapPlugin);
    app.add_plugins(MazeMakerPlugin);
    app.add_plugins(MousePlugin);
    app.add_plugins(LiDARPlugin); 

    // Add systems and resources
    app.add_systems(Startup, spawn_camera);
    app.insert_resource(Gravity(Vec2::ZERO));

    app.run();
}
