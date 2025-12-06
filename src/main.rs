mod brain;
mod lidar;
mod maze_maker;
mod mouse;

use crate::lidar::LiDAR;
use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_ecs_tilemap::TilemapPlugin;
use lidar::add_lidar;
use maze_maker::MazeReady;
use mouse::MouseBundle;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PhysicsPlugins::default().with_length_unit(16.0))
        .insert_resource(Gravity(Vec2::ZERO))
        .add_plugins(PhysicsDebugPlugin::default())
        .add_plugins(TilemapPlugin)
        .add_plugins((
            mouse::MousePlugin,
            lidar::LiDARPlugin,
            maze_maker::MazeMakerPlugin,
            brain::BrainPlugin,
        ))
        .add_systems(Update, handle_maze_ready)
        .run();
}

fn handle_maze_ready(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut maze_ready_events: EventReader<MazeReady>,
) {
    for event in maze_ready_events.read() {
        let mouse_entity = commands
            .spawn(MouseBundle::new(
                &mut meshes,
                &mut materials,
                event.start_world_pos,
            ))
            .id();

        let lidar_config = LiDAR {
            range: 100.0,
            num_rays: 32,
            update_frequency: 10.0,
            last_update: 0.0,
        };

        add_lidar(&mut commands, mouse_entity, lidar_config);

        info!("Maze ready! Spawning mouse at: {:?}", event.start_world_pos);
        info!("Goal position: {:?}", event.goal_world_pos);
    }
}
