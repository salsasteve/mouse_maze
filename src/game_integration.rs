use bevy::prelude::*;
use crate::maze_maker::MazeReady;
use crate::lidar::{LiDAR, add_lidar_to_entity};
use crate::mouse::MouseBundle;  


pub struct GameIntegrationPlugin;

impl Plugin for GameIntegrationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, handle_maze_ready);
    }
}


fn handle_maze_ready(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut maze_ready_events: EventReader<MazeReady>,
) {
    for event in maze_ready_events.read() {
        info!("Maze ready! Spawning mouse at: {}", event.start_world_pos);
        
        let mouse_entity = commands
            .spawn(MouseBundle::new(&mut meshes, &mut materials, event.start_world_pos))
            .id();

        add_lidar_to_entity(&mut commands, mouse_entity, LiDAR::default());

        if let Some(goal_pos) = event.goal_world_pos {
            info!("Goal position: {}", goal_pos);
            // Spawn goal indicator, UI, etc.
        }
    }
}