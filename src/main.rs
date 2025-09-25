use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_ecs_tilemap::TilemapPlugin;

mod maze_maker;
mod lidar;
mod mouse;
mod game_integration;

use maze_maker::MazeMakerPlugin;
use lidar::LiDARPlugin;
use mouse::MousePlugin;
use game_integration::GameIntegrationPlugin;

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
    app.add_plugins(GameIntegrationPlugin);

    app.insert_resource(Gravity(Vec2::ZERO));

    app.run();
}
