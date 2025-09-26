use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_ecs_tilemap::prelude::*;
use knossos::maze::{GameMap, OrthogonalMazeBuilder, RecursiveBacktracking};

// --- Constants ---
const MAZE_WIDTH: usize = 10;
const MAZE_HEIGHT: usize = 10;
const SEED: u64 = 490;
const TILE_SIZE: f32 = 16.0;
const GAME_MAP_SPAN: usize = 3;

pub struct MazeMakerPlugin;

impl Plugin for MazeMakerPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<MazeReady>();
        app.add_systems(
            Startup,
            (
                generate_maze_system,
                spawn_maze_tilemap,
                create_wall_colliders,
            )
                .chain(),
        );
    }
}

#[derive(Component)]
pub struct Wall;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileType {
    Passage = 0,
    Wall = 1,
    Goal = 5,
    Start = 6,
}

impl From<char> for TileType {
    fn from(ch: char) -> Self {
        match ch {
            ' ' => Self::Passage,
            '#' => Self::Wall,
            'S' => Self::Start,
            'G' => Self::Goal,
            _ => {
                warn!("Unknown tile char '{}', defaulting to Wall.", ch);
                Self::Wall
            }
        }
    }
}

#[derive(Debug, Resource)]
pub struct MazeData {
    pub pattern: Vec<Vec<TileType>>,
    pub start_pos: Option<UVec2>,
    #[allow(dead_code)]
    pub goal_pos: Option<UVec2>,
    pub width: u32,
    pub height: u32,
}

impl MazeData {
    fn is_valid(&self) -> bool {
        self.width > 5 && self.height > 5 && !self.pattern.is_empty() && !self.pattern[0].is_empty()
    }
}

#[derive(Event)]
pub struct MazeReady {
    pub start_world_pos: Vec3,
    pub start_tile_pos: UVec2,
    pub goal_world_pos: Option<Vec3>,
    pub goal_tile_pos: Option<UVec2>,
    pub maze_size: UVec2,
}

#[derive(Debug, Clone)]
pub struct MazeGenerator {
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub span: usize,
}

impl Default for MazeGenerator {
    fn default() -> Self {
        Self {
            width: MAZE_WIDTH,
            height: MAZE_HEIGHT,
            seed: SEED,
            span: GAME_MAP_SPAN,
        }
    }
}

pub fn generate_maze_system(mut commands: Commands) {
    let generator = MazeGenerator::default();

    let maze = OrthogonalMazeBuilder::new()
        .width(generator.width)
        .height(generator.height)
        .algorithm(Box::new(RecursiveBacktracking))
        .seed(Some(generator.seed))
        .build();

    let formatter = GameMap::new()
        .span(generator.span)
        .wall('#')
        .passage(' ')
        .with_start_goal()
        .seed(Some(generator.seed));

    let mut start_pos = None;
    let mut goal_pos = None;
    let pattern: Vec<Vec<TileType>> = maze
        .format(formatter)
        .into_inner()
        .lines()
        .enumerate()
        .map(|(y, line)| {
            line.chars()
                .enumerate()
                .map(|(x, ch)| {
                    let tile_type = TileType::from(ch);
                    if tile_type == TileType::Start {
                        start_pos = Some(UVec2::new(x as u32, y as u32));
                    } else if tile_type == TileType::Goal {
                        goal_pos = Some(UVec2::new(x as u32, y as u32));
                    }

                    tile_type
                })
                .collect()
        })
        .collect();

    let maze_data = MazeData {
        pattern: pattern.clone(),
        start_pos,
        goal_pos,
        width: pattern[0].len() as u32,
        height: pattern.len() as u32,
    };
    if maze_data.is_valid() {
        commands.insert_resource(maze_data);
        info!("Maze generated successfully");
    } else {
        error!("Generated maze data is invalid");
    }
}

pub fn spawn_maze_tilemap(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    maze_data: Res<MazeData>,
    mut maze_ready_events: EventWriter<MazeReady>,
) {
    let texture_handle = asset_server.load("tiles.png");
    let map_size = TilemapSize {
        x: maze_data.width,
        y: maze_data.height,
    };
    let mut tile_storage = TileStorage::empty(map_size);
    let tilemap_entity = commands.spawn_empty().id();

    let mut start_world_pos = None;
    let mut goal_world_pos = None;

    for y in 0..map_size.y {
        for x in 0..map_size.x {
            let tile_type = maze_data.pattern[(map_size.y - 1 - y) as usize][x as usize];
            let tile_pos = TilePos { x, y };

            let mut tile_commands = commands.spawn(TileBundle {
                position: tile_pos,
                tilemap_id: TilemapId(tilemap_entity),
                texture_index: TileTextureIndex(tile_type as u32),
                ..default()
            });

            let tile_size = TilemapTileSize {
                x: TILE_SIZE,
                y: TILE_SIZE,
            };
            let grid_size = TilemapGridSize::from(tile_size);
            let map_type = TilemapType::Square;
            let anchor = TilemapAnchor::Center;

            match tile_type {
                TileType::Wall => {
                    tile_commands.insert(Wall);
                }
                TileType::Start => {
                    let world_pos = TilePos { x, y }
                        .center_in_world(
                            &map_size.into(),
                            &grid_size,
                            &tile_size,
                            &map_type,
                            &anchor,
                        )
                        .extend(0.1);
                    start_world_pos = Some(world_pos);
                }
                TileType::Goal => {
                    let world_pos = TilePos { x, y }
                        .center_in_world(
                            &map_size.into(),
                            &grid_size,
                            &tile_size,
                            &map_type,
                            &anchor,
                        )
                        .extend(0.1);
                    goal_world_pos = Some(world_pos);
                }
                _ => {}
            }
            tile_storage.set(&tile_pos, tile_commands.id());
        }
    }

    let tile_size = TilemapTileSize {
        x: TILE_SIZE,
        y: TILE_SIZE,
    };
    commands.entity(tilemap_entity).insert(TilemapBundle {
        size: map_size,
        storage: tile_storage,
        texture: TilemapTexture::Single(texture_handle),
        tile_size,
        grid_size: tile_size.into(),
        anchor: TilemapAnchor::Center,
        ..default()
    });

    // Send the maze ready event with all important information
    if let Some(start_pos) = start_world_pos {
        maze_ready_events.write(MazeReady {
            start_world_pos: start_pos,
            start_tile_pos: maze_data.start_pos.unwrap_or(UVec2::ZERO),
            goal_world_pos: goal_world_pos,
            goal_tile_pos: maze_data.goal_pos,
            maze_size: UVec2::new(maze_data.width, maze_data.height),
        });
        info!("Maze is ready! Start position: {}", start_pos);
    } else {
        error!("Maze generated without start position!");
    }
}

fn create_wall_colliders(
    mut commands: Commands,
    wall_query: Query<(Entity, &TilePos), With<Wall>>,
    tilemap_query: Query<(
        &TilemapSize,
        &TilemapTileSize,
        &TilemapGridSize,
        &TilemapType,
        &TilemapAnchor,
    )>,
) {
    // Get tilemap parameters for proper coordinate conversion
    let Ok((map_size, tile_size, grid_size, map_type, anchor)) = tilemap_query.single() else {
        warn!("Could not find tilemap for wall collider creation");
        return;
    };

    let mut wall_positions = Vec::new();

    for (entity, tile_pos) in wall_query.iter() {
        // Convert tile position to world position using the same method as the tilemap
        let world_pos =
            tile_pos.center_in_world(&(*map_size).into(), grid_size, tile_size, map_type, anchor);

        wall_positions.push(world_pos);

        // Remove the Wall component to avoid processing again
        commands.entity(entity).remove::<Wall>();
    }

    // Create compound collider with correct world positions
    if !wall_positions.is_empty() {
        commands.spawn((
            RigidBody::Static,
            Collider::compound(
                wall_positions
                    .iter()
                    .map(|pos| (*pos, 0.0, Collider::rectangle(tile_size.x, tile_size.y)))
                    .collect(),
            ),
            Transform::default(),
        ));

        info!("Created wall colliders for {} walls", wall_positions.len());
    }
}

// Add a helper function to convert tile pos to world pos
pub fn tile_pos_to_world_pos(tile_pos: UVec2, maze_data: &MazeData) -> Vec3 {
    let map_size = TilemapSize {
        x: maze_data.width,
        y: maze_data.height,
    };
    let tile_size = TilemapTileSize {
        x: TILE_SIZE,
        y: TILE_SIZE,
    };
    let grid_size = TilemapGridSize::from(tile_size);
    let map_type = TilemapType::Square;
    let anchor = TilemapAnchor::Center;

    // Convert to flipped coordinates (matching the tilemap)
    let flipped_y = map_size.y - 1 - tile_pos.y;
    let bevy_tile_pos = TilePos {
        x: tile_pos.x,
        y: flipped_y,
    };

    bevy_tile_pos
        .center_in_world(&map_size.into(), &grid_size, &tile_size, &map_type, &anchor)
        .extend(0.1)
}
