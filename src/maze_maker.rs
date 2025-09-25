use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_ecs_tilemap::prelude::*;
use knossos::maze::{GameMap, HuntAndKill, OrthogonalMazeBuilder, Prim, RecursiveBacktracking};

// --- Constants ---
const MAZE_WIDTH: usize = 10;
const MAZE_HEIGHT: usize = 10;
const SEED: u64 = 490;
const TILE_SIZE: f32 = 16.0;
const GAME_MAP_SPAN: usize = 3;

pub struct MazeMakerPlugin;

impl Plugin for MazeMakerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_maze_system);
        app.add_systems(Startup, create_wall_colliders.after(setup_maze_system));
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
    pub goal_pos: Option<UVec2>,
    pub width: u32,
    pub height: u32,
}

impl MazeData {
    pub fn is_valid(&self) -> bool {
        !self.pattern.is_empty() && self.width > 0 && self.height > 0
    }
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

impl MazeGenerator {
    /// Generates a maze, parses it, and saves artifacts to disk.
    pub fn generate(&self) -> Result<MazeData, String> {

        println!("Generating maze with size {}x{} and seed {}", self.width, self.height, self.seed);
        let maze = OrthogonalMazeBuilder::new()
            .width(self.width)
            .height(self.height)
            .algorithm(Box::new(RecursiveBacktracking))
            .seed(Some(self.seed))
            .build();

        let formatter = GameMap::new()
            .span(self.span)
            .wall('#')
            .passage(' ')
            .with_start_goal().seed(Some(self.seed));


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

        if pattern.is_empty() || pattern[0].is_empty() {
            return Err("Generated an empty maze pattern.".to_string());
        }

        let height = pattern.len() as u32;
        let width = pattern[0].len() as u32;

        Ok(MazeData {
            pattern,
            start_pos,
            goal_pos,
            width,
            height,
        })
    }
}

/// Bevy system to set up the camera and trigger maze generation and spawning.
pub fn setup_maze_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    match MazeGenerator::default().generate() {
        Ok(maze_data) => {
            if let Some(start_pos) = spawn_maze_tilemap(&mut commands, &asset_server, &maze_data) {
                info!("Maze spawned. Player start position: {start_pos:?}");
            } else {
                warn!("Maze spawned, but no start position was found.");
            }
            
            // Store the entire maze data as a resource
            commands.insert_resource(maze_data);
        }
        Err(e) => error!("Failed to generate maze: {e}"),
    }
}

/// Spawns a bevy_ecs_tilemap entity from MazeData.
pub fn spawn_maze_tilemap(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    maze_data: &MazeData,
) -> Option<Vec3> {
    let texture_handle = asset_server.load("tiles.png");
    let map_size = TilemapSize {
        x: maze_data.width,
        y: maze_data.height,
    };
    let mut tile_storage = TileStorage::empty(map_size);
    let tilemap_entity = commands.spawn_empty().id();
    let mut start_world_pos = None;

    for y in 0..map_size.y {
        for x in 0..map_size.x {
            // Flip y-coordinate to match bevy_ecs_tilemap's origin (bottom-left)
            let tile_type = maze_data.pattern[(map_size.y - 1 - y) as usize][x as usize];
            let tile_pos = TilePos { x, y };

            let mut tile_commands = commands.spawn(TileBundle {
                position: tile_pos,
                tilemap_id: TilemapId(tilemap_entity),
                texture_index: TileTextureIndex(tile_type as u32),
                ..default()
            });

            match tile_type {
                TileType::Wall => {
                    tile_commands.insert(Wall);
                }
                TileType::Start => {
                    let tile_size = TilemapTileSize {
                        x: TILE_SIZE,
                        y: TILE_SIZE,
                    };
                    let grid_size = TilemapGridSize::from(tile_size);
                    let map_type = TilemapType::Square;
                    let anchor = TilemapAnchor::Center;

                    start_world_pos = Some(
                        TilePos { x, y }
                            .center_in_world(
                                &map_size.into(),
                                &grid_size,
                                &tile_size,
                                &map_type,
                                &anchor,
                            )
                            .extend(0.1),
                    );
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

    start_world_pos
}

fn create_wall_colliders(
    mut commands: Commands,
    wall_query: Query<(Entity, &TilePos), With<Wall>>,
    tilemap_query: Query<(&TilemapSize, &TilemapTileSize, &TilemapGridSize, &TilemapType, &TilemapAnchor)>,
) {
    // Get tilemap parameters for proper coordinate conversion
    let Ok((map_size, tile_size, grid_size, map_type, anchor)) = tilemap_query.get_single() else {
        warn!("Could not find tilemap for wall collider creation");
        return;
    };

    let mut wall_positions = Vec::new();
    
    for (entity, tile_pos) in wall_query.iter() {
        // Convert tile position to world position using the same method as the tilemap
        let world_pos = tile_pos.center_in_world(
            &(*map_size).into(),
            grid_size,
            tile_size,
            map_type,
            anchor,
        );
        
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
                    .collect()
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
        y: flipped_y 
    };

    bevy_tile_pos
        .center_in_world(
            &map_size.into(),
            &grid_size,
            &tile_size,
            &map_type,
            &anchor,
        )
        .extend(0.1)
}
