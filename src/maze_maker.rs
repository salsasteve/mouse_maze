use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_ecs_tilemap::prelude::*;
use knossos::maze::{GameMap, OrthogonalMazeBuilder, RecursiveBacktracking};
use std::fs;

// --- Constants ---
const MAZE_WIDTH: usize = 10;
const MAZE_HEIGHT: usize = 10;
const SEED: u64 = 490;
const TILE_SIZE: f32 = 16.0;
const GAME_MAP_SPAN: usize = 1;

pub struct MazeMakerPlugin;

impl Plugin for MazeMakerPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<MazeReady>().add_systems(
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
    Passage = 5,
    Wall = 1,
    Goal = 3,
    Start = 0,
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

impl TileType {
    fn to_char(self) -> char {
        match self {
            Self::Passage => ' ',
            Self::Wall => '#',
            Self::Start => 'S',
            Self::Goal => 'G',
        }
    }
}

#[derive(Debug, Resource, Clone)]
pub struct MazeData {
    pub pattern: Vec<Vec<TileType>>,
    pub start_pos: UVec2,
    pub goal_pos: UVec2,
    pub width: u32,
    pub height: u32,
}

#[derive(Event)]
pub struct MazeReady {
    pub start_world_pos: Vec3,
    pub goal_world_pos: Vec3,
}

pub fn generate_maze_system(mut commands: Commands) {
    let maze = OrthogonalMazeBuilder::new()
        .width(MAZE_WIDTH)
        .height(MAZE_HEIGHT)
        .algorithm(Box::new(RecursiveBacktracking))
        .seed(Some(SEED))
        .build();

    let formatter = GameMap::new()
        .span(GAME_MAP_SPAN)
        .wall('#')
        .passage(' ')
        .with_start_goal()
        .seed(Some(SEED));

    let maze_string = maze.format(formatter).into_inner();

    save_maze_to_file(&maze_string, "maze_original.txt");

    let mut maze_data = parse_maze(&maze_string);
    add_outer_wall_layer(&mut maze_data);

    save_maze_to_file(&format_maze_as_grid(&maze_data), "maze_modified.txt");

    commands.insert_resource(maze_data);
    info!("Maze generated with double-thick outer walls");
}

fn add_outer_wall_layer(maze_data: &mut MazeData) {
    let (old_width, old_height) = (maze_data.width as usize, maze_data.height as usize);
    let (new_width, new_height) = (old_width + 2, old_height + 2);

    let mut new_pattern = vec![vec![TileType::Wall; new_width]; new_height];

    for y in 0..old_height {
        for x in 0..old_width {
            new_pattern[y + 1][x + 1] = maze_data.pattern[y][x];
        }
    }

    maze_data.pattern = new_pattern;
    maze_data.width = new_width as u32;
    maze_data.height = new_height as u32;
    maze_data.start_pos += UVec2::ONE;
    maze_data.goal_pos += UVec2::ONE;

    info!(
        "Added outer wall layer. New size: {}x{}, Start: {:?}, Goal: {:?}",
        new_width, new_height, maze_data.start_pos, maze_data.goal_pos
    );
}

fn save_maze_to_file(content: &str, filename: &str) {
    fs::write(filename, content)
        .unwrap_or_else(|e| error!("Failed to save maze to {}: {}", filename, e));
}

fn parse_maze(maze_string: &str) -> MazeData {
    let lines: Vec<&str> = maze_string.lines().collect();
    let (width, height) = (lines[0].len(), lines.len());

    let pattern: Vec<Vec<TileType>> = lines
        .iter()
        .map(|line| line.chars().map(TileType::from).collect())
        .collect();

    let (mut start_pos, mut goal_pos) = (None, None);

    for (y, row) in pattern.iter().enumerate() {
        for (x, &tile) in row.iter().enumerate() {
            match tile {
                TileType::Start => start_pos = Some(UVec2::new(x as u32, y as u32)),
                TileType::Goal => goal_pos = Some(UVec2::new(x as u32, y as u32)),
                _ => {}
            }
        }
    }

    MazeData {
        pattern,
        start_pos: start_pos.expect("Maze must have a start position"),
        goal_pos: goal_pos.expect("Maze must have a goal position"),
        width: width as u32,
        height: height as u32,
    }
}

fn format_maze_as_grid(maze_data: &MazeData) -> String {
    let mut output = format!(
        "Maze: {}x{}\nStart: {:?}\nGoal: {:?}\n\n",
        maze_data.width, maze_data.height, maze_data.start_pos, maze_data.goal_pos
    );

    for row in &maze_data.pattern {
        for &tile in row {
            output.push(tile.to_char());
        }
        output.push('\n');
    }

    output
}

pub fn spawn_maze_tilemap(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    maze_data: Res<MazeData>,
    mut maze_ready_events: EventWriter<MazeReady>,
) {
    let map_size = TilemapSize {
        x: maze_data.width,
        y: maze_data.height,
    };
    let tile_size = TilemapTileSize {
        x: TILE_SIZE,
        y: TILE_SIZE,
    };

    let mut tile_storage = TileStorage::empty(map_size);
    let tilemap_entity = commands.spawn_empty().id();

    let (mut start_world_pos, mut goal_world_pos) = (None, None);

    for y in 0..map_size.y {
        for x in 0..map_size.x {
            let tile_type = maze_data.pattern[(map_size.y - 1 - y) as usize][x as usize];
            let tile_pos = TilePos { x, y };

            let tile_entity = spawn_tile(&mut commands, tilemap_entity, tile_pos, tile_type);

            if tile_type == TileType::Start {
                start_world_pos = Some(calculate_world_pos(tile_pos, &map_size, &tile_size));
            } else if tile_type == TileType::Goal {
                goal_world_pos = Some(calculate_world_pos(tile_pos, &map_size, &tile_size));
            }

            tile_storage.set(&tile_pos, tile_entity);
        }
    }

    commands.entity(tilemap_entity).insert(TilemapBundle {
        size: map_size,
        storage: tile_storage,
        texture: TilemapTexture::Single(asset_server.load("tiles.png")),
        tile_size,
        grid_size: tile_size.into(),
        anchor: TilemapAnchor::Center,
        ..default()
    });

    if let (Some(start), Some(goal)) = (start_world_pos, goal_world_pos) {
        maze_ready_events.write(MazeReady {
            start_world_pos: start,
            goal_world_pos: goal,
        });
        info!("Maze ready! Start: {}, Goal: {}", start, goal);
    }
}

fn spawn_tile(
    commands: &mut Commands,
    tilemap_entity: Entity,
    tile_pos: TilePos,
    tile_type: TileType,
) -> Entity {
    let mut tile_commands = commands.spawn(TileBundle {
        position: tile_pos,
        tilemap_id: TilemapId(tilemap_entity),
        texture_index: TileTextureIndex(tile_type as u32),
        ..default()
    });

    if tile_type == TileType::Wall {
        tile_commands.insert(Wall);
    }

    tile_commands.id()
}

fn calculate_world_pos(
    tile_pos: TilePos,
    map_size: &TilemapSize,
    tile_size: &TilemapTileSize,
) -> Vec3 {
    tile_pos
        .center_in_world(
            &(*map_size).into(),
            &TilemapGridSize::from(*tile_size),
            tile_size,
            &TilemapType::Square,
            &TilemapAnchor::Center,
        )
        .extend(0.1)
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
    let Ok((map_size, tile_size, grid_size, map_type, anchor)) = tilemap_query.single() else {
        return;
    };

    let wall_positions: Vec<Vec2> = wall_query
        .iter()
        .map(|(entity, tile_pos)| {
            commands.entity(entity).remove::<Wall>();
            tile_pos.center_in_world(&(*map_size).into(), grid_size, tile_size, map_type, anchor)
        })
        .collect();

    if !wall_positions.is_empty() {
        commands.spawn((
            RigidBody::Static,
            Collider::compound(
                wall_positions
                    .into_iter()
                    .map(|pos| (pos, 0.0, Collider::rectangle(tile_size.x, tile_size.y)))
                    .collect(),
            ),
            Transform::default(),
        ));

        info!(
            "Created wall colliders for {} walls",
            wall_query.iter().len()
        );
    }
}

impl MazeData {
    /// Convert world position to tile coordinates
    pub fn world_to_tile(&self, world_pos: Vec2) -> Option<UVec2> {
        const TILE_SIZE: f32 = 16.0;

        let half_width = (self.width as f32 * TILE_SIZE) / 2.0;
        let half_height = (self.height as f32 * TILE_SIZE) / 2.0;

        let tile_x = ((world_pos.x + half_width) / TILE_SIZE).floor() as i32;
        let tile_y = ((world_pos.y + half_height) / TILE_SIZE).floor() as i32;

        let array_y = (self.height as i32 - 1) - tile_y;

        if tile_x >= 0 && tile_x < self.width as i32 && array_y >= 0 && array_y < self.height as i32
        {
            Some(UVec2::new(tile_x as u32, array_y as u32))
        } else {
            None
        }
    }

    /// Convert tile coordinates to world position
    pub fn tile_to_world(&self, tile_pos: UVec2) -> Vec2 {
        const TILE_SIZE: f32 = 16.0;

        let half_width = (self.width as f32 * TILE_SIZE) / 2.0;
        let half_height = (self.height as f32 * TILE_SIZE) / 2.0;

        // Flip Y because array index and world Y are inverted
        let tile_y = (self.height - 1) - tile_pos.y;

        let world_x = (tile_pos.x as f32 * TILE_SIZE) - half_width + (TILE_SIZE / 2.0);
        let world_y = (tile_y as f32 * TILE_SIZE) - half_height + (TILE_SIZE / 2.0);

        Vec2::new(world_x, world_y)
    }

    /// Get goal position in world coordinates
    pub fn goal_world_pos(&self) -> Vec2 {
        self.tile_to_world(self.goal_pos)
    }

    /// Get start position in world coordinates
    pub fn start_world_pos(&self) -> Vec2 {
        self.tile_to_world(self.start_pos)
    }

    /// Check if a world position is a wall
    pub fn is_wall_at(&self, world_pos: Vec2) -> bool {
        self.world_to_tile(world_pos)
            .map(|tile| self.pattern[tile.y as usize][tile.x as usize] == TileType::Wall)
            .unwrap_or(true)
    }

    /// Get tile type at world position
    pub fn tile_at(&self, world_pos: Vec2) -> Option<TileType> {
        self.world_to_tile(world_pos)
            .map(|tile| self.pattern[tile.y as usize][tile.x as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_type_char_conversion() {
        assert_eq!(TileType::from(' '), TileType::Passage);
        assert_eq!(TileType::from('#'), TileType::Wall);
        assert_eq!(TileType::from('S'), TileType::Start);
        assert_eq!(TileType::from('G'), TileType::Goal);

        assert_eq!(TileType::Passage.to_char(), ' ');
        assert_eq!(TileType::Wall.to_char(), '#');
        assert_eq!(TileType::Start.to_char(), 'S');
        assert_eq!(TileType::Goal.to_char(), 'G');
    }

    #[test]
    fn test_parse_simple_maze() {
        let maze_string = "###\n#S#\n#G#\n###";
        let maze_data = parse_maze(maze_string);

        assert_eq!(maze_data.width, 3);
        assert_eq!(maze_data.height, 4);
        assert_eq!(maze_data.start_pos, UVec2::new(1, 1));
        assert_eq!(maze_data.goal_pos, UVec2::new(1, 2));
    }

    #[test]
    fn test_add_outer_wall_layer() {
        let mut maze_data = MazeData {
            pattern: vec![
                vec![TileType::Wall, TileType::Wall, TileType::Wall],
                vec![TileType::Wall, TileType::Start, TileType::Wall],
                vec![TileType::Wall, TileType::Goal, TileType::Wall],
                vec![TileType::Wall, TileType::Wall, TileType::Wall],
            ],
            start_pos: UVec2::new(1, 1),
            goal_pos: UVec2::new(1, 2),
            width: 3,
            height: 4,
        };

        add_outer_wall_layer(&mut maze_data);

        assert_eq!(maze_data.width, 5);
        assert_eq!(maze_data.height, 6);
        assert_eq!(maze_data.start_pos, UVec2::new(2, 2));
        assert_eq!(maze_data.goal_pos, UVec2::new(2, 3));

        // Check all edges are walls
        for x in 0..5 {
            assert_eq!(maze_data.pattern[0][x], TileType::Wall);
            assert_eq!(maze_data.pattern[5][x], TileType::Wall);
        }
        for y in 0..6 {
            assert_eq!(maze_data.pattern[y][0], TileType::Wall);
            assert_eq!(maze_data.pattern[y][4], TileType::Wall);
        }
    }

    #[test]
    fn test_format_maze_as_grid() {
        let maze_data = MazeData {
            pattern: vec![
                vec![TileType::Wall, TileType::Wall],
                vec![TileType::Start, TileType::Goal],
            ],
            start_pos: UVec2::new(0, 1),
            goal_pos: UVec2::new(1, 1),
            width: 2,
            height: 2,
        };

        let output = format_maze_as_grid(&maze_data);

        assert!(output.contains("Maze: 2x2"));
        assert!(output.contains("Start: UVec2(0, 1)"));
        assert!(output.contains("Goal: UVec2(1, 1)"));
        assert!(output.contains("##"));
        assert!(output.contains("SG"));
    }
}
