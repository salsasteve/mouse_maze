use super::{BrainMode, BrainState};
use crate::{
    lidar::LiDARScanEvent,
    maze_maker::MazeData,
    mouse::{Direction, Mouse, MouseCommands, MovementState},
};
use bevy::prelude::*;
use rand::Rng;

#[derive(Resource)]
pub struct RLConfig {
    pub exploration_rate: f32,  // Epsilon for epsilon-greedy
    pub learning_rate: f32,
    pub discount_factor: f32,   // Gamma
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            exploration_rate: 0.1,
            learning_rate: 0.001,
            discount_factor: 0.99,
        }
    }
}

#[derive(Resource, Default)]
pub struct RLAgent {
    pub total_reward: f32,
    pub episode_steps: u32,
}

pub fn execute_rl_agent(
    brain_state: Res<BrainState>,
    mut lidar_events: EventReader<LiDARScanEvent>,
    mut mouse_query: Query<(&Transform, &MovementState, &mut MouseCommands), With<Mouse>>,
    maze_data: Option<Res<MazeData>>,
    config: Res<RLConfig>,
    mut agent: ResMut<RLAgent>,
) {
    if brain_state.mode != BrainMode::ReinforcementLearning {
        return;
    }

    for event in lidar_events.read() {
        if let Ok((transform, movement_state, mut commands)) = mouse_query.get_mut(event.entity) {
            if movement_state.is_moving() {
                continue;
            }

            let pos = transform.translation.truncate();

            // Check if goal reached
            if let Some(maze) = &maze_data {
                let dist = pos.distance(maze.goal_world_pos());
                if dist < 8.0 {
                    info!("🎉 [RL] GOAL REACHED! Steps: {}, Reward: {:.1}", 
                          agent.episode_steps, agent.total_reward);
                    return;
                }
            }

            let observation = get_observation(&event.scan, transform);
            let action = select_action(&observation, &config);
            let reward = calculate_reward(transform, movement_state, &maze_data);
            
            agent.total_reward += reward;
            agent.episode_steps += 1;

            info!("[RL] Step {}: Action {:?}, Reward: {:.2}, Total: {:.2}", 
                  agent.episode_steps, action, reward, agent.total_reward);

            commands.move_in(action);
        }
    }
}

fn get_observation(scan: &crate::lidar::LiDARScan, transform: &Transform) -> Vec<f32> {
    let mut obs = scan.distances.clone();
    obs.push(transform.translation.x);
    obs.push(transform.translation.y);
    obs.push(transform.rotation.to_euler(bevy::math::EulerRot::ZYX).0);
    obs
}

fn select_action(observation: &[f32], config: &RLConfig) -> Direction {
    
    
    if rand::rng().random::<f32>() < config.exploration_rate {
        // Random action - use `random_range()` instead of `gen_range()`
        match rand::rng().random_range(0..4) {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            _ => Direction::Right,
        }
    } else {
        // Greedy: move towards most open direction
        let front_idx = observation.len() / 2;
        let max_idx = observation[..observation.len()-3]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(front_idx);
        
        let angle_ratio = max_idx as f32 / (observation.len() - 3) as f32;
        
        match angle_ratio {
            r if r < 0.25 => Direction::Right,
            r if r < 0.5 => Direction::Up,
            r if r < 0.75 => Direction::Left,
            _ => Direction::Down,
        }
    }
}

fn calculate_reward(
    transform: &Transform,
    movement_state: &MovementState,
    maze_data: &Option<Res<MazeData>>,
) -> f32 {
    let mut reward = -0.1;
    
    if let Some(maze) = maze_data {
        let pos = transform.translation.truncate();
        let dist_to_goal = pos.distance(maze.goal_world_pos());
        
        reward += 10.0 / (dist_to_goal + 1.0);
        
        if dist_to_goal < 8.0 {
            reward += 1000.0;
        }
        
        if movement_state.target_position.is_none() {
            reward -= 1.0;
        }
    }
    
    reward
}