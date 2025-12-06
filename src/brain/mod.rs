mod left_hand_on_wall;
mod manual;
mod reinforcement_learning;

use bevy::prelude::*;
use left_hand_on_wall::LeftHandOnWallConfig;
use reinforcement_learning::{RLAgent, RLConfig};

pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BrainState>()
            .init_resource::<LeftHandOnWallConfig>()
            .init_resource::<RLConfig>()
            .init_resource::<RLAgent>()
            .add_systems(Startup, setup_ui)
            .add_systems(Update, toggle_brain_mode)
            .add_systems(Update, update_mode_display)
            .add_systems(Update, manual::manual_control)
            .add_systems(Update, left_hand_on_wall::execute_left_hand_on_wall)
            .add_systems(Update, reinforcement_learning::execute_rl_agent);
    }
}

#[derive(Resource, Default)]
pub struct BrainState {
    pub mode: BrainMode,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainMode {
    #[default]
    Manual,
    LeftHandOnWall,
    ReinforcementLearning,
}

impl BrainMode {
    pub fn next(self) -> Self {
        match self {
            BrainMode::Manual => BrainMode::LeftHandOnWall,
            BrainMode::LeftHandOnWall => BrainMode::ReinforcementLearning,
            BrainMode::ReinforcementLearning => BrainMode::Manual,
        }
    }
}

impl std::fmt::Display for BrainMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrainMode::Manual => write!(f, "Manual"),
            BrainMode::LeftHandOnWall => write!(f, "Left-Hand-on-Wall"),
            BrainMode::ReinforcementLearning => write!(f, "Reinforcement Learning"),
        }
    }
}

fn toggle_brain_mode(keyboard: Res<ButtonInput<KeyCode>>, mut brain_state: ResMut<BrainState>) {
    if keyboard.just_pressed(KeyCode::Space) {
        brain_state.mode = brain_state.mode.next();
        info!("[BRAIN] Mode switched to: {:?}", brain_state.mode);
    }
}

#[derive(Component)]
struct ModeText;

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Mode: Manual"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        ModeText,
    ));
}

fn update_mode_display(brain_state: Res<BrainState>, mut query: Query<&mut Text, With<ModeText>>) {
    if brain_state.is_changed() {
        for mut text in query.iter_mut() {
            **text = format!("Mode: {:?}", brain_state.mode);
        }
    }
}
