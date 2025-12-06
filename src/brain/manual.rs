use super::{BrainMode, BrainState};
use crate::mouse::{Direction, Mouse, MouseCommands};
use bevy::prelude::*;

pub fn manual_control(
    keyboard: Res<ButtonInput<KeyCode>>,
    brain_state: Res<BrainState>,
    mut mice: Query<(&Transform, &mut MouseCommands), With<Mouse>>,
) {
    if brain_state.mode != BrainMode::Manual {
        return;
    }

    let direction = [
        (
            keyboard.just_pressed(KeyCode::ArrowUp) || keyboard.just_pressed(KeyCode::KeyW),
            Direction::Up,
        ),
        (
            keyboard.just_pressed(KeyCode::ArrowDown) || keyboard.just_pressed(KeyCode::KeyS),
            Direction::Down,
        ),
        (
            keyboard.just_pressed(KeyCode::ArrowLeft) || keyboard.just_pressed(KeyCode::KeyA),
            Direction::Left,
        ),
        (
            keyboard.just_pressed(KeyCode::ArrowRight) || keyboard.just_pressed(KeyCode::KeyD),
            Direction::Right,
        ),
    ]
    .iter()
    .find(|(pressed, _)| *pressed)
    .map(|(_, dir)| *dir);

    if let Some(dir) = direction {
        for (transform, mut commands) in &mut mice {
            let pos = transform.translation;
            info!(
                "[MANUAL] Key pressed: {:?} at position ({:.1}, {:.1}, {:.1})",
                dir, pos.x, pos.y, pos.z
            );
            commands.move_in(dir);
        }
    }
}
