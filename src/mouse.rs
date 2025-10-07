use avian2d::prelude::*;
use bevy::prelude::*;

pub struct MousePlugin;

impl Plugin for MousePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera);
        app.add_systems(Update, (camera_follow_mouse, camera_zoom));
        app.add_systems(Update, mouse_manual_control);
    }
}

pub const MOUSE_RADIUS: f32 = 4.0;
const MOUSE_SPEED: f32 = 200.0;

#[derive(Component)]
pub struct Mouse;

fn mouse_manual_control(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut mouse_query: Query<&mut LinearVelocity, With<Mouse>>,
) {
    for mut velocity in mouse_query.iter_mut() {
        let mut direction = Vec2::ZERO;

        if keyboard_input.pressed(KeyCode::ArrowUp) || keyboard_input.pressed(KeyCode::KeyW) {
            direction.y += 1.0;
        }
        if keyboard_input.pressed(KeyCode::ArrowDown) || keyboard_input.pressed(KeyCode::KeyS) {
            direction.y -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::ArrowLeft) || keyboard_input.pressed(KeyCode::KeyA) {
            direction.x -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::ArrowRight) || keyboard_input.pressed(KeyCode::KeyD) {
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

#[derive(Component)]
pub struct FollowCamera;

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

fn camera_zoom(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_query: Query<&mut Projection, With<FollowCamera>>,
) {
    if let Ok(mut projection) = camera_query.single_mut() {
        const ZOOM_SPEED: f32 = 0.1;
        const MIN_SCALE: f32 = 0.2;
        const MAX_SCALE: f32 = 5.0;

        // Match on the projection type to handle orthographic projection
        if let Projection::Orthographic(ortho) = projection.as_mut() {
            if keyboard.pressed(KeyCode::KeyZ) {
                // Zoom in (decrease scale)
                ortho.scale = (ortho.scale - ZOOM_SPEED).max(MIN_SCALE);
            }
            if keyboard.pressed(KeyCode::KeyX) {
                // Zoom out (increase scale)
                ortho.scale = (ortho.scale + ZOOM_SPEED).min(MAX_SCALE);
            }
        }
    }
}

#[derive(Bundle)]
pub struct MouseBundle {
    pub mouse: Mouse,
    pub rigid_body: RigidBody,
    pub collider: Collider,
    pub mesh: Mesh2d,
    pub material: MeshMaterial2d<ColorMaterial>,
    pub transform: Transform,
    pub friction: Friction,
    pub restitution: Restitution,
}

impl MouseBundle {
    pub fn new(
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<ColorMaterial>,
        position: Vec3,
    ) -> Self {
        let mouse_circle = Circle::new(MOUSE_RADIUS);
        let mesh = meshes.add(mouse_circle);
        let material = materials.add(Color::srgb(1.0, 0.0, 0.0));

        Self {
            mouse: Mouse,
            rigid_body: RigidBody::Dynamic,
            collider: Collider::circle(MOUSE_RADIUS),
            mesh: Mesh2d(mesh),
            material: MeshMaterial2d(material),
            transform: Transform::from_translation(position)
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
            friction: Friction::ZERO.with_combine_rule(CoefficientCombine::Min),
            restitution: Restitution::ZERO.with_combine_rule(CoefficientCombine::Min),
        }
    }
}
