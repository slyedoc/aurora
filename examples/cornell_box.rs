//! The Cornell box: the classic 1984 measurement scene (Cornell's published geometry, cm to
//! metres), lit only by the ceiling panel. Red wall left, green wall right, everything else
//! white, a short and a tall block. No sky: every bounce of light comes from the panel, so
//! the colour bleeding on the blocks and the soft shadows are the path tracer's, not an
//! environment's.
//!
//!   AUTO_SCREENSHOT_MS=9000 AURORA_EXIT_SECS=11 cargo run --example cornell_box

use bevy::{
    asset::RenderAssetUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use bevy_aurora::{
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    AuroraDefaultPlugins,
    sky::Sky,
    util::{ScreenshotExt, TimeoutAppExt},
};

/// Panel brightness in nits (cd/m^2): a bright ceiling light panel. The camera's auto
/// exposure meters the box, so the absolute value only sets the ratio to the walls.
const LIGHT_NITS: f32 = 60_000.0;

fn main() {
    App::new()
        .add_plugins((
            AuroraDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .add_systems(Startup, setup)
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .run();
}

/// A vertex of the published data (cm), moved into bevy's frame: the original camera sits
/// on -z looking down +z with the red wall on its left, so the scene is turned half a turn
/// about y and the camera goes on +z looking down -z, red wall still on the left.
fn p(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(-x, y, -z) * 0.01
}

/// One quad, `a b c d` in order round the edge, its normal facing `toward`. The walls face
/// the middle of the room, the blocks face away from their own centres.
fn quad(meshes: &mut Assets<Mesh>, corners: [Vec3; 4], toward: Vec3) -> Handle<Mesh> {
    let [a, b, c, d] = corners;
    let mut normal = (b - a).cross(c - a).normalize();
    let centre = (a + b + c + d) * 0.25;
    let mut indices = vec![0u32, 1, 2, 0, 2, 3];
    if normal.dot(toward - centre) < 0.0 {
        normal = -normal;
        indices.reverse();
    }
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        corners.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![normal.to_array(); 4])
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    )
    .with_inserted_indices(Indices::U32(indices));
    meshes.add(mesh)
}

/// A block from its top face (published order) down to the floor.
fn block(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<AuroraMaterial>,
    top: [Vec3; 4],
) {
    let height = top[0].y;
    let centre = (top[0] + top[1] + top[2] + top[3]) * 0.25 - Vec3::Y * height * 0.5;
    let away = |c: Vec3| c + (c - centre) * 10.0;
    let mut faces = vec![top];
    for i in 0..4 {
        let a = top[i];
        let b = top[(i + 1) % 4];
        faces.push([a, b, b - Vec3::Y * height, a - Vec3::Y * height]);
    }
    for face in faces {
        let face_centre = (face[0] + face[1] + face[2] + face[3]) * 0.25;
        commands.spawn((
            Mesh3d(quad(meshes, face, away(face_centre))),
            AuroraMaterial3d(material.clone()),
        ));
    }
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    commands.insert_resource(Sky::Color {
        radiance: Vec3::ZERO,
    });

    // The published camera: 278, 273, -800 looking down +z, 39.3 degrees for a square
    // image; a touch wider here for a landscape window.
    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 39.3_f32.to_radians(),
            ..default()
        }),
        Transform::from_translation(p(278.0, 273.0, -800.0))
            .looking_at(p(278.0, 273.0, 0.0), Vec3::Y),
    ));

    let diffuse = |materials: &mut Assets<AuroraMaterial>, rgb: [f32; 3]| {
        materials.add(AuroraMaterial {
            base_color: Color::linear_rgb(rgb[0], rgb[1], rgb[2]),
            perceptual_roughness: 1.0,
            ..default()
        })
    };
    // Cornell's measured reflectances, averaged over the visible band.
    let white = diffuse(&mut materials, [0.73, 0.73, 0.73]);
    let red = diffuse(&mut materials, [0.65, 0.05, 0.05]);
    let green = diffuse(&mut materials, [0.12, 0.45, 0.15]);
    let light = materials.add(AuroraMaterial {
        base_color: Color::BLACK,
        emissive: LinearRgba::rgb(1.0, 0.85, 0.6) * LIGHT_NITS,
        ..default()
    });

    let room = p(278.0, 274.4, 279.6);
    let walls: [([Vec3; 4], &Handle<AuroraMaterial>); 5] = [
        // Floor.
        (
            [
                p(552.8, 0.0, 0.0),
                p(0.0, 0.0, 0.0),
                p(0.0, 0.0, 559.2),
                p(549.6, 0.0, 559.2),
            ],
            &white,
        ),
        // Ceiling.
        (
            [
                p(556.0, 548.8, 0.0),
                p(556.0, 548.8, 559.2),
                p(0.0, 548.8, 559.2),
                p(0.0, 548.8, 0.0),
            ],
            &white,
        ),
        // Back wall.
        (
            [
                p(549.6, 0.0, 559.2),
                p(0.0, 0.0, 559.2),
                p(0.0, 548.8, 559.2),
                p(556.0, 548.8, 559.2),
            ],
            &white,
        ),
        // Green wall (x = 0, on the camera's right).
        (
            [
                p(0.0, 0.0, 559.2),
                p(0.0, 0.0, 0.0),
                p(0.0, 548.8, 0.0),
                p(0.0, 548.8, 559.2),
            ],
            &green,
        ),
        // Red wall (x = 556, on the camera's left).
        (
            [
                p(552.8, 0.0, 0.0),
                p(549.6, 0.0, 559.2),
                p(556.0, 548.8, 559.2),
                p(556.0, 548.8, 0.0),
            ],
            &red,
        ),
    ];
    for (corners, material) in walls {
        commands.spawn((
            Mesh3d(quad(&mut meshes, corners, room)),
            AuroraMaterial3d(material.clone()),
        ));
    }

    // The light panel, a hair below the ceiling so the two never coincide.
    commands.spawn((
        Mesh3d(quad(
            &mut meshes,
            [
                p(343.0, 548.0, 227.0),
                p(343.0, 548.0, 332.0),
                p(213.0, 548.0, 332.0),
                p(213.0, 548.0, 227.0),
            ],
            room,
        )),
        AuroraMaterial3d(light),
    ));

    // Short block.
    block(
        &mut commands,
        &mut meshes,
        &white,
        [
            p(130.0, 165.0, 65.0),
            p(82.0, 165.0, 225.0),
            p(240.0, 165.0, 272.0),
            p(290.0, 165.0, 114.0),
        ],
    );
    // Tall block.
    block(
        &mut commands,
        &mut meshes,
        &white,
        [
            p(423.0, 330.0, 247.0),
            p(265.0, 330.0, 296.0),
            p(314.0, 330.0, 456.0),
            p(472.0, 330.0, 406.0),
        ],
    );
}
