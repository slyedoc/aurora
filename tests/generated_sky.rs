//! `Sky::Generated` on the GPU: noon is blue overhead, a low sun reddens the horizon under it,
//! and with the sun well below the horizon the sky is dark.

use bevy::{asset::RenderAssetUsages, camera::RenderTarget, prelude::*, winit::WinitPlugin};
use bevy_aurora::{
    AuroraDefaultPlugins,
    camera_target::camera_target_placeholder,
    environment::{Environment, RenderEnvironments},
    render_device::RenderDevice,
    sky::{Sky, SkyAir},
    sky_gen::SkyGen,
    sky_gen::{SKY_H as ENV_H, SKY_W as ENV_W},
};
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

/// The texel looking along `d` (rt_types.slang `env_dir_to_uv`).
fn texel(sky: &[Vec4], d: Vec3) -> Vec3 {
    let d = d.normalize();
    let phi = d.x.atan2(d.z);
    let mut u = (if phi > 0.0 {
        phi
    } else {
        phi + std::f32::consts::TAU
    }) / std::f32::consts::TAU
        - 0.5;
    if u < 0.0 {
        u += 1.0;
    }
    let v = d.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    let x = ((u * ENV_W as f32) as u32).min(ENV_W - 1);
    let y = ((v * ENV_H as f32) as u32).min(ENV_H - 1);
    sky[(y * ENV_W + x) as usize].truncate()
}

fn luma(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// The light direction of a sun `elevation` degrees up in the +X direction.
fn sun_at(elevation: f32) -> Transform {
    let e = elevation.to_radians();
    let towards = Vec3::new(e.cos(), e.sin(), 0.0);
    Transform::default().looking_to(-towards, Vec3::Z)
}

fn render(app: &mut App, environment: Entity, sun: Entity, elevation: f32) -> Vec<Vec4> {
    app.world_mut().entity_mut(sun).insert(sun_at(elevation));
    let bit = built(app, environment);
    let world = app.world();
    let gen_ = world.resource::<SkyGen>();
    let rd = world.resource::<RenderDevice>();
    check_table(&gen_.read_back_table(rd, bit).expect("a table"));
    gen_.read_back(rd, bit)
        .expect("the generated sky was built")
}

/// Updates until environment's importance table is built (kernels compiled and run), then a few
/// more frames; its bit.
fn built(app: &mut App, environment: Entity) -> usize {
    let start = std::time::Instant::now();
    let bit = loop {
        app.update();
        let bit = app
            .world()
            .resource::<RenderEnvironments>()
            .bit(environment)
            .map(usize::from);
        if let Some(bit) = bit
            && app.world().resource::<SkyGen>().table(bit) != 0
        {
            break bit;
        }
        assert!(
            start.elapsed().as_secs() < 30,
            "the sky table was never built"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    for _ in 0..4 {
        app.update();
    }
    bit
}

/// raygen.slang `skySample`'s search over `cdf`, from `lo` to `hi`.
fn search(cdf: &[f32], u: f32, mut lo: usize, mut hi: usize) -> usize {
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid] <= u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The cdf runs 0 to 1 and the guided search picks the texel the full search does.
fn check_table(table: &[f32]) {
    let n = (ENV_W * ENV_H) as usize;
    let cdf = &table[..=n];
    assert_eq!(cdf[0], 0.0);
    assert_eq!(cdf[n], 1.0);
    assert!(cdf.windows(2).all(|w| w[0] <= w[1]), "a monotone cdf");
    let guide: Vec<usize> = table[2 * n + 1..]
        .iter()
        .map(|g| g.to_bits() as usize)
        .collect();
    let mut widest = 0;
    let mut state = 0x9e37_79b9u32;
    for k in 0..100_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let u = match k {
            0 => 0.0,
            1 => 1.0 - f32::EPSILON,
            2 => 1.0 / n as f32,
            _ => f32::from_bits(0x3f80_0000 | (state >> 9)) - 1.0,
        };
        let j = ((u * n as f32) as usize).min(n - 1);
        let (lo, hi) = (guide[j], (guide[j + 1] + 1).min(n));
        widest = widest.max(hi - lo);
        assert_eq!(search(cdf, u, lo, hi), search(cdf, u, 0, n), "u {u}");
    }
    println!("guide: widest search {widest} texels");
}

#[test]
fn an_hdr_sky_is_box_filtered_into_its_table_on_the_gpu() {
    let mut app = App::new();
    app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    app.finish();
    app.cleanup();

    // A flat sky of 1 with a 3x3-texel sun of 1e4.
    let (w, h) = (4 * ENV_W, 4 * ENV_H);
    let (sx, sy) = (1234u32, 321u32);
    let mut texels = vec![[1.0f32, 1.0, 1.0, 1.0]; (w * h) as usize];
    for y in sy..sy + 3 {
        for x in sx..sx + 3 {
            texels[(y * w + x) as usize] = [1.0e4, 1.0e4, 1.0e4, 1.0];
        }
    }
    let image = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytemuck::cast_slice(&texels).to_vec(),
        TextureFormat::Rgba32Float,
        RenderAssetUsages::all(),
    );
    let image = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let environment = app
        .world_mut()
        .spawn((Environment::default(), Sky::Hdr { image, scale: 1.0 }))
        .id();
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_target_placeholder(UVec2::new(64, 64)));
    app.world_mut().spawn((
        ChildOf(environment),
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        Transform::default(),
    ));

    let bit = built(&mut app, environment);
    let world = app.world();
    let gen_ = world.resource::<SkyGen>();
    let rd = world.resource::<RenderDevice>();
    let grid = gen_.read_back(rd, bit).expect("the grid");
    let total: f64 = grid.iter().map(|t| t.x as f64).sum::<f64>() * 16.0;
    let expected = (w * h) as f64 + 9.0 * (1.0e4 - 1.0);
    println!("grid energy {total} vs image {expected}");
    assert!(
        (total / expected - 1.0).abs() < 1.0e-4,
        "{total} vs {expected}"
    );
    let brightest = (0..grid.len())
        .max_by(|a, b| grid[*a].x.total_cmp(&grid[*b].x))
        .unwrap() as u32;
    assert_eq!(
        (brightest % ENV_W, brightest / ENV_W),
        (sx / 4, sy / 4),
        "the sun's grid texel"
    );
    check_table(&gen_.read_back_table(rd, bit).expect("a table"));

    bevy_aurora::ray_render_plugin::shutdown_renderer(app.world_mut());
}

#[test]
fn the_generated_sky_is_blue_at_noon_red_at_sunset_and_dark_at_night() {
    let mut app = App::new();
    app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    app.finish();
    app.cleanup();

    let environment = app
        .world_mut()
        .spawn((Environment::default(), Sky::Generated, SkyAir::default()))
        .id();
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_target_placeholder(UVec2::new(64, 64)));
    app.world_mut().spawn((
        ChildOf(environment),
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        Transform::default(),
    ));
    let sun = app
        .world_mut()
        .spawn((
            ChildOf(environment),
            DirectionalLight {
                illuminance: 100_000.0,
                ..default()
            },
            sun_at(60.0),
        ))
        .id();

    let noon = render(&mut app, environment, sun, 60.0);
    let zenith = texel(&noon, Vec3::Y);
    let horizon_away = texel(&noon, Vec3::new(-1.0, 0.05, 0.0));
    println!("noon: zenith {zenith}, horizon {horizon_away}");
    assert!(zenith.z > zenith.x * 1.5, "a blue zenith: {zenith}");
    assert!(luma(zenith) > 500.0 && luma(zenith) < 50_000.0, "{zenith}");
    assert!(
        horizon_away.x / horizon_away.z > zenith.x / zenith.z,
        "the horizon paler than the zenith: {horizon_away} vs {zenith}"
    );

    let sunset = render(&mut app, environment, sun, 2.0);
    let under_sun = texel(&sunset, Vec3::new(1.0, 0.03, 0.0));
    let zenith = texel(&sunset, Vec3::Y);
    println!("sunset: under the sun {under_sun}, zenith {zenith}");
    assert!(
        under_sun.x > under_sun.z,
        "a red horizon under the sun: {under_sun}"
    );
    assert!(
        luma(zenith) < luma(texel(&noon, Vec3::Y)),
        "dimmer than noon"
    );

    let night = render(&mut app, environment, sun, -20.0);
    let brightest = night.iter().map(|t| luma(t.truncate())).fold(0.0, f32::max);
    println!("night: brightest {brightest}");
    assert!(brightest < 1.0, "a dark night: {brightest}");
    // The airglow floor: the night zenith at `SkyAir::airglow_nits` (the sun adds nothing).
    let zenith = luma(texel(&night, Vec3::Y));
    let floor = SkyAir::default().airglow_nits;
    assert!(
        (zenith / floor - 1.0).abs() < 0.1,
        "night zenith {zenith} vs airglow {floor}"
    );

    bevy_aurora::ray_render_plugin::shutdown_renderer(app.world_mut());
}
