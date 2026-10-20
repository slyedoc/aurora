//! Exit the app after a fixed duration — headless/agent runs and smoke tests.
//!
//! Exits through `AppExit`, so the render plugin's ordered GPU teardown (queue idle,
//! `TeardownSchedule`, NGX release) runs as on any normal quit. Prefer this over a shell
//! `timeout`, whose SIGTERM skips that teardown entirely.

use bevy::prelude::*;

#[derive(Resource)]
struct Timeout {
    timer: Timer,
}

pub trait TimeoutAppExt {
    /// Exit after `duration` secs (none: run on). `AURORA_EXIT_SECS` overrides it.
    fn add_timeout_exit(&mut self, duration: Option<f32>) -> &mut Self;
}

impl TimeoutAppExt for App {
    fn add_timeout_exit(&mut self, duration: Option<f32>) -> &mut Self {
        let secs = std::env::var("AURORA_EXIT_SECS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .or(duration);
        if let Some(secs) = secs {
            self.insert_resource(Timeout {
                timer: Timer::from_seconds(secs, TimerMode::Once),
            })
            .add_systems(Update, exit_on_timeout);
        }
        self
    }
}

fn exit_on_timeout(
    time: Res<Time>,
    mut timeout: ResMut<Timeout>,
    mut exit: MessageWriter<AppExit>,
) {
    if timeout.timer.tick(time.delta()).just_finished() {
        info!("timeout reached, exiting");
        exit.write(AppExit::Success);
    }
}
