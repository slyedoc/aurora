//! Small app-level helpers shared by aurora tools, examples, and games.

pub mod capture;
pub mod headless;
pub mod screenshot;
pub mod timeout;
pub mod video;

pub use headless::{Headless, HeadlessTarget};
pub use screenshot::{Capture, ScreenshotExt, ScreenshotRequests};
pub use timeout::TimeoutAppExt;
pub use video::{VideoExt, VideoRecorder, video_path};
