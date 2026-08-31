//! Small app-level helpers shared by aurora tools, examples, and games.

pub mod screenshot;
pub mod timeout;

pub use screenshot::{ScreenshotExt, ScreenshotRequests};
pub use timeout::TimeoutAppExt;
