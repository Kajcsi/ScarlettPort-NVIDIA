//! A cap on the frames shown a second: `SCARLETT_SHOWN_CAP=<frames a second>`
//! (unset or 0: none).
//!
//! The title renders as fast as the layer lets it, and how fast that is
//! changes from one frame to the next. Shown as they come, frames of 50 and
//! 90 milliseconds side by side look worse than frames of 67 milliseconds
//! all the time, and frame generation spaces its frames between two rendered
//! ones by the time between them. So a present is held back until its turn:
//! one every `shown for each rendered / cap` seconds, where frame generation
//! shows several frames for each rendered one (45 shown a second at three
//! for one: a present every 66.7 ms). A frame that comes late is shown at
//! once and the turns start anew from it: nothing is caught up.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The cap, in frames shown a second, if one is set.
fn cap() -> Option<f64> {
    static CAP: OnceLock<Option<f64>> = OnceLock::new();
    *CAP.get_or_init(|| {
        let cap = std::env::var("SCARLETT_SHOWN_CAP").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|c| c.is_finite() && *c >= 1.0);
        if let Some(cap) = cap {
            tracing::info!(cap, "the frames shown a second are capped (SCARLETT_SHOWN_CAP): presents wait for their turn");
        }
        cap
    })
}

/// The time between two presents for `cap` frames shown a second with
/// `shown` frames shown for each one presented.
fn interval(cap: f64, shown: u32) -> Duration {
    Duration::from_secs_f64(f64::from(shown.max(1)) / cap)
}

/// When the present after one at `now` is due, given when this one was
/// (`due`, none for the first): on the grid while presents keep up, anew
/// from `now` after one that came late.
fn next(due: Option<Instant>, now: Instant, interval: Duration) -> Instant {
    match due {
        Some(due) if due >= now => due + interval,
        _ => now + interval,
    }
}

/// Before a present that shows `shown` frames (1 without frame generation):
/// waits until its turn.
pub(crate) fn before_present(shown: u32) {
    let Some(cap) = cap() else { return };
    static DUE: Mutex<Option<Instant>> = Mutex::new(None);
    let mut due = DUE.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    if let Some(at) = *due
        && at > now
    {
        let _waiting = crate::timeline::span("waiting for the present's turn (SCARLETT_SHOWN_CAP)");
        // Asleep for most of it, then awake for the last millisecond and a
        // half: a sleep ends when the system's clock says so.
        let left = at - now;
        if left > Duration::from_micros(1500) {
            std::thread::sleep(left - Duration::from_micros(1500));
        }
        while Instant::now() < at {
            std::hint::spin_loop();
        }
    }
    *due = Some(next(*due, now, interval(cap, shown)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_present_every_third_of_the_cap_with_three_frames_for_one() {
        assert_eq!(interval(45.0, 3), Duration::from_secs_f64(3.0 / 45.0));
        assert_eq!(interval(45.0, 1), Duration::from_secs_f64(1.0 / 45.0));
        assert_eq!(interval(30.0, 0), Duration::from_secs_f64(1.0 / 30.0));
    }

    #[test]
    fn turns_keep_their_grid_until_a_frame_comes_late() {
        let step = Duration::from_millis(60);
        let start = Instant::now();
        // The first: its follower is one step on.
        assert_eq!(next(None, start, step), start + step);
        // In time (it waited until its turn): the grid goes on.
        let due = start + step;
        assert_eq!(next(Some(due), start + Duration::from_millis(20), step), due + step);
        // Late: anew from where it came, nothing caught up.
        let late = due + Duration::from_millis(200);
        assert_eq!(next(Some(due), late, step), late + step);
    }
}
