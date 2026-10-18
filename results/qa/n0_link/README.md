# N0 18 Oct: tests for ipanema/link.py (worker, $0, local only)

link.py turns a follow-cam frame into pitch metres through the same-moment panorama frame (used by modal_app's follow-cam/panorama proof). It had no tests.

- tests/test_link.py, 7 tests on a made-up textured pitch with a known panorama camera and a known zoomed follow-cam view:
  the recovered view matches the truth to < 0.5 m over the whole frame (residual 0.004 m, 64 ground inliers);
  an unrelated or blank frame gives no calibration with a reason; panorama points off the pitch are refused;
  best_frame picks the frame with the most ground matches and returns (-1, None, None) when none works.
- One small fault fixed: `match()` returned empty arrays of shape (0,) instead of (0, 2) when no feature passed the ratio test (the test fails on the old code). The calibration itself never hit this (it stops at < 12 matches first), so nothing in the app changes.
- grid_check.jpg: red = true pitch grid in the follow-cam frame, yellow = link.py's grid. By eye they lie on top of each other everywhere.
