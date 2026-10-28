# N0-28d runner check (2026-10-28 13:20 UTC)

OpenCV the workflow installs: **4.10.0**

## Stall test (local server goes silent on the header read and on a mid-file read)

| OpenCV | frames read | open s | total s | requests to the server (stalled) |
|---|---|---|---|---|
| 4.10.0 (installed) | 300/300 | 10.1 | 20.4 | 6 (2) |
| 4.11.0 (old 4.11.0.86) | 0/300 | 120.0 | 120.0 | 2 (1) |

## Real R2 (20-s piece from 25:00, as the nightly check)

| round | match | opens | piece | seconds |
|---|---|---|---|---|
| 1 | p15u-vs-djursholm-2026-09-26 | 1 | 599/599 | 8.1 |
| 1 | p15u-vs-spanga-2026-09-25 | 1 | 599/599 | 9.0 |
| 2 | p15u-vs-djursholm-2026-09-26 | 1 | 599/599 | 8.7 |
| 2 | p15u-vs-spanga-2026-09-25 | 1 | 599/599 | 12.5 |
| 3 | p15u-vs-djursholm-2026-09-26 | 1 | 599/599 | 8.0 |
| 3 | p15u-vs-spanga-2026-09-25 | 1 | 599/599 | 8.6 |
