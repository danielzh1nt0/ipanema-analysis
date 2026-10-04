# Scoreboard — one line per run

Ball check = correct picks on held-out labelled frames; ceiling = frames where the ball was among the candidates.

| when (UTC) | commit | match | version | ball check | ceiling | held-out (train) | possession | events | players/frame | candidates |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-21 17:15 | f965aec | SFKBP1109_s1200 | 0.21.1 | 18/34 | 25 | (trained in the earlier, cut-off run) | OK | withheld | {'A': 5.0, 'B': 8.0} | WASB-only candidates, 2.9/frame · with YOLO: 18/34, ceiling 30 |
| 2026-09-21 21:05 | da1d423 | SFKBP1109_full | ? | 22/34 | 25 | — | withheld | withheld | {'A': 5.0, 'B': 5.0} |  · other picker v1: 19/34 · picker test SFKBP1109: v2 41/59, other 31/59 · picker test SFKBP1109b: v2 91/134, other 56/134 |
| 2026-09-21 23:53 | dbfce07 | SFKBP1109_full | ? | 21/34 | 24 | — | withheld | withheld | {'A': 5.0, 'B': 6.0} |  · other picker v1: 18/34 · picker test SFKBP1109: v2 36/59, other 26/59 · picker test SFKBP1109b: v2 70/134, other 38/134 |
| 2026-09-22 00:46 | c87fe53 | SFKBP1109_full | ? | 21/34 | 24 | — | withheld | withheld | {'A': 5.0, 'B': 6.0} |  · other picker v1: 18/34 · picker test SFKBP1109: v2 36/59, other 26/59 · picker test SFKBP1109b: v2 70/134, other 38/134 |
| 2026-09-22 05:53 | 97916c0 | SFKBP1109_full | ? | 21/34 | 24 | — | withheld | withheld | {'A': 5.0, 'B': 6.0} |  · other picker v1: 18/34 · picker test SFKBP1109: v2 34/59, other 24/59 · picker test SFKBP1109b: v2 65/134, other 37/134 |
| 2026-09-22 06:05 | 8a3db83 | SFKBP1109_full | ? | 12/34 | 12 | — | withheld | withheld | {'A': 6.0, 'B': 6.0} |  · other picker v1: 10/34 · picker test SFKBP1109: v2 12/59, other 7/59 · picker test SFKBP1109b: v2 22/134, other 19/134 |
| 2026-09-22 18:58 | e86aa0d | p15u-vs-bp-2026-09-22-2000_full | 0.27.0 | —/— | — | — | withheld | withheld | — |  |
| 2026-09-26 15:28 | fbbdb8a | SFKBP1109_s1200 | 0.27.0 | 22/34 | 25 | — | withheld | withheld | {'A': 5.0, 'B': 8.0} | WASB-only candidates, 2.9/frame · with YOLO: 16/34, ceiling 30 · other picker v1: 17/34 |
| 2026-09-26 17:04 | 0fe2583 | SFKBP1109_s1200 | 0.27.0 | 20/34 | 19 | — | withheld | withheld | {'A': 5.0, 'B': 8.0} |  · with YOLO: 16/34, ceiling 29 · other picker v1: 16/34 |
| 2026-09-26 19:55 | bb11ea0 | SFKBP1109_s1200 | 0.27.0 | —/— | — | — | withheld | withheld | — |  |
| 2026-09-27 16:51 | 0a8efe2 | SFKBP1109_s1200 | 0.27.0 | 23/34 | 27 | — | OK | withheld | {'A': 6.0, 'B': 8.0} |  · with YOLO: 14/34, ceiling 31 · other picker v1: 15/34 |
| 2026-09-28 06:04 | 96de549 | SFKBP1109_s1200 | 0.27.0 | 22/34 | 27 | — | OK | withheld | {'A': 6.0, 'B': 8.0} |  · with YOLO: 15/34, ceiling 30 · other picker v1: 16/34 |
| 2026-09-29 10:52 | b87b48a | SFKBP1109_s1200 | 0.27.0 | 22/34 | 27 | — | OK | withheld | {'A': 8.0, 'B': 7.0} |  · with YOLO: 15/34, ceiling 30 · other picker v1: 17/34 |
| 2026-09-29 16:26 | b0fc4df | SFKBP1109_s1200 | 0.27.0 | 29/34 | 34 | — | OK | withheld | {'A': 8.0, 'B': 7.0} |  · with YOLO: 14/34, ceiling 34 · other picker v1: 16/34 |
| 2026-09-29 21:31 | 0f9e463 | p15u-vs-aik-2026-09-21-bd09_s2520 | 0.27.0 | —/— | — | — | withheld | withheld | {'A': 7.0, 'B': 7.0} |  |
| 2026-10-01 17:16 | a9135b5 | p15u-vs-aik-2026-09-21-bd09_s2520 | 0.27.0 | 26/39 | 39 | — | OK | withheld | {'A': 7.0, 'B': 7.0} |  · other picker v1: 18/39 |
| 2026-10-01 18:24 | fb4ac50 | p15u-vs-aik-2026-09-21-bd09_s2520 | 0.27.0 | 26/39 | 39 | — | OK | withheld | {'A': 7.0, 'B': 7.0} |  · other picker v1: 18/39 |
| 2026-10-01 18:44 | 5122b6c | SFKBP1109_s1200 | 0.27.0 | 28/34 | 34 | — | OK | withheld | {'A': 8.0, 'B': 7.0} |  · with YOLO: 14/34, ceiling 34 · other picker v1: 16/34 |
| 2026-10-01 19:10 | 031452c | SFKBP1109_s1200 | 0.27.0 | 29/34 | 34 | — | OK | withheld | {'A': 8.0, 'B': 7.0} |  · with YOLO: 14/34, ceiling 34 · other picker v1: 16/34 |
| 2026-10-01 19:38 | 124afb3 | p15u-vs-aik-2026-09-21-bd09_s2520 | 0.27.0 | 30/39 | 39 | — | OK | withheld | {'A': 7.0, 'B': 7.0} |  · other picker v1: 18/39 |
| 2026-10-01 23:18 | 12bcccd | SFKBP1109_full | ? | —/— | — | — | withheld | withheld | — |  |
| 2026-10-02 00:14 | 5ffaccf | SFKBP1109_full | ? | 13/34 | 14 | — | withheld | withheld | {'A': 9.0, 'B': 8.0} |  · other picker v1: 10/34 · picker test SFKBP1109: v2 16/59, other 9/59 · picker test SFKBP1109b: v2 31/134, other 20/134 |
| 2026-10-02 00:34 | 4a618c0 | SFKBP1109_full | 0.27.0 | —/— | — | — | withheld | withheld | — |  |
| 2026-10-02 01:07 | 6662137 | SFKBP1109_full | ? | 25/34 | 30 | — | OK | withheld | {'A': 8.0, 'B': 7.0} |  · other picker v1: 15/34 · picker test SFKBP1109: v2 40/59, other 22/59 · picker test SFKBP1109b: v2 87/134, other 50/134 |
| 2026-10-02 03:32 | 163b227 | p15u-vs-aik-2026-09-21-bd09_full | ? | 7/39 | 9 | — | withheld | withheld | {'A': 6.0, 'B': 7.0} |  · other picker v1: 5/39 · picker test p15u-vs-aik-2026-09-21-bd09: v2 4/62, other 1/62 |
| 2026-10-02 03:49 | 57cdcff | p15u-vs-aik-2026-09-21-bd09_full | ? | 26/39 | 39 | — | OK | withheld | {'A': 6.0, 'B': 6.0} |  · other picker v1: 13/39 · picker test p15u-vs-aik-2026-09-21-bd09: v2 40/62, other 22/62 |
| 2026-10-02 06:31 | fa7f2f0 | SFKBP1109_full | ? | 28/34 | 33 | — | OK | withheld | {'A': 8.0, 'B': 7.0} |  · other picker v1: 17/34 · picker test SFKBP1109: v2 48/59, other 26/59 · picker test SFKBP1109b: v2 100/134, other 59/134 |
| 2026-10-02 07:42 | c4e4a75 | SFKBP1109_full | ? | 28/34 | 33 | — | OK | withheld | {'A': 8.0, 'B': 7.0} |  · other picker v1: 17/34 · picker test SFKBP1109: v2 48/59, other 26/59 · picker test SFKBP1109b: v2 100/134, other 59/134 |
| 2026-10-02 08:20 | d569859 | SFKBP1109_full | ? | 28/34 | 33 | — | OK | withheld | {'A': 7.0, 'B': 7.0} |  · other picker v1: 17/34 · picker test SFKBP1109: v2 47/59, other 26/59 · picker test SFKBP1109b: v2 97/134, other 59/134 |
| 2026-10-02 08:44 | 5d51c3b | p15u-vs-aik-2026-09-21-bd09_full | ? | 26/39 | 39 | — | OK | withheld | {'A': 6.0, 'B': 6.0} |  · other picker v1: 13/39 · picker test p15u-vs-aik-2026-09-21-bd09: v2 40/62, other 22/62 |
| 2026-10-02 19:21 | 7dc2b7c | SFKBP1109_full | ? | 28/34 | 33 | — | OK | withheld | {'A': 7.0, 'B': 7.0} |  · other picker v1: 17/34 · picker test SFKBP1109: v2 23/59, other 10/59 · picker test SFKBP1109b: v2 53/134, other 31/134 |
| 2026-10-02 19:39 | a391e06 | p15u-vs-aik-2026-09-21-bd09_full | ? | 26/39 | 39 | — | OK | withheld | {'A': 5.0, 'B': 6.0} |  · other picker v1: 13/39 · picker test p15u-vs-aik-2026-09-21-bd09: v2 19/62, other 9/62 |
| 2026-10-03 05:23 | 4b5bab5 | p15u-vs-aik-2026-09-21-bd09_full | ? | 26/39 | 39 | — | OK | withheld | {'A': 5.0, 'B': 6.0} |  · other picker v1: 13/39 · picker test p15u-vs-aik-2026-09-21-bd09: v2 18/62, other 9/62 |
| 2026-10-04 03:50 | 4c41620 | p15u-vs-vallentuna-2026-10-03-6cce_full | 0.27.0 | —/— | — | — | withheld | withheld | — |  |
| 2026-10-04 04:19 | 4283726 | p15u-vs-vallentuna-2026-10-03-6cce_full | ? | —/— | — | — | withheld | withheld | {'A': 3.0, 'B': 11.0} |  |
| 2026-10-04 09:27 | 00bfd9a | p15u-vs-vallentuna-2026-10-03-6cce_full | ? | —/— | — | — | withheld | withheld | {'A': 4.0, 'B': 5.0} |  |
| 2026-10-04 10:05 | 6dde114 | SFKBP1109_full | ? | 28/34 | 33 | — | OK | withheld | {'A': 7.0, 'B': 7.0} |  · other picker v1: 17/34 · picker test SFKBP1109: v2 23/59, other 10/59 · picker test SFKBP1109b: v2 53/134, other 31/134 |
| 2026-10-04 10:26 | 3ae26bb | p15u-vs-aik-2026-09-21-bd09_full | ? | 26/39 | 39 | — | OK | withheld | {'A': 5.0, 'B': 6.0} |  · other picker v1: 13/39 · picker test p15u-vs-aik-2026-09-21-bd09: v2 18/62, other 9/62 |
| 2026-10-04 11:44 | bdc9fbb | p15u-vs-vallentuna-2026-10-03-6cce_full | ? | —/— | — | — | withheld | withheld | {'A': 4.0, 'B': 5.0} |  |
| 2026-10-04 12:12 | f7c79d8 | p15u-vs-vallentuna-2026-10-03-6cce_full | ? | —/— | — | — | withheld | withheld | {'A': 4.0, 'B': 5.0} |  |
| 2026-10-04 14:41 | f83ff32 | p15u-vs-vallentuna-2026-10-03-6cce_full | ? | —/— | — | — | withheld | withheld | {'A': 4.0, 'B': 5.0} |  |
| 2026-10-04 16:08 | a485ff2 | SFKBP1109_full | ? | 28/34 | 33 | — | OK | withheld | {'A': 7.0, 'B': 7.0} |  · other picker v1: 17/34 · picker test SFKBP1109: v2 23/59, other 10/59 · picker test SFKBP1109b: v2 53/134, other 31/134 |
| 2026-10-04 16:47 | f99e87d | SFKBP1109_full | ? | 28/34 | 33 | — | OK | withheld | {'A': 7.0, 'B': 7.0} |  · other picker v1: 17/34 · picker test SFKBP1109: v2 23/59, other 10/59 · picker test SFKBP1109b: v2 53/134, other 31/134 |
