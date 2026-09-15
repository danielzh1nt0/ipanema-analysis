import os
from dataclasses import dataclass, field

@dataclass
class Settings:
    root: str = "/content/drive/MyDrive/match_analysis"      # Drive project folder
    sports_dir: str = "/content/sports"                       # roboflow/sports checkout
    work: str = "/content/work"                               # fast local scratch
    process_stride: int = 1                                   # analyse every Nth frame (1 = all)
    conf_player: float = 0.3
    conf_ball: float = 0.05
    ball_tiles: tuple = (3, 2)
    kp_conf: float = 0.5
    carrier_r: float = 2.5
    press_r: float = 2.0
    near_r: float = 5.0
    lane_half: float = 1.5
    max_lane: float = 45.0
    hold_s: float = 0.4
    def path(self, *p): return os.path.join(self.root, *p)
    @property
    def weights(self):
        d = os.path.join(self.sports_dir, "examples/soccer/data")
        ft = os.path.join(self.root, "models", "ball_finetuned.pt"); pt = os.path.join(self.root, "models", "pitch_finetuned.pt")
        return {"player": f"{d}/football-player-detection.pt", "ball": ft if os.path.exists(ft) else f"{d}/football-ball-detection.pt", "pitch": pt if os.path.exists(pt) else f"{d}/football-pitch-detection.pt"}
