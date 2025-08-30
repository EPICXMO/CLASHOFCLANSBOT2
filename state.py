from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class GameState:
    elixir: int = 0
    hand_cards: List[str] = field(default_factory=lambda: ["card0", "card1", "card2", "card3"])  # placeholders
    tower_hps: List[int] = field(default_factory=lambda: [100, 100, 100])
    enemy_tower_hps: List[int] = field(default_factory=lambda: [100, 100, 100])
    gold: int = 0
    daily_wins: int = 0
    chest_slots: List[Tuple[int, int, int, int]] = field(default_factory=list)
    ui: Dict = field(default_factory=dict)

    def update_from_frame(self, frame: np.ndarray, ui: Dict) -> None:
        self.ui = ui
        self.elixir = int(ui.get("elixir", 0))
        self.gold = int(ui.get("gold", 0))
        self.daily_wins = int(ui.get("daily_wins", 0))
        self.hand_cards = list(ui.get("hand_cards", self.hand_cards))
        self.tower_hps = list(ui.get("tower_hps", self.tower_hps))
        self.enemy_tower_hps = list(ui.get("enemy_tower_hps", self.enemy_tower_hps))
        self.chest_slots = list(ui.get("yolo", {}).get("chest", []))

    def to_dict(self) -> Dict:
        return asdict(self)
