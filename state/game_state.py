from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class GameState:
    elixir: int = 0
    hand: List[str] = field(default_factory=lambda: ["card0", "card1", "card2", "card3"])  # placeholders
    my_towers: List[int] = field(default_factory=lambda: [100, 100, 100])
    enemy_towers: List[int] = field(default_factory=lambda: [100, 100, 100])
    ui: Dict = field(default_factory=dict)

    def update_from_frame(self, frame: np.ndarray, ui: Dict) -> None:
        self.ui = ui
        self.elixir = int(ui.get("elixir", 0))
        # Note: towers/cards detection would go here; stubbed for scaffold

