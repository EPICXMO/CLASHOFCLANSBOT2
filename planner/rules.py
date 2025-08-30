import os
import random
import logging
from typing import Optional, Tuple, List

import numpy as np


logger = logging.getLogger(__name__)


from .bandit import EpsilonGreedy


class RulePlanner:
    """Simple heuristic planner.

    - If elixir >= 4, deploy a random card to a safe zone.
    - Prioritize defense if we detect an enemy push (stubbed heuristic).
    - Handles chests/upgrades in non-battle contexts (via UI stubs).
    """

    def __init__(self):
        self.bandit = EpsilonGreedy(k=4, epsilon=0.5)
        self.episode_idx = 0
        self._store_path = os.path.join("logs", "bandit.json")
        self._load_state()

    def enemy_push_detected(self, ui: dict) -> bool:
        yolo = ui.get("yolo", {})
        troops: List[Tuple[int, int, int, int]] = yolo.get("troop", [])
        my_towers: List[Tuple[int, int, int, int]] = yolo.get("my_tower", [])
        # Heuristic: any troop within 150px of a my_tower center
        for t in troops:
            tx = (t[0] + t[2]) / 2
            ty = (t[1] + t[3]) / 2
            for mt in my_towers:
                mx = (mt[0] + mt[2]) / 2
                my = (mt[1] + mt[3]) / 2
                if (tx - mx) ** 2 + (ty - my) ** 2 <= (150 ** 2):
                    return True
        return False

    def plan_battle(self, state, frame: np.ndarray) -> Optional[Tuple[int, Tuple[float, float]]]:
        elixir = int(getattr(state, "elixir", 0))
        ui = getattr(state, "ui", {})
        card_slots = ui.get("card_slots", {})
        safe_zone = ui.get("safe_zone")

        if elixir < 4:
            logger.info("Thought: Low elixir (%d) — waiting 0.5s before next try", elixir)
            return None

        # Defensive preference if push detected
        if self.enemy_push_detected(ui):
            self.bandit.ensure_arms(max(4, len(card_slots)))
            idx = self._select_card_index(card_slots)
            target = self._defensive_point(ui, frame)
            logger.info("Thought: Enemy push detected — defending with card %d at defensive point", idx)
            return (idx, target)

        # Otherwise random card to safe zone
        self.bandit.ensure_arms(max(4, len(card_slots)))
        idx = self._select_card_index(card_slots)
        target = self._offensive_point(ui, frame)
        logger.info("Thought: No push — attacking with card %d at offensive point", idx)
        return (idx, target)

    @staticmethod
    def _random_point_in_zone(frame: np.ndarray, zone, bias_low: bool = False) -> Tuple[float, float]:
        h, w = frame.shape[:2]
        if zone is None:
            x, y = random.uniform(0.1, 0.9), random.uniform(0.55, 0.75)
            return (x, y)
        x1, y1, x2, y2 = zone
        if bias_low:
            y_mid = (y1 + y2) // 2
            y1 = y_mid  # bias towards lower half of safe zone
        x = random.uniform(x1 / w, x2 / w)
        y = random.uniform(y1 / h, y2 / h)
        return (x, y)

    def handle_progression(self, controller, ui: dict) -> None:
        logger.info("Thought: Checking progression — rewards/chests/upgrades")
        # Tap rewards button (YOLO or ROI fallbacks)
        rb = None
        yb = ui.get("yolo", {}).get("rewards_button", [])
        if yb:
            rb = yb[0]
        rb = rb or ui.get("rewards_button_bottom") or ui.get("rewards_button_center")
        if rb:
            x1, y1, x2, y2 = rb
            controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
            logger.info("Thought: Collecting rewards via rewards button")

        # Tap chests if detected by YOLO
        for ch in ui.get("yolo", {}).get("chest", [])[:4]:
            x1, y1, x2, y2 = ch
            controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
            logger.info("Thought: Opening chest at (%d,%d)", int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Tap upgrade button (YOLO or ROI) if we likely have enough gold
        ub = None
        yub = ui.get("yolo", {}).get("upgrade_button", [])
        if yub:
            ub = yub[0]
        ub = ub or ui.get("upgrade_button")
        gold = int(ui.get("gold", 0))
        if ub and gold >= 100:
            x1, y1, x2, y2 = ub
            controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
            logger.info("Thought: Upgrading a card (gold=%d)", gold)

    # Bandit helpers
    def _select_card_index(self, card_slots) -> int:
        if not card_slots:
            return 0
        # Choose with bandit
        return self.bandit.select()

    def record_outcome(self, card_idx: int, reward: float) -> None:
        # Update bandit if index in range
        if 0 <= card_idx < self.bandit.k:
            self.bandit.update(card_idx, reward)
            self._save_state()

    def _defensive_point(self, ui: dict, frame: np.ndarray) -> Tuple[float, float]:
        # pick a point near the closest my_tower to bottom
        my_towers: List[Tuple[int, int, int, int]] = ui.get("yolo", {}).get("my_tower", [])
        h, w = frame.shape[:2]
        if my_towers:
            # choose lowest (max y)
            mt = max(my_towers, key=lambda b: (b[1] + b[3]) / 2)
            cx = (mt[0] + mt[2]) / 2 / w
            cy = (mt[1] + mt[3]) / 2 / h
            # add a small offset toward center
            cx = min(max(cx * 0.9 + 0.05, 0.1), 0.9)
            cy = min(max(cy * 0.9 + 0.05, 0.1), 0.9)
            return (cx, cy)
        return self._random_point_in_zone(frame, ui.get("safe_zone"), bias_low=True)

    def _offensive_point(self, ui: dict, frame: np.ndarray) -> Tuple[float, float]:
        # Aim toward enemy towers (choose highest y enemy tower)
        enemy_towers: List[Tuple[int, int, int, int]] = ui.get("yolo", {}).get("enemy_tower", [])
        h, w = frame.shape[:2]
        if enemy_towers:
            # choose closest to river (max y among enemy towers)
            et = max(enemy_towers, key=lambda b: (b[1] + b[3]) / 2)
            cx = (et[0] + et[2]) / 2 / w
            cy = ((et[1] + et[3]) / 2) / h
            cy = max(0.25, min(cy - 0.1, 0.6))
            return (cx, cy)
        # fallback upper half of safe zone
        zone = ui.get("safe_zone")
        h2, w2 = frame.shape[:2]
        if zone is not None:
            x1, y1, x2, y2 = zone
            y2 = y1 + (y2 - y1) // 2  # upper half
            return self._random_point_in_zone(frame, (x1, y1, x2, y2))
        return self._random_point_in_zone(frame, None)

    def set_exploration(self, episode_idx: int) -> None:
        # Decay epsilon from 0.5 to 0.05 over 50 episodes
        self.episode_idx = max(self.episode_idx, int(episode_idx))
        self.bandit.decay(self.episode_idx, total=50, min_epsilon=0.05)
        self._save_state()

    # Persistence helpers
    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
            data = {
                "k": self.bandit.k,
                "epsilon": self.bandit.epsilon,
                "counts": self.bandit.counts,
                "values": self.bandit.values,
                "episode_idx": self.episode_idx,
            }
            import json
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            import json
            if not os.path.exists(self._store_path):
                return
            with open(self._store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            k = int(data.get("k", self.bandit.k))
            self.bandit.ensure_arms(k)
            self.bandit.epsilon = float(data.get("epsilon", self.bandit.epsilon))
            counts = data.get("counts")
            values = data.get("values")
            if isinstance(counts, list) and isinstance(values, list):
                if len(counts) == self.bandit.k and len(values) == self.bandit.k:
                    self.bandit.counts = [int(x) for x in counts]
                    self.bandit.values = [float(x) for x in values]
            self.episode_idx = int(data.get("episode_idx", 0))
        except Exception:
            pass
