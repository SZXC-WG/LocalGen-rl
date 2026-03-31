from __future__ import annotations

from enum import IntEnum


class TileType(IntEnum):
    GENERAL = 0
    BLANK = 1
    MOUNTAIN = 2
    CITY = 3
    SWAMP = 4
    DESERT = 5
    LOOKOUT = 6
    OBSERVATORY = 7


IMPASSABLE_TILE_TYPES = {
    TileType.MOUNTAIN,
    TileType.LOOKOUT,
    TileType.OBSERVATORY,
}

DIRECTIONS = ((-1, 0), (0, -1), (1, 0), (0, 1))
SPLIT_MODES = (False, True)
OUTPUT_ACTION_COUNT = len(DIRECTIONS) * len(SPLIT_MODES)

ARMY_SCALE = 64.0
TOTAL_ARMY_SCALE = 256.0
LAND_SCALE = 128.0
TURN_SCALE = 200.0
VISIT_SCALE = 16.0
NEIGHBOR_SCALE = 4.0
CAPTURE_MARGIN_SCALE = 32.0

FEATURE_NAMES = (
    "bias",
    "source_army_norm",
    "source_is_general",
    "source_is_city",
    "turn_progress",
    "own_total_army_norm",
    "own_total_land_norm",
    "enemy_total_army_norm",
    "enemy_total_land_norm",
    "dir_dx",
    "dir_dy",
    "target_in_bounds",
    "legal_flag",
    "target_owned",
    "target_enemy",
    "target_neutral",
    "target_is_general",
    "target_is_city",
    "target_is_swamp",
    "target_is_obstacle",
    "target_army_norm",
    "capture_margin_norm",
    "reverse_last_move",
    "target_visit_count_norm",
    "source_visit_count_norm",
    "source_frontier_norm",
    "target_enemy_neighbors_norm",
    "target_friendly_neighbors_norm",
    "source_is_edge",
    "target_is_edge",
    "moved_army_norm",
    "remaining_army_norm",
    "take_half_flag",
    "knows_enemy_general",
    "source_enemy_general_closeness",
    "target_enemy_general_closeness",
    "source_enemy_pressure_norm",
)

INPUT_FEATURE_COUNT = len(FEATURE_NAMES)
