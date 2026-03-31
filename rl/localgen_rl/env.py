from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

from .constants import (
    ARMY_SCALE,
    CAPTURE_MARGIN_SCALE,
    DIRECTIONS,
    IMPASSABLE_TILE_TYPES,
    INPUT_FEATURE_COUNT,
    LAND_SCALE,
    NEIGHBOR_SCALE,
    OUTPUT_ACTION_COUNT,
    SPLIT_MODES,
    TOTAL_ARMY_SCALE,
    VISIT_SCALE,
    TileType,
)


@dataclass(slots=True)
class Cell:
    occupier: int = -1
    tile_type: TileType = TileType.BLANK
    army: int = 0


@dataclass(slots=True)
class Observation:
    action_features: list[list[float]]
    legal_mask: list[bool]
    source: Optional[tuple[int, int]]


class LocalGenMiniEnv:
    """Small LocalGen-inspired environment for XrzBot policy training.

    The environment keeps the action format aligned with the C++ policy:
    for a selected source tile, score 8 actions ordered as
    [up/full, up/half, left/full, left/half, down/full, down/half,
     right/full, right/half].
    """

    def __init__(
        self,
        *,
        seed: int = 1337,
        board_min_size: int = 12,
        board_max_size: int = 18,
        max_half_turns: int = 240,
    ) -> None:
        self.rng = random.Random(seed)
        self.board_min_size = board_min_size
        self.board_max_size = board_max_size
        self.max_half_turns = max_half_turns

        self.height = 0
        self.width = 0
        self.board: list[Cell] = []
        self.alive = [True, True]
        self.current_turn = 0
        self.half_turn_phase = 0
        self.half_turn_count = 0
        self.visit_counts: list[list[int]] = [[], []]
        self.last_moves: list[tuple[tuple[int, int], tuple[int, int]]] = [
            ((-1, -1), (-1, -1)),
            ((-1, -1), (-1, -1)),
        ]
        self.selected_sources: list[Optional[tuple[int, int]]] = [None, None]

    def reset(self) -> Observation:
        self.height = self.rng.randint(self.board_min_size, self.board_max_size)
        self.width = self.rng.randint(self.board_min_size, self.board_max_size)
        self.board = [Cell() for _ in range((self.height + 2) * (self.width + 2))]
        self.alive = [True, True]
        self.current_turn = 0
        self.half_turn_phase = 0
        self.half_turn_count = 0
        board_size = len(self.board)
        self.visit_counts = [[0 for _ in range(board_size)] for _ in range(2)]
        self.last_moves = [((-1, -1), (-1, -1)), ((-1, -1), (-1, -1))]
        self.selected_sources = [None, None]

        self._generate_board()
        return self._build_observation(0)

    def step(self, action: int) -> tuple[Observation, float, bool, dict[str, object]]:
        if action < 0 or action >= OUTPUT_ACTION_COUNT:
            raise ValueError(f"invalid action {action}")

        own_army_before, own_land_before = self._totals(0)
        enemy_army_before, enemy_land_before = self._totals(1)

        direction_index = action // len(SPLIT_MODES)
        take_half = SPLIT_MODES[action % len(SPLIT_MODES)]
        direction = DIRECTIONS[direction_index]

        agent_source = self.selected_sources[0] or self._select_source(0)
        agent_move: Optional[tuple[int, tuple[int, int], tuple[int, int], bool]] = None
        invalid_action = False
        target_before: Optional[Cell] = None

        if agent_source is not None:
            target = self._add(agent_source, direction)
            if self._available(0, agent_source, target):
                target_before = self._copy_cell(self._tile_at(target))
                agent_move = (0, agent_source, target, take_half)
            else:
                invalid_action = True
        else:
            invalid_action = True

        enemy_move = self._scripted_move(1)
        executed_moves = self._execute_moves(
            [move for move in (agent_move, enemy_move) if move is not None]
        )

        for player, source, target, _ in executed_moves:
            self.last_moves[player] = (source, target)
            self.visit_counts[player][self._idx(*source)] += 1
            self.visit_counts[player][self._idx(*target)] += 1

        if self.half_turn_phase == 0:
            self._update_board(
                increase_all_army=self.current_turn > 0 and self.current_turn % 25 == 0
            )
        self.current_turn += self.half_turn_phase
        self.half_turn_phase ^= 1
        self.half_turn_count += 1

        own_army_after, own_land_after = self._totals(0)
        enemy_army_after, enemy_land_after = self._totals(1)

        reward = -0.02
        reward += 0.40 * (own_land_after - own_land_before)
        reward += 0.03 * (own_army_after - own_army_before)
        reward -= 0.10 * (enemy_land_after - enemy_land_before)
        reward -= 0.02 * (enemy_army_after - enemy_army_before)

        if invalid_action:
            reward -= 0.75

        if agent_move is not None and target_before is not None:
            target_after = self._tile_at(agent_move[2])
            if target_before.tile_type == TileType.SWAMP:
                reward -= 1.0
            if target_after.occupier == 0 and target_before.occupier != 0:
                if target_before.tile_type == TileType.GENERAL:
                    reward += 30.0
                elif target_before.tile_type == TileType.CITY:
                    reward += 6.0
                elif target_before.occupier == -1:
                    reward += 1.0
                else:
                    reward += 3.5

        done = False
        winner: Optional[int] = None
        if not self.alive[0]:
            done = True
            winner = 1
            reward -= 30.0
        elif not self.alive[1]:
            done = True
            winner = 0
            reward += 30.0
        elif self.half_turn_count >= self.max_half_turns:
            done = True
            score0 = own_army_after + 4 * own_land_after
            score1 = enemy_army_after + 4 * enemy_land_after
            if score0 > score1:
                winner = 0
                reward += 8.0 + min(12.0, (score0 - score1) / 12.0)
            elif score1 > score0:
                winner = 1
                reward -= 8.0 + min(12.0, (score1 - score0) / 12.0)
            else:
                winner = None

        observation = self._build_observation(0)
        info: dict[str, object] = {
            "winner": winner,
            "own_army": own_army_after,
            "own_land": own_land_after,
            "enemy_army": enemy_army_after,
            "enemy_land": enemy_land_after,
            "invalid_action": invalid_action,
        }
        return observation, reward, done, info

    def _generate_board(self) -> None:
        area = self.height * self.width
        num_mountains = self.rng.randint(max(1, area // 7), max(1, area // 7 + area // 20))
        num_cities = self.rng.randint(max(1, area // 30), max(1, area // 15))
        num_swamps = self.rng.randint(max(1, area // 20), max(1, area // 15))

        for _ in range(num_mountains):
            coord = self._random_blank_coord()
            tile_roll = self.rng.randint(0, 9)
            tile_type = (
                TileType.LOOKOUT
                if tile_roll == 0
                else TileType.OBSERVATORY
                if tile_roll == 1
                else TileType.MOUNTAIN
            )
            tile = self._tile_at(coord)
            tile.tile_type = tile_type
            tile.occupier = -1
            tile.army = 0

        for _ in range(num_cities):
            coord = self._random_blank_coord()
            tile = self._tile_at(coord)
            tile.tile_type = TileType.CITY
            tile.occupier = -1
            tile.army = self.rng.randint(40, 49)

        for _ in range(num_swamps):
            coord = self._random_blank_coord()
            tile = self._tile_at(coord)
            tile.tile_type = TileType.SWAMP
            tile.occupier = -1
            tile.army = 0

        general0 = self._random_blank_coord()
        general1 = self._random_blank_coord(
            min_distance=(self.height + self.width) // 2, avoid=general0
        )

        tile0 = self._tile_at(general0)
        tile0.tile_type = TileType.GENERAL
        tile0.occupier = 0
        tile0.army = 1

        tile1 = self._tile_at(general1)
        tile1.tile_type = TileType.GENERAL
        tile1.occupier = 1
        tile1.army = 1

    def _build_observation(self, player: int) -> Observation:
        source = self._select_source(player)
        self.selected_sources[player] = source
        if source is None:
            return Observation(
                action_features=[[0.0] * INPUT_FEATURE_COUNT for _ in range(OUTPUT_ACTION_COUNT)],
                legal_mask=[False] * OUTPUT_ACTION_COUNT,
                source=None,
            )

        source_cell = self._tile_at(source)
        own_army, own_land = self._totals(player)
        enemy_army, enemy_land = self._totals(1 - player)
        source_frontier = self._count_frontier(player, source)
        source_visit_count = self.visit_counts[player][self._idx(*source)]
        last_from, last_to = self.last_moves[player]
        enemy_general = self._find_general(1 - player)
        known_enemy_general = enemy_general is not None
        source_general_distance = self._distance_to_enemy_general(player, source)
        source_general_closeness = self._distance_closeness(source_general_distance)
        source_enemy_pressure = self._enemy_pressure(player, source)

        action_features: list[list[float]] = []
        legal_mask: list[bool] = []

        for dx, dy in DIRECTIONS:
            target = (source[0] + dx, source[1] + dy)
            in_bounds = self._in_bounds(target)
            base_legal = self._available(player, source, target) if in_bounds else False
            target_army = 0
            target_owned = 0.0
            target_enemy = 0.0
            target_neutral = 0.0
            target_is_general = 0.0
            target_is_city = 0.0
            target_is_swamp = 0.0
            target_is_obstacle = 0.0
            target_enemy_neighbors = 0
            target_friendly_neighbors = 0
            target_visit_count = 0
            target_is_edge = 0.0
            target_general_closeness = 0.0

            if in_bounds:
                target_cell = self._tile_at(target)
                target_army = target_cell.army
                target_owned = 1.0 if target_cell.occupier == player else 0.0
                target_enemy = 1.0 if target_cell.occupier not in (-1, player) else 0.0
                target_neutral = 1.0 if target_cell.occupier == -1 else 0.0
                target_is_general = 1.0 if target_cell.tile_type == TileType.GENERAL else 0.0
                target_is_city = 1.0 if target_cell.tile_type == TileType.CITY else 0.0
                target_is_swamp = 1.0 if target_cell.tile_type == TileType.SWAMP else 0.0
                target_is_obstacle = 1.0 if target_cell.tile_type in IMPASSABLE_TILE_TYPES else 0.0
                target_enemy_neighbors, target_friendly_neighbors = self._neighbor_counts(player, target)
                target_visit_count = self.visit_counts[player][self._idx(*target)]
                target_is_edge = 1.0 if self._is_edge(target) else 0.0
                target_general_closeness = self._distance_closeness(
                    self._distance_to_enemy_general(player, target)
                )

            for take_half in SPLIT_MODES:
                moved_army = self._moved_army(source_cell.army, take_half)
                remaining_army = source_cell.army - moved_army
                capture_margin = moved_army - target_army
                reverse_last_move = 1.0 if source == last_to and target == last_from else 0.0
                features = [
                    1.0,
                    self._scale(source_cell.army, ARMY_SCALE),
                    1.0 if source_cell.tile_type == TileType.GENERAL else 0.0,
                    1.0 if source_cell.tile_type == TileType.CITY else 0.0,
                    min(1.0, self.half_turn_count / 240.0),
                    self._scale(own_army, TOTAL_ARMY_SCALE),
                    self._scale(own_land, LAND_SCALE),
                    self._scale(enemy_army, TOTAL_ARMY_SCALE),
                    self._scale(enemy_land, LAND_SCALE),
                    float(dx),
                    float(dy),
                    1.0 if in_bounds else 0.0,
                    1.0 if base_legal else 0.0,
                    target_owned,
                    target_enemy,
                    target_neutral,
                    target_is_general,
                    target_is_city,
                    target_is_swamp,
                    target_is_obstacle,
                    self._scale(target_army, ARMY_SCALE),
                    self._scale(capture_margin, CAPTURE_MARGIN_SCALE),
                    reverse_last_move,
                    self._scale(target_visit_count, VISIT_SCALE),
                    self._scale(source_visit_count, VISIT_SCALE),
                    self._scale(source_frontier, NEIGHBOR_SCALE),
                    self._scale(target_enemy_neighbors, NEIGHBOR_SCALE),
                    self._scale(target_friendly_neighbors, NEIGHBOR_SCALE),
                    1.0 if self._is_edge(source) else 0.0,
                    target_is_edge,
                    self._scale(moved_army, ARMY_SCALE),
                    self._scale(remaining_army, ARMY_SCALE),
                    1.0 if take_half else 0.0,
                    1.0 if known_enemy_general else 0.0,
                    source_general_closeness,
                    target_general_closeness,
                    self._scale(source_enemy_pressure, ARMY_SCALE),
                ]
                action_features.append(features)
                legal_mask.append(base_legal)

        return Observation(action_features=action_features, legal_mask=legal_mask, source=source)

    def _scripted_move(
        self, player: int
    ) -> Optional[tuple[int, tuple[int, int], tuple[int, int], bool]]:
        source = self._select_source(player)
        self.selected_sources[player] = source
        if source is None:
            return None

        best_score = float("-inf")
        best_move: Optional[tuple[int, tuple[int, int], tuple[int, int], bool]] = None
        source_bias = 0.02 * self._source_score(player, source)

        for direction in DIRECTIONS:
            target = self._add(source, direction)
            if not self._available(player, source, target):
                continue
            for take_half in SPLIT_MODES:
                score = self._heuristic_prior(player, source, target, take_half)
                score += source_bias
                score += self.rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_move = (player, source, target, take_half)

        return best_move

    def _execute_moves(
        self, moves: list[tuple[int, tuple[int, int], tuple[int, int], bool]]
    ) -> list[tuple[int, tuple[int, int], tuple[int, int], bool]]:
        move_out_map = {source: player for player, source, _, _ in moves}
        ordered_moves = sorted(
            moves,
            key=lambda move: self._sort_key(move, move_out_map),
        )

        executed: list[tuple[int, tuple[int, int], tuple[int, int], bool]] = []
        for player, source, target, take_half in ordered_moves:
            if not self.alive[player] or not self._available(player, source, target):
                continue
            from_cell = self._tile_at(source)
            to_cell = self._tile_at(target)

            taken_army = self._moved_army(from_cell.army, take_half)
            if taken_army <= 0:
                continue
            from_cell.army -= taken_army
            if to_cell.occupier == player:
                to_cell.army += taken_army
            else:
                to_cell.army -= taken_army
                if to_cell.army < 0:
                    to_cell.army = -to_cell.army
                    defender = to_cell.occupier
                    if to_cell.tile_type == TileType.GENERAL and defender not in (-1, player):
                        self._capture(player, defender)
                    to_cell.occupier = player
            executed.append((player, source, target, take_half))
        return executed

    def _capture(self, attacker: int, defender: int) -> None:
        self.alive[defender] = False
        for cell in self.board:
            if cell.occupier != defender:
                continue
            cell.occupier = attacker
            if cell.tile_type == TileType.GENERAL:
                cell.tile_type = TileType.CITY
            elif cell.army > 1:
                cell.army //= 2

    def _update_board(self, *, increase_all_army: bool) -> None:
        for cell in self.board:
            if cell.occupier == -1:
                continue
            if cell.tile_type in (TileType.CITY, TileType.GENERAL):
                cell.army += 1
                if increase_all_army:
                    cell.army += 1
            elif cell.tile_type == TileType.BLANK:
                if increase_all_army:
                    cell.army += 1
            elif cell.tile_type == TileType.SWAMP:
                if cell.army > 0:
                    cell.army -= 1
                if cell.army == 0:
                    cell.occupier = -1

    def _totals(self, player: int) -> tuple[int, int]:
        army = 0
        land = 0
        for cell in self.board:
            if cell.occupier == player:
                army += cell.army
                land += 1
        return army, land

    def _select_source(self, player: int) -> Optional[tuple[int, int]]:
        best_score = float("-inf")
        best_coord: Optional[tuple[int, int]] = None
        for x in range(1, self.height + 1):
            for y in range(1, self.width + 1):
                coord = (x, y)
                score = self._source_score(player, coord)
                if score > best_score:
                    best_score = score
                    best_coord = coord
        return best_coord

    def _source_score(self, player: int, coord: tuple[int, int]) -> float:
        if not self._in_bounds(coord):
            return float("-inf")
        cell = self._tile_at(coord)
        if cell.occupier != player or cell.army <= 1:
            return float("-inf")

        enemy_adjacent = 0
        city_adjacent = 0
        neutral_adjacent = 0
        frontier = 0
        for direction in DIRECTIONS:
            target = self._add(coord, direction)
            if not self._in_bounds(target):
                continue
            target_cell = self._tile_at(target)
            if target_cell.tile_type in IMPASSABLE_TILE_TYPES:
                continue
            if target_cell.occupier != player:
                frontier += 1
            if target_cell.occupier not in (-1, player):
                enemy_adjacent += 1
            if target_cell.tile_type == TileType.CITY and target_cell.occupier != player:
                city_adjacent += 1
            if target_cell.occupier == -1:
                neutral_adjacent += 1
            if target_cell.tile_type == TileType.GENERAL and target_cell.occupier not in (-1, player):
                city_adjacent += 2

        visit_penalty = 0.7 * self.visit_counts[player][self._idx(*coord)]
        general_penalty = 8.0 if cell.tile_type == TileType.GENERAL else 0.0
        pressure_penalty = max(0.0, float(self._enemy_pressure(player, coord))) * 0.15
        strategic_bonus = 16.0 * self._distance_closeness(
            self._distance_to_enemy_general(player, coord)
        )
        return (
            cell.army * 4.0
            + frontier * 8.0
            + enemy_adjacent * 14.0
            + city_adjacent * 12.0
            + neutral_adjacent * 3.5
            + strategic_bonus
            - visit_penalty
            - general_penalty
            - pressure_penalty
        )

    def _heuristic_prior(
        self,
        player: int,
        source: tuple[int, int],
        target: tuple[int, int],
        take_half: bool,
    ) -> float:
        if not self._in_bounds(source) or not self._in_bounds(target):
            return -64.0

        source_cell = self._tile_at(source)
        target_cell = self._tile_at(target)
        moved_army = self._moved_army(source_cell.army, take_half)
        remaining_army = source_cell.army - moved_army

        score = 0.0
        if target_cell.tile_type == TileType.GENERAL and target_cell.occupier not in (-1, player):
            score += 1500.0
        if target_cell.tile_type == TileType.CITY and target_cell.occupier != player:
            score += 180.0
        if target_cell.occupier not in (-1, player):
            score += 70.0
            score += max(0.0, float(moved_army - target_cell.army))
        elif target_cell.occupier == -1:
            score += 18.0
        else:
            score += 4.0

        if target_cell.tile_type == TileType.SWAMP:
            score -= 8.0 if take_half else 16.0
        last_from, last_to = self.last_moves[player]
        if source == last_to and target == last_from:
            score -= 18.0
        if not take_half and source_cell.tile_type == TileType.GENERAL and remaining_army <= 1:
            score -= 24.0
        if take_half and target_cell.occupier != player and moved_army <= target_cell.army:
            score -= 28.0
        if take_half and target_cell.occupier == player:
            score += 4.0
        score -= 1.3 * self.visit_counts[player][self._idx(*target)]

        source_distance = self._distance_to_enemy_general(player, source)
        target_distance = self._distance_to_enemy_general(player, target)
        if source_distance is not None:
            if target_distance is None:
                target_distance = source_distance
            score += float(source_distance - target_distance) * 6.5

        return score

    def _count_frontier(self, player: int, coord: tuple[int, int]) -> int:
        frontier = 0
        for direction in DIRECTIONS:
            target = self._add(coord, direction)
            if not self._in_bounds(target):
                continue
            target_cell = self._tile_at(target)
            if target_cell.tile_type in IMPASSABLE_TILE_TYPES:
                continue
            if target_cell.occupier != player:
                frontier += 1
        return frontier

    def _neighbor_counts(self, player: int, coord: tuple[int, int]) -> tuple[int, int]:
        enemy_count = 0
        friendly_count = 0
        for direction in DIRECTIONS:
            target = self._add(coord, direction)
            if not self._in_bounds(target):
                continue
            target_cell = self._tile_at(target)
            if target_cell.occupier == player:
                friendly_count += 1
            elif target_cell.occupier != -1:
                enemy_count += 1
        return enemy_count, friendly_count

    def _enemy_pressure(self, player: int, coord: tuple[int, int]) -> int:
        source_army = self._tile_at(coord).army
        pressure = 0
        for direction in DIRECTIONS:
            target = self._add(coord, direction)
            if not self._in_bounds(target):
                continue
            target_cell = self._tile_at(target)
            if target_cell.occupier in (-1, player):
                continue
            pressure = max(pressure, max(0, target_cell.army - source_army))
        return pressure

    def _find_general(self, player: int) -> Optional[tuple[int, int]]:
        for x in range(1, self.height + 1):
            for y in range(1, self.width + 1):
                coord = (x, y)
                cell = self._tile_at(coord)
                if cell.tile_type == TileType.GENERAL and cell.occupier == player:
                    return coord
        return None

    def _distance_to_enemy_general(
        self, player: int, coord: tuple[int, int]
    ) -> Optional[int]:
        enemy_general = self._find_general(1 - player)
        if enemy_general is None:
            return None
        return abs(coord[0] - enemy_general[0]) + abs(coord[1] - enemy_general[1])

    def _distance_closeness(self, distance: Optional[int]) -> float:
        if distance is None:
            return 0.0
        max_distance = max(1, self.height + self.width)
        return 1.0 - min(float(distance), float(max_distance)) / float(max_distance)

    def _available(
        self, player: int, source: tuple[int, int], target: tuple[int, int]
    ) -> bool:
        if not self._in_bounds(source) or not self._in_bounds(target):
            return False
        if abs(source[0] - target[0]) + abs(source[1] - target[1]) != 1:
            return False
        from_cell = self._tile_at(source)
        if from_cell.occupier != player or from_cell.army <= 1:
            return False
        return self._tile_at(target).tile_type not in IMPASSABLE_TILE_TYPES

    def _sort_key(
        self,
        move: tuple[int, tuple[int, int], tuple[int, int], bool],
        move_out_map: dict[tuple[int, int], int],
    ) -> tuple[float, float, int]:
        player, source, target, _ = move
        to_cell = self._tile_at(target)
        if to_cell.occupier not in (-1, player) and move_out_map.get(target) == to_cell.occupier:
            priority = 3
        elif to_cell.occupier == player:
            priority = 2
        elif to_cell.tile_type == TileType.GENERAL and to_cell.occupier not in (-1, player):
            priority = 0
        else:
            priority = 1

        army_size = self._tile_at(source).army
        index_priority = player if self.half_turn_phase == 0 else -player
        return (-priority, -army_size, index_priority)

    def _random_blank_coord(
        self,
        *,
        min_distance: int = 0,
        avoid: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int]:
        fallback: Optional[tuple[int, int]] = None
        for _ in range(512):
            coord = (
                self.rng.randint(1, self.height),
                self.rng.randint(1, self.width),
            )
            cell = self._tile_at(coord)
            if cell.occupier != -1 or cell.tile_type != TileType.BLANK or cell.army != 0:
                continue
            if fallback is None:
                fallback = coord
            if avoid is not None:
                distance = abs(coord[0] - avoid[0]) + abs(coord[1] - avoid[1])
                if distance < min_distance:
                    continue
            return coord

        for x in range(1, self.height + 1):
            for y in range(1, self.width + 1):
                coord = (x, y)
                cell = self._tile_at(coord)
                if cell.occupier == -1 and cell.tile_type == TileType.BLANK and cell.army == 0:
                    if fallback is None:
                        fallback = coord
                    if avoid is None:
                        return coord
                    distance = abs(coord[0] - avoid[0]) + abs(coord[1] - avoid[1])
                    if distance >= min_distance:
                        return coord
        if fallback is not None:
            return fallback
        raise RuntimeError("failed to allocate a blank coordinate")

    def _tile_at(self, coord: tuple[int, int]) -> Cell:
        return self.board[self._idx(*coord)]

    def _copy_cell(self, cell: Cell) -> Cell:
        return Cell(occupier=cell.occupier, tile_type=cell.tile_type, army=cell.army)

    def _idx(self, x: int, y: int) -> int:
        return x * (self.width + 2) + y

    def _in_bounds(self, coord: tuple[int, int]) -> bool:
        x, y = coord
        return 1 <= x <= self.height and 1 <= y <= self.width

    def _is_edge(self, coord: tuple[int, int]) -> bool:
        x, y = coord
        return x in (1, self.height) or y in (1, self.width)

    def _add(self, coord: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
        return coord[0] + delta[0], coord[1] + delta[1]

    @staticmethod
    def _moved_army(source_army: int, take_half: bool) -> int:
        if source_army <= 1:
            return 0
        return source_army >> 1 if take_half else source_army - 1

    @staticmethod
    def _scale(value: float, scale: float) -> float:
        bounded = max(-4.0 * scale, min(4.0 * scale, value))
        return bounded / scale
