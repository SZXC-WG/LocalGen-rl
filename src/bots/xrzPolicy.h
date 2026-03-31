/**
 * @file xrzPolicy.h
 *
 * Shared candidate generation and feature encoding for XrzBot training and
 * inference.
 */

#ifndef LGEN_BOTS_XRZ_POLICY_H
#define LGEN_BOTS_XRZ_POLICY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core/game.hpp"

namespace xrz_policy {

inline constexpr std::size_t kDirectionCount = 4;
inline constexpr std::size_t kSplitModeCount = 2;
inline constexpr std::size_t kCandidateSourceCount = 16;
inline constexpr std::size_t kActionsPerSource =
    kDirectionCount * kSplitModeCount;
inline constexpr std::size_t kDatasetActionCount =
    kCandidateSourceCount * kActionsPerSource;
inline constexpr std::size_t kFeatureCount = 37;
inline constexpr double kNegativeInfinity = -1e18;
inline constexpr std::array<Coord, kDirectionCount> kDirs = {
    Coord{-1, 0},
    Coord{0, -1},
    Coord{1, 0},
    Coord{0, 1},
};

struct MemoryCell {
    tile_type_e terrain = TILE_BLANK;
    index_t occupier = -1;
    army_t army = 0;
    bool seen = false;
    bool visible = false;
    int lastSeenHalfTurn = -1;
};

struct CandidateAction {
    Move move{};
    Coord source{-1, -1};
    Coord target{-1, -1};
    std::array<double, kFeatureCount> features{};
    double sourceScore = kNegativeInfinity;
    double heuristicScore = kNegativeInfinity;
    bool legal = false;
    bool takeHalf = false;
    std::size_t sourceSlot = 0;
    std::size_t directionIndex = 0;
};

class PolicyState {
   public:
    void init(index_t playerId, const GameConstantsPack& constants) {
        id = playerId;
        height = constants.mapHeight;
        width = constants.mapWidth;
        stride = width + 2;
        playerCount = constants.playerCount;
        teams = constants.teams;
        team = teams.at(playerId);
        config = constants.config;

        rankById.assign(playerCount, RankItem{});
        for (index_t i = 0; i < playerCount; ++i) rankById[i].player = i;
        visitTime.assign((height + 2) * stride, 0);
        memory.assign((height + 2) * stride, MemoryCell{});
        knownGenerals.assign(playerCount, Coord{-1, -1});
        halfTurnCount = 0;
        lastMoveFrom = Coord{-1, -1};
        lastMoveTo = Coord{-1, -1};
    }

    index_t activePlayerCount() const { return playerCount; }

    void observe(const BoardView& boardView,
                 const std::vector<RankItem>& rank) {
        board = boardView;
        ++halfTurnCount;
        syncRank(rank);
        updateMemory();
    }

    void commitSelectedMove(const Move& move) {
        if (move.type != MoveType::MOVE_ARMY) return;
        if (inside(move.from)) ++visitTime[idx(move.from)];
        if (inside(move.to)) ++visitTime[idx(move.to)];
        lastMoveFrom = move.from;
        lastMoveTo = move.to;
    }

    bool isLegalMove(Coord from, Coord to) const {
        if (!inside(from) || !inside(to)) return false;
        if (std::abs(from.x - to.x) + std::abs(from.y - to.y) != 1)
            return false;
        const TileView& fromTile = board.tileAt(from);
        if (!fromTile.visible || fromTile.occupier != id || fromTile.army <= 1)
            return false;
        const tile_type_e targetType = board.tileAt(to).type;
        return !isImpassableTile(targetType) && targetType != TILE_OBSTACLE;
    }

    std::vector<CandidateAction> enumerateCandidateActions(
        std::size_t sourceLimit = kCandidateSourceCount) const {
        const std::vector<Coord> sources = candidateSources(sourceLimit);
        std::vector<CandidateAction> result;
        result.reserve(sources.size() * kActionsPerSource);

        for (std::size_t sourceSlot = 0; sourceSlot < sources.size();
             ++sourceSlot) {
            const Coord source = sources[sourceSlot];
            const double score = sourceScore(source);
            for (std::size_t directionIndex = 0;
                 directionIndex < kDirectionCount; ++directionIndex) {
                for (std::size_t splitIndex = 0; splitIndex < kSplitModeCount;
                     ++splitIndex) {
                    const bool takeHalf = splitIndex == 1;
                    result.push_back(makeCandidate(
                        sourceSlot, source, directionIndex, takeHalf, score));
                }
            }
        }

        return result;
    }

    std::array<CandidateAction, kDatasetActionCount>
    enumerateFixedCandidateActions() const {
        std::array<CandidateAction, kDatasetActionCount> actions{};
        const std::vector<Coord> sources =
            candidateSources(kCandidateSourceCount);

        for (std::size_t sourceSlot = 0; sourceSlot < kCandidateSourceCount;
             ++sourceSlot) {
            const bool sourceExists = sourceSlot < sources.size();
            const Coord source =
                sourceExists ? sources[sourceSlot] : Coord{-1, -1};
            const double score =
                sourceExists ? sourceScore(source) : kNegativeInfinity;
            for (std::size_t directionIndex = 0;
                 directionIndex < kDirectionCount; ++directionIndex) {
                for (std::size_t splitIndex = 0; splitIndex < kSplitModeCount;
                     ++splitIndex) {
                    const std::size_t actionIndex =
                        sourceSlot * kActionsPerSource +
                        directionIndex * kSplitModeCount + splitIndex;
                    if (!sourceExists) {
                        CandidateAction placeholder;
                        placeholder.sourceSlot = sourceSlot;
                        placeholder.directionIndex = directionIndex;
                        placeholder.takeHalf = splitIndex == 1;
                        actions[actionIndex] = placeholder;
                        continue;
                    }
                    actions[actionIndex] =
                        makeCandidate(sourceSlot, source, directionIndex,
                                      splitIndex == 1, score);
                }
            }
        }

        return actions;
    }

    std::optional<std::size_t> labelForMove(
        const std::array<CandidateAction, kDatasetActionCount>& actions,
        const Move& move) const {
        if (move.type != MoveType::MOVE_ARMY) return std::nullopt;
        for (std::size_t i = 0; i < actions.size(); ++i) {
            const CandidateAction& action = actions[i];
            if (!action.legal) continue;
            if (action.move.from == move.from && action.move.to == move.to &&
                action.move.takeHalf == move.takeHalf) {
                return i;
            }
        }
        return std::nullopt;
    }

    double sourceScore(Coord source) const {
        if (!inside(source)) return kNegativeInfinity;
        const TileView& tile = board.tileAt(source);
        if (!tile.visible || tile.occupier != id || tile.army <= 1)
            return kNegativeInfinity;

        int enemyAdjacent = 0;
        int cityAdjacent = 0;
        int neutralAdjacent = 0;
        const int frontier = countFrontier(source);
        for (Coord delta : kDirs) {
            const Coord target = source + delta;
            if (!inside(target)) continue;
            const tile_type_e targetType = terrainAt(target);
            if (isImpassableTile(targetType) || targetType == TILE_OBSTACLE)
                continue;
            const index_t targetOccupier = occupierAt(target);
            if (isEnemy(targetOccupier)) ++enemyAdjacent;
            if (targetType == TILE_CITY && !isFriendly(targetOccupier))
                ++cityAdjacent;
            if (targetOccupier == -1) ++neutralAdjacent;
            if (targetType == TILE_GENERAL && isEnemy(targetOccupier))
                cityAdjacent += 2;
        }

        const double visitPenalty = 0.7 * visitTime[idx(source)];
        const double generalPenalty = tile.type == TILE_GENERAL ? 8.0 : 0.0;
        const double pressurePenalty =
            std::max(0.0, static_cast<double>(enemyPressure(source))) * 0.15;

        double strategicBonus = 0.0;
        if (const auto closestEnemyGeneral = nearestKnownEnemyGeneral(source)) {
            const double closeness = normalizedDistanceCloseness(
                *closestEnemyGeneral, height + width);
            strategicBonus += 16.0 * closeness;
        }

        return tile.army * 4.0 + frontier * 8.0 + enemyAdjacent * 14.0 +
               cityAdjacent * 12.0 + neutralAdjacent * 3.5 + strategicBonus -
               visitPenalty - generalPenalty - pressurePenalty;
    }

    double heuristicPrior(Coord source, Coord target, bool takeHalf) const {
        if (!inside(source) || !inside(target)) return -64.0;

        const TileView& sourceTile = board.tileAt(source);
        const tile_type_e targetType = terrainAt(target);
        const index_t targetOccupier = occupierAt(target);
        const army_t targetArmy = armyAt(target);
        const army_t movedArmy = movedArmyFor(sourceTile.army, takeHalf);
        const army_t remainingArmy = sourceTile.army - movedArmy;

        double score = 0.0;
        if (targetType == TILE_GENERAL && isEnemy(targetOccupier))
            score += 1500.0;
        if (targetType == TILE_CITY && !isFriendly(targetOccupier))
            score += 180.0;
        if (isEnemy(targetOccupier)) {
            score += 70.0;
            score += std::max(0.0, static_cast<double>(movedArmy - targetArmy));
        } else if (targetOccupier == -1) {
            score += 18.0;
        } else {
            score += 4.0;
        }

        if (targetType == TILE_SWAMP) score -= takeHalf ? 8.0 : 16.0;
        if (source == lastMoveTo && target == lastMoveFrom) score -= 18.0;
        if (!takeHalf && sourceTile.type == TILE_GENERAL && remainingArmy <= 1)
            score -= 24.0;
        if (takeHalf && !isFriendly(targetOccupier) && movedArmy <= targetArmy)
            score -= 28.0;
        if (takeHalf && isFriendly(targetOccupier)) score += 4.0;
        score -= 1.3 * visitTime[idx(target)];

        if (const auto closestEnemyGeneral = nearestKnownEnemyGeneral(source)) {
            const int sourceDist = *closestEnemyGeneral;
            const int targetDist =
                distanceToClosestKnownEnemyGeneral(target).value_or(sourceDist);
            score += static_cast<double>(sourceDist - targetDist) * 6.5;
        }

        return score;
    }

   private:
    pos_t height = 0;
    pos_t width = 0;
    pos_t stride = 0;
    index_t playerCount = 0;
    index_t id = -1;
    index_t team = -1;
    config::Config config;

    std::vector<index_t> teams;
    BoardView board;
    std::vector<RankItem> rankById;
    std::vector<int> visitTime;
    std::vector<MemoryCell> memory;
    std::vector<Coord> knownGenerals;
    int halfTurnCount = 0;
    Coord lastMoveFrom{-1, -1};
    Coord lastMoveTo{-1, -1};

    static double scale(double value, double denominator) {
        if (denominator <= 0.0) return 0.0;
        return std::clamp(value / denominator, -4.0, 4.0);
    }

    static double normalizedDistanceCloseness(int distance, int maxDistance) {
        if (distance < 0 || maxDistance <= 0) return 0.0;
        const double clamped = std::min(static_cast<double>(distance),
                                        static_cast<double>(maxDistance));
        return 1.0 - clamped / static_cast<double>(maxDistance);
    }

    inline std::size_t idx(Coord coord) const {
        return static_cast<std::size_t>(coord.x * stride + coord.y);
    }

    inline bool inside(Coord coord) const {
        return coord.x >= 1 && coord.x <= height && coord.y >= 1 &&
               coord.y <= width;
    }

    inline bool validPlayer(index_t player) const {
        return player >= 0 && player < playerCount;
    }

    inline bool sameTeam(index_t a, index_t b) const {
        if (!validPlayer(a) || !validPlayer(b) || teams.empty()) return a == b;
        return teams[a] == teams[b];
    }

    inline bool isFriendly(index_t occupier) const {
        return validPlayer(occupier) && sameTeam(occupier, id);
    }

    inline bool isEnemy(index_t occupier) const {
        return validPlayer(occupier) && !sameTeam(occupier, id);
    }

    inline bool isEdge(Coord coord) const {
        return coord.x == 1 || coord.x == height || coord.y == 1 ||
               coord.y == width;
    }

    inline int manhattan(Coord lhs, Coord rhs) const {
        return std::abs(lhs.x - rhs.x) + std::abs(lhs.y - rhs.y);
    }

    void syncRank(const std::vector<RankItem>& rank) {
        rankById.assign(playerCount, RankItem{});
        for (index_t i = 0; i < playerCount; ++i) rankById[i].player = i;
        for (const RankItem& item : rank) {
            if (validPlayer(item.player)) rankById[item.player] = item;
        }
    }

    void updateMemory() {
        for (index_t player = 0; player < playerCount; ++player) {
            if (!rankById.empty() &&
                player < static_cast<index_t>(rankById.size()) &&
                !rankById[player].alive) {
                knownGenerals[player] = Coord{-1, -1};
            }
        }

        for (pos_t x = 1; x <= height; ++x) {
            for (pos_t y = 1; y <= width; ++y) {
                const Coord coord{x, y};
                const TileView& tile = board.tileAt(coord);
                MemoryCell& mem = memory[idx(coord)];
                mem.visible = tile.visible;
                if (!tile.visible) continue;

                mem.seen = true;
                mem.terrain = tile.type;
                mem.occupier = tile.occupier;
                mem.army = tile.army;
                mem.lastSeenHalfTurn = halfTurnCount;

                for (index_t player = 0; player < playerCount; ++player) {
                    if (knownGenerals[player] == coord &&
                        (tile.type != TILE_GENERAL ||
                         tile.occupier != player)) {
                        knownGenerals[player] = Coord{-1, -1};
                    }
                }

                if (tile.type == TILE_GENERAL && validPlayer(tile.occupier)) {
                    knownGenerals[tile.occupier] = coord;
                }
            }
        }
    }

    tile_type_e terrainAt(Coord coord) const {
        const TileView& tile = board.tileAt(coord);
        if (tile.visible) return tile.type;
        const MemoryCell& mem = memory[idx(coord)];
        if (mem.seen) return mem.terrain;
        return tile.type;
    }

    index_t occupierAt(Coord coord) const {
        const TileView& tile = board.tileAt(coord);
        if (tile.visible) return tile.occupier;
        const MemoryCell& mem = memory[idx(coord)];
        return mem.seen ? mem.occupier : -1;
    }

    army_t armyAt(Coord coord) const {
        const TileView& tile = board.tileAt(coord);
        if (tile.visible) return tile.army;
        const MemoryCell& mem = memory[idx(coord)];
        return mem.seen ? mem.army : 0;
    }

    bool visibleAt(Coord coord) const { return board.tileAt(coord).visible; }

    int countFrontier(Coord source) const {
        int frontier = 0;
        for (Coord delta : kDirs) {
            const Coord target = source + delta;
            if (!inside(target)) continue;
            const tile_type_e targetType = terrainAt(target);
            if (isImpassableTile(targetType) || targetType == TILE_OBSTACLE)
                continue;
            if (!isFriendly(occupierAt(target))) ++frontier;
        }
        return frontier;
    }

    std::pair<int, int> neighborCounts(Coord target) const {
        int enemyCount = 0;
        int friendlyCount = 0;
        for (Coord delta : kDirs) {
            const Coord next = target + delta;
            if (!inside(next)) continue;
            const index_t occupier = occupierAt(next);
            if (isFriendly(occupier))
                ++friendlyCount;
            else if (isEnemy(occupier))
                ++enemyCount;
        }
        return {enemyCount, friendlyCount};
    }

    int enemyPressure(Coord source) const {
        const army_t sourceArmy = board.tileAt(source).army;
        int pressure = 0;
        for (Coord delta : kDirs) {
            const Coord target = source + delta;
            if (!inside(target)) continue;
            const index_t occupier = occupierAt(target);
            if (!isEnemy(occupier)) continue;
            pressure = std::max(pressure, static_cast<int>(std::max<army_t>(
                                              0, armyAt(target) - sourceArmy)));
        }
        return pressure;
    }

    std::pair<long long, long long> strongestEnemyTotals() const {
        long long bestArmy = 0;
        long long bestLand = 0;
        for (const RankItem& item : rankById) {
            if (item.player == id || isFriendly(item.player)) continue;
            if (item.army > bestArmy ||
                (item.army == bestArmy && item.land > bestLand)) {
                bestArmy = item.army;
                bestLand = item.land;
            }
        }
        return {bestArmy, bestLand};
    }

    std::optional<int> nearestKnownEnemyGeneral(Coord source) const {
        int bestDistance = std::numeric_limits<int>::max();
        bool found = false;
        for (index_t player = 0; player < playerCount; ++player) {
            if (!isEnemy(player)) continue;
            const Coord general = knownGenerals[player];
            if (!inside(general)) continue;
            found = true;
            bestDistance = std::min(bestDistance, manhattan(source, general));
        }
        if (!found) return std::nullopt;
        return bestDistance;
    }

    std::optional<int> distanceToClosestKnownEnemyGeneral(Coord coord) const {
        return nearestKnownEnemyGeneral(coord);
    }

    static army_t movedArmyFor(army_t sourceArmy, bool takeHalf) {
        if (sourceArmy <= 1) return 0;
        return takeHalf ? (sourceArmy >> 1) : (sourceArmy - 1);
    }

    CandidateAction makeCandidate(std::size_t sourceSlot, Coord source,
                                  std::size_t directionIndex, bool takeHalf,
                                  double score) const {
        CandidateAction action;
        action.sourceSlot = sourceSlot;
        action.directionIndex = directionIndex;
        action.takeHalf = takeHalf;
        action.source = source;
        action.sourceScore = score;
        action.target = source + kDirs[directionIndex];
        action.legal = isLegalMove(source, action.target);
        action.move =
            Move(MoveType::MOVE_ARMY, source, action.target, takeHalf);
        action.features = buildFeatures(source, directionIndex, takeHalf);
        action.heuristicScore = heuristicPrior(source, action.target, takeHalf);
        return action;
    }

    std::vector<Coord> candidateSources(std::size_t limit) const {
        struct ScoredSource {
            Coord coord{-1, -1};
            double score = kNegativeInfinity;
        };

        std::vector<ScoredSource> scored;
        for (pos_t x = 1; x <= height; ++x) {
            for (pos_t y = 1; y <= width; ++y) {
                const Coord coord{x, y};
                const double score = sourceScore(coord);
                if (score == kNegativeInfinity) continue;
                scored.push_back({coord, score});
            }
        }

        std::stable_sort(scored.begin(), scored.end(),
                         [](const ScoredSource& lhs, const ScoredSource& rhs) {
                             return lhs.score > rhs.score;
                         });
        if (scored.size() > limit) scored.resize(limit);

        std::vector<Coord> result;
        result.reserve(scored.size());
        for (const ScoredSource& entry : scored) result.push_back(entry.coord);
        return result;
    }

    std::array<double, kFeatureCount> buildFeatures(Coord source,
                                                    std::size_t directionIndex,
                                                    bool takeHalf) const {
        std::array<double, kFeatureCount> features{};
        const Coord delta = kDirs[directionIndex];
        const Coord target = source + delta;
        const TileView& sourceTile = board.tileAt(source);
        const bool targetInBounds = inside(target);
        const bool legal = targetInBounds && isLegalMove(source, target);
        const RankItem ownRank =
            id >= 0 && id < static_cast<index_t>(rankById.size()) ? rankById[id]
                                                                  : RankItem{};
        const auto [enemyArmy, enemyLand] = strongestEnemyTotals();
        const int sourceFrontier = countFrontier(source);
        const int sourceVisitCount = visitTime[idx(source)];
        const army_t movedArmy = movedArmyFor(sourceTile.army, takeHalf);
        const army_t remainingArmy = sourceTile.army - movedArmy;
        const auto sourceGeneralDistance =
            distanceToClosestKnownEnemyGeneral(source);
        const auto targetGeneralDistance =
            targetInBounds ? distanceToClosestKnownEnemyGeneral(target)
                           : std::nullopt;
        const bool knownEnemyGeneral = sourceGeneralDistance.has_value();
        const double sourceEnemyGeneralCloseness =
            sourceGeneralDistance ? normalizedDistanceCloseness(
                                        *sourceGeneralDistance, height + width)
                                  : 0.0;
        const double targetEnemyGeneralCloseness =
            targetGeneralDistance ? normalizedDistanceCloseness(
                                        *targetGeneralDistance, height + width)
                                  : 0.0;

        features[0] = 1.0;
        features[1] = scale(static_cast<double>(sourceTile.army), 64.0);
        features[2] = sourceTile.type == TILE_GENERAL ? 1.0 : 0.0;
        features[3] = sourceTile.type == TILE_CITY ? 1.0 : 0.0;
        features[4] = std::min(1.0, static_cast<double>(halfTurnCount) / 240.0);
        features[5] = scale(static_cast<double>(ownRank.army), 256.0);
        features[6] = scale(static_cast<double>(ownRank.land), 128.0);
        features[7] = scale(static_cast<double>(enemyArmy), 256.0);
        features[8] = scale(static_cast<double>(enemyLand), 128.0);
        features[9] = static_cast<double>(delta.x);
        features[10] = static_cast<double>(delta.y);
        features[11] = targetInBounds ? 1.0 : 0.0;
        features[12] = legal ? 1.0 : 0.0;
        features[24] = scale(static_cast<double>(sourceVisitCount), 16.0);
        features[25] = scale(static_cast<double>(sourceFrontier), 4.0);
        features[28] = isEdge(source) ? 1.0 : 0.0;

        features[30] = scale(static_cast<double>(movedArmy), 64.0);
        features[31] = scale(static_cast<double>(remainingArmy), 64.0);
        features[32] = takeHalf ? 1.0 : 0.0;
        features[33] = knownEnemyGeneral ? 1.0 : 0.0;
        features[34] = sourceEnemyGeneralCloseness;
        features[35] = targetEnemyGeneralCloseness;
        features[36] = scale(static_cast<double>(enemyPressure(source)), 64.0);

        if (targetInBounds) {
            const tile_type_e targetType = terrainAt(target);
            const index_t targetOccupier = occupierAt(target);
            const army_t targetArmy = armyAt(target);
            const bool targetFriendly = isFriendly(targetOccupier);
            const bool targetEnemy = isEnemy(targetOccupier);
            const bool targetNeutral = targetOccupier == -1;
            const bool targetObstacle =
                isImpassableTile(targetType) || targetType == TILE_OBSTACLE;
            const auto [enemyNeighbors, friendlyNeighbors] =
                neighborCounts(target);
            const bool reverseMove =
                source == lastMoveTo && target == lastMoveFrom;

            features[13] = targetFriendly ? 1.0 : 0.0;
            features[14] = targetEnemy ? 1.0 : 0.0;
            features[15] = targetNeutral ? 1.0 : 0.0;
            features[16] = targetType == TILE_GENERAL ? 1.0 : 0.0;
            features[17] = targetType == TILE_CITY ? 1.0 : 0.0;
            features[18] = targetType == TILE_SWAMP ? 1.0 : 0.0;
            features[19] = targetObstacle ? 1.0 : 0.0;
            features[20] = scale(static_cast<double>(targetArmy), 64.0);
            features[21] =
                scale(static_cast<double>(movedArmy - targetArmy), 32.0);
            features[22] = reverseMove ? 1.0 : 0.0;
            features[23] =
                scale(static_cast<double>(visitTime[idx(target)]), 16.0);
            features[26] = scale(static_cast<double>(enemyNeighbors), 4.0);
            features[27] = scale(static_cast<double>(friendlyNeighbors), 4.0);
            features[29] = isEdge(target) ? 1.0 : 0.0;
        }

        return features;
    }
};

}  // namespace xrz_policy

#endif  // LGEN_BOTS_XRZ_POLICY_H
