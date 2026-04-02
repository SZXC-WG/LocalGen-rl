/**
 * @file xrzBot.cpp
 *
 * RL-guided XrzBot with shared feature extraction used by both inference and
 * imitation-learning data export.
 */

#ifndef LGEN_BOTS_XRZBOT
#define LGEN_BOTS_XRZBOT

#include <array>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <vector>

#include "bots/generated/xrzRlWeightsDuel.h"
#include "bots/generated/xrzRlWeightsFfa.h"
#include "bots/xrzPolicy.h"
#include "core/bot.h"
#include "core/game.hpp"

class XrzBot : public BasicBot {
   private:
    static constexpr std::size_t kRuntimeDuelSourceCount = 48;
    static constexpr std::size_t kRuntimeFfaSourceCount = 64;
    static constexpr int kUnreachableDistance = 1'000'000;

    static_assert(
        xrz_rl_duel_model::kInputSize <= xrz_policy::kFeatureCount &&
            xrz_rl_ffa_model::kInputSize <= xrz_policy::kFeatureCount,
        "exported RL weights expect more features than xrzPolicy provides");

    struct StrategicMaps {
        std::vector<int> enemyGeneralDistances;
        std::vector<int> cityDistances;
        std::vector<int> enemyDistances;
        std::vector<int> expansionDistances;
    };

    xrz_policy::PolicyState state;

    static double relu(double value) { return value > 0.0 ? value : 0.0; }

    template <std::size_t InputSize>
    std::array<double, InputSize> adaptFeatures(
        const std::array<double, xrz_policy::kFeatureCount>& features) const {
        std::array<double, InputSize> adapted{};
        for (std::size_t i = 0; i < adapted.size(); ++i)
            adapted[i] = features[i];
        return adapted;
    }

    template <std::size_t InputSize, std::size_t Hidden1Size,
              std::size_t Hidden2Size, std::size_t Hidden3Size>
    double evaluatePolicyModel(
        const std::array<double, InputSize>& features,
        const std::array<double, InputSize * Hidden1Size>& layer1Weights,
        const std::array<double, Hidden1Size>& layer1Bias,
        const std::array<double, Hidden1Size * Hidden2Size>& layer2Weights,
        const std::array<double, Hidden2Size>& layer2Bias,
        const std::array<double, Hidden2Size * Hidden3Size>& layer3Weights,
        const std::array<double, Hidden3Size>& layer3Bias,
        const std::array<double, (Hidden3Size > 0 ? Hidden3Size : Hidden2Size)>&
            outputWeights,
        const std::array<double, 1>& outputBias) const {
        std::array<double, Hidden1Size> hidden1{};
        for (std::size_t outIndex = 0; outIndex < Hidden1Size; ++outIndex) {
            double sum = layer1Bias[outIndex];
            for (std::size_t inIndex = 0; inIndex < InputSize; ++inIndex) {
                sum += layer1Weights[outIndex * InputSize + inIndex] *
                       features[inIndex];
            }
            hidden1[outIndex] = relu(sum);
        }

        std::array<double, Hidden2Size> hidden2{};
        for (std::size_t outIndex = 0; outIndex < Hidden2Size; ++outIndex) {
            double sum = layer2Bias[outIndex];
            for (std::size_t inIndex = 0; inIndex < Hidden1Size; ++inIndex) {
                sum += layer2Weights[outIndex * Hidden1Size + inIndex] *
                       hidden1[inIndex];
            }
            hidden2[outIndex] = relu(sum);
        }

        if constexpr (Hidden3Size > 0) {
            std::array<double, Hidden3Size> hidden3{};
            for (std::size_t outIndex = 0; outIndex < Hidden3Size; ++outIndex) {
                double sum = layer3Bias[outIndex];
                for (std::size_t inIndex = 0; inIndex < Hidden2Size;
                     ++inIndex) {
                    sum += layer3Weights[outIndex * Hidden2Size + inIndex] *
                           hidden2[inIndex];
                }
                hidden3[outIndex] = relu(sum);
            }

            double output = outputBias[0];
            for (std::size_t index = 0; index < Hidden3Size; ++index) {
                output += outputWeights[index] * hidden3[index];
            }
            return output;
        }

        double output = outputBias[0];
        for (std::size_t index = 0; index < Hidden2Size; ++index) {
            output += outputWeights[index] * hidden2[index];
        }
        return output;
    }

    double evaluateDuelPolicy(
        const std::array<double, xrz_rl_duel_model::kInputSize>& features)
        const {
        return evaluatePolicyModel(
            features, xrz_rl_duel_model::kLayer1Weights,
            xrz_rl_duel_model::kLayer1Bias, xrz_rl_duel_model::kLayer2Weights,
            xrz_rl_duel_model::kLayer2Bias, xrz_rl_duel_model::kLayer3Weights,
            xrz_rl_duel_model::kLayer3Bias, xrz_rl_duel_model::kOutputWeights,
            xrz_rl_duel_model::kOutputBias);
    }

    double evaluateFfaPolicy(
        const std::array<double, xrz_rl_ffa_model::kInputSize>& features)
        const {
        return evaluatePolicyModel(
            features, xrz_rl_ffa_model::kLayer1Weights,
            xrz_rl_ffa_model::kLayer1Bias, xrz_rl_ffa_model::kLayer2Weights,
            xrz_rl_ffa_model::kLayer2Bias, xrz_rl_ffa_model::kLayer3Weights,
            xrz_rl_ffa_model::kLayer3Bias, xrz_rl_ffa_model::kOutputWeights,
            xrz_rl_ffa_model::kOutputBias);
    }

    bool isPassableTerrain(tile_type_e terrain) const {
        return !isImpassableTile(terrain) && terrain != TILE_OBSTACLE;
    }

    std::size_t cellIndex(Coord coord) const {
        return static_cast<std::size_t>(coord.x * (state.mapWidth() + 2) +
                                        coord.y);
    }

    template <typename Predicate>
    std::vector<int> buildDistanceMap(Predicate&& isTarget) const {
        const pos_t height = state.mapHeight();
        const pos_t width = state.mapWidth();
        const pos_t stride = width + 2;
        std::vector<int> distances((height + 2) * stride, kUnreachableDistance);
        std::deque<Coord> frontier;

        for (pos_t x = 1; x <= height; ++x) {
            for (pos_t y = 1; y <= width; ++y) {
                const Coord coord{x, y};
                const tile_type_e terrain = state.observedTerrainAt(coord);
                if (!isPassableTerrain(terrain)) continue;
                if (!isTarget(coord)) continue;
                distances[cellIndex(coord)] = 0;
                frontier.push_back(coord);
            }
        }

        while (!frontier.empty()) {
            const Coord current = frontier.front();
            frontier.pop_front();
            const int nextDistance = distances[cellIndex(current)] + 1;
            for (Coord delta : xrz_policy::kDirs) {
                const Coord next = current + delta;
                if (!state.insideMap(next)) continue;
                if (!isPassableTerrain(state.observedTerrainAt(next))) continue;
                const std::size_t nextIndex = cellIndex(next);
                if (nextDistance >= distances[nextIndex]) continue;
                distances[nextIndex] = nextDistance;
                frontier.push_back(next);
            }
        }

        return distances;
    }

    StrategicMaps buildStrategicMaps() const {
        StrategicMaps maps;
        maps.enemyGeneralDistances = buildDistanceMap([this](Coord coord) {
            return state.observedTerrainAt(coord) == TILE_GENERAL &&
                   state.isEnemyOccupier(state.observedOccupierAt(coord));
        });
        maps.cityDistances = buildDistanceMap([this](Coord coord) {
            return state.observedTerrainAt(coord) == TILE_CITY &&
                   !state.isFriendlyOccupier(state.observedOccupierAt(coord));
        });
        maps.enemyDistances = buildDistanceMap([this](Coord coord) {
            return state.isEnemyOccupier(state.observedOccupierAt(coord));
        });
        maps.expansionDistances = buildDistanceMap([this](Coord coord) {
            const tile_type_e terrain = state.observedTerrainAt(coord);
            return state.observedOccupierAt(coord) == -1 &&
                   terrain != TILE_CITY && terrain != TILE_SWAMP;
        });
        return maps;
    }

    double distanceProgressBonus(const std::vector<int>& distances,
                                 const xrz_policy::CandidateAction& action,
                                 double stepWeight, double arrivalBonus) const {
        if (distances.empty()) return 0.0;
        const int fromDistance = distances[cellIndex(action.source)];
        const int toDistance = distances[cellIndex(action.target)];
        if (fromDistance >= kUnreachableDistance ||
            toDistance >= kUnreachableDistance) {
            return 0.0;
        }

        double bonus =
            stepWeight * static_cast<double>(fromDistance - toDistance);
        if (toDistance == 0 && fromDistance > 0) bonus += arrivalBonus;
        return bonus;
    }

    double strategicRouteBonus(
        const StrategicMaps& maps,
        const xrz_policy::CandidateAction& action) const {
        const int halfTurnCount = state.observedHalfTurnCount();
        const bool useFfaModel = state.activePlayerCount() > 2;
        double bonus = 0.0;

        bonus += distanceProgressBonus(maps.enemyGeneralDistances, action, 4.0,
                                       12.0);
        bonus += distanceProgressBonus(maps.cityDistances, action,
                                       halfTurnCount <= 120 ? 2.4 : 1.6, 4.0);
        bonus += distanceProgressBonus(maps.enemyDistances, action,
                                       halfTurnCount <= 40 ? 0.5 : 1.4, 1.5);
        bonus += distanceProgressBonus(
            maps.expansionDistances, action,
            halfTurnCount <= 70 ? (useFfaModel ? 1.6 : 1.2) : 0.3, 0.8);
        return bonus;
    }

    double evaluateCandidate(const xrz_policy::CandidateAction& action,
                             const StrategicMaps& maps) const {
        const int halfTurnCount = state.observedHalfTurnCount();
        const bool useFfaModel = state.activePlayerCount() > 2;
        const double modelScore =
            useFfaModel
                ? evaluateFfaPolicy(adaptFeatures<xrz_rl_ffa_model::kInputSize>(
                      action.features))
                : evaluateDuelPolicy(
                      adaptFeatures<xrz_rl_duel_model::kInputSize>(
                          action.features));

        double heuristicWeight = useFfaModel ? 0.050 : 0.040;
        double sourceWeight = useFfaModel ? 0.010 : 0.014;
        if (halfTurnCount <= 50) {
            heuristicWeight += useFfaModel ? 0.018 : 0.014;
            sourceWeight += useFfaModel ? 0.003 : 0.006;
        } else if (halfTurnCount <= 120) {
            heuristicWeight += 0.006;
            sourceWeight += useFfaModel ? 0.002 : 0.004;
        }

        double score = modelScore;
        score += heuristicWeight * action.heuristicScore;
        score += sourceWeight * action.sourceScore;
        score += strategicRouteBonus(maps, action);
        if (!action.takeHalf) score += 0.35;

        if (action.features[16] > 0.5 && action.features[14] > 0.5)
            score += 120.0;
        else if (action.features[17] > 0.5 && action.features[13] < 0.5)
            score += 10.0;
        else if (action.features[14] > 0.5 && action.features[21] > 0.0)
            score += 2.5;

        if (halfTurnCount <= 60 && action.features[15] > 0.5 &&
            !action.takeHalf) {
            score += 2.0;
        }
        if (action.features[18] > 0.5) score -= action.takeHalf ? 0.5 : 1.5;
        if (action.features[22] > 0.5) score -= 1.5;
        return score;
    }

    std::size_t runtimeSourceLimit() const {
        return state.activePlayerCount() > 2 ? kRuntimeFfaSourceCount
                                             : kRuntimeDuelSourceCount;
    }

    std::optional<Move> chooseModelMove(const StrategicMaps& maps) const {
        const auto actions =
            state.enumerateCandidateActions(runtimeSourceLimit());
        double bestScore = xrz_policy::kNegativeInfinity;
        std::optional<Move> bestMove;

        for (const xrz_policy::CandidateAction& action : actions) {
            if (!action.legal) continue;
            const double score = evaluateCandidate(action, maps);
            if (!bestMove || score > bestScore) {
                bestScore = score;
                bestMove = action.move;
            }
        }

        return bestMove;
    }

    std::optional<Move> chooseFallbackMove(const StrategicMaps& maps) const {
        const auto actions =
            state.enumerateCandidateActions(runtimeSourceLimit());
        double bestScore = xrz_policy::kNegativeInfinity;
        std::optional<Move> bestMove;

        for (const xrz_policy::CandidateAction& action : actions) {
            if (!action.legal) continue;
            const double score = action.heuristicScore +
                                 0.02 * action.sourceScore +
                                 strategicRouteBonus(maps, action);
            if (!bestMove || score > bestScore) {
                bestScore = score;
                bestMove = action.move;
            }
        }

        return bestMove;
    }

   public:
    void init(index_t playerId, const GameConstantsPack& constants) override {
        state.init(playerId, constants);
    }

    void requestMove(const BoardView& boardView,
                     const std::vector<RankItem>& rank) override {
        moveQueue.clear();

        state.observe(boardView, rank);

        const StrategicMaps maps = buildStrategicMaps();
        std::optional<Move> selectedMove = chooseModelMove(maps);
        if (!selectedMove) selectedMove = chooseFallbackMove(maps);
        if (!selectedMove) return;

        state.commitSelectedMove(*selectedMove);
        moveQueue.push_back(*selectedMove);
    }

    void onGameEvent(const GameEvent& event) override {}
};

static BotRegistrar<XrzBot> xrzBot_reg("XrzBot");

#endif
