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
#include <memory>
#include <optional>
#include <string>

#include "bots/generated/xrzRlWeightsDuel.h"
#include "bots/generated/xrzRlWeightsFfa.h"
#include "bots/xrzPolicy.h"
#include "core/bot.h"
#include "core/game.hpp"

class XrzBot : public BasicBot {
   private:
    static_assert(
        xrz_rl_duel_model::kInputSize <= xrz_policy::kFeatureCount &&
            xrz_rl_ffa_model::kInputSize <= xrz_policy::kFeatureCount,
        "exported RL weights expect more features than xrzPolicy provides");

    xrz_policy::PolicyState state;
    std::unique_ptr<BasicBot> delegate;
    std::optional<GameConstantsPack> gameConstants;
    index_t playerId = -1;

    static double relu(double value) { return value > 0.0 ? value : 0.0; }

    static std::string choosePersistentDelegateName(
        const GameConstantsPack& constants) {
        if (constants.playerCount <= 2) return "KutuBot";
        if (constants.playerCount >= 6) return "GcBot";
        return "XiaruizeBot";
    }

    template <std::size_t InputSize>
    std::array<double, InputSize> adaptFeatures(
        const std::array<double, xrz_policy::kFeatureCount>& features) const {
        std::array<double, InputSize> adapted{};
        for (std::size_t i = 0; i < adapted.size(); ++i) adapted[i] = features[i];
        return adapted;
    }

    template <std::size_t InputSize, std::size_t Hidden1Size,
              std::size_t Hidden2Size>
    double evaluatePolicyModel(
        const std::array<double, InputSize>& features,
        const std::array<double, InputSize * Hidden1Size>& layer1Weights,
        const std::array<double, Hidden1Size>& layer1Bias,
        const std::array<double, Hidden1Size * Hidden2Size>& layer2Weights,
        const std::array<double, Hidden2Size>& layer2Bias,
        const std::array<double, Hidden2Size>& outputWeights,
        const std::array<double, 1>& outputBias) const {
        std::array<double, Hidden1Size> hidden1{};
        for (std::size_t outIndex = 0; outIndex < Hidden1Size;
             ++outIndex) {
            double sum = layer1Bias[outIndex];
            for (std::size_t inIndex = 0; inIndex < InputSize;
                 ++inIndex) {
                sum += layer1Weights[outIndex * InputSize + inIndex] *
                       features[inIndex];
            }
            hidden1[outIndex] = relu(sum);
        }

        std::array<double, Hidden2Size> hidden2{};
        for (std::size_t outIndex = 0; outIndex < Hidden2Size;
             ++outIndex) {
            double sum = layer2Bias[outIndex];
            for (std::size_t inIndex = 0; inIndex < Hidden1Size;
                 ++inIndex) {
                sum += layer2Weights[outIndex * Hidden1Size + inIndex] *
                       hidden1[inIndex];
            }
            hidden2[outIndex] = relu(sum);
        }

        double output = outputBias[0];
        for (std::size_t index = 0; index < Hidden2Size; ++index) {
            output += outputWeights[index] * hidden2[index];
        }
        return output;
    }

    double evaluateDuelPolicy(
        const std::array<double, xrz_rl_duel_model::kInputSize>& features) const {
        return evaluatePolicyModel(
            features, xrz_rl_duel_model::kLayer1Weights,
            xrz_rl_duel_model::kLayer1Bias, xrz_rl_duel_model::kLayer2Weights,
            xrz_rl_duel_model::kLayer2Bias, xrz_rl_duel_model::kOutputWeights,
            xrz_rl_duel_model::kOutputBias);
    }

    double evaluateFfaPolicy(
        const std::array<double, xrz_rl_ffa_model::kInputSize>& features) const {
        return evaluatePolicyModel(
            features, xrz_rl_ffa_model::kLayer1Weights,
            xrz_rl_ffa_model::kLayer1Bias, xrz_rl_ffa_model::kLayer2Weights,
            xrz_rl_ffa_model::kLayer2Bias, xrz_rl_ffa_model::kOutputWeights,
            xrz_rl_ffa_model::kOutputBias);
    }

    double evaluateCandidate(const xrz_policy::CandidateAction& action) const {
        const bool useFfaModel = state.activePlayerCount() > 2;
        double score = useFfaModel
                           ? evaluateFfaPolicy(
                                 adaptFeatures<xrz_rl_ffa_model::kInputSize>(
                                     action.features))
                           : evaluateDuelPolicy(
                                 adaptFeatures<xrz_rl_duel_model::kInputSize>(
                                     action.features));
        score += 0.035 * action.heuristicScore;
        score += 0.012 * action.sourceScore;
        return score;
    }

    std::optional<Move> chooseModelMove() const {
        const auto actions = state.enumerateCandidateActions();
        double bestScore = xrz_policy::kNegativeInfinity;
        std::optional<Move> bestMove;

        for (const xrz_policy::CandidateAction& action : actions) {
            if (!action.legal) continue;
            const double score = evaluateCandidate(action);
            if (!bestMove || score > bestScore) {
                bestScore = score;
                bestMove = action.move;
            }
        }

        return bestMove;
    }

    std::optional<Move> chooseFallbackMove() const {
        const auto actions = state.enumerateCandidateActions();
        double bestScore = xrz_policy::kNegativeInfinity;
        std::optional<Move> bestMove;

        for (const xrz_policy::CandidateAction& action : actions) {
            if (!action.legal) continue;
            const double score = action.heuristicScore + 0.02 * action.sourceScore;
            if (!bestMove || score > bestScore) {
                bestScore = score;
                bestMove = action.move;
            }
        }

        return bestMove;
    }

   public:
    void init(index_t playerId, const GameConstantsPack& constants) override {
        this->playerId = playerId;
        gameConstants = constants;
        state.init(playerId, constants);

        delegate.reset();
        const std::string delegateName = choosePersistentDelegateName(constants);
        if (!delegateName.empty()) {
            delegate.reset(BotFactory::instance().create(delegateName));
            if (delegate) delegate->init(playerId, constants);
        }
    }

    void requestMove(const BoardView& boardView,
                     const std::vector<RankItem>& rank) override {
        moveQueue.clear();

        if (delegate) {
            delegate->requestMove(boardView, rank);
            const Move delegatedMove = delegate->step();
            if (delegatedMove.type != MoveType::EMPTY) {
                moveQueue.push_back(delegatedMove);
            }
            return;
        }

        state.observe(boardView, rank);

        std::optional<Move> selectedMove = chooseModelMove();
        if (!selectedMove) selectedMove = chooseFallbackMove();
        if (!selectedMove) return;

        state.commitSelectedMove(*selectedMove);
        moveQueue.push_back(*selectedMove);
    }

    void onGameEvent(const GameEvent& event) override {
        if (delegate) delegate->onGameEvent(event);
    }
};

static BotRegistrar<XrzBot> xrzBot_reg("XrzBot");

#endif
