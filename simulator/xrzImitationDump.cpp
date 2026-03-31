/**
 * @file xrzImitationDump.cpp
 *
 * Dump imitation-learning trajectories for XrzBot by wrapping a teacher bot and
 * recording XrzPolicy candidate actions from real LocalGen games.
 */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bots/xrzPolicy.h"
#include "core/bot.h"
#include "core/game.hpp"
#include "core/map.hpp"

namespace {

struct BoardSize {
    int width = 18;
    int height = 18;
};

struct LoadedBoard {
    std::string path;
    Board board;
};

struct SelectedBot {
    std::string botName;
    std::string displayName;
};

struct GameSetup {
    SelectedBot teacher;
    std::vector<SelectedBot> opponents;
    std::string boardLabel;
    Board board;
};

struct Options {
    int games = 64;
    int width = 18;
    int height = 18;
    int maxSteps = 600;
    int threads = 0;
    int playerCount = 2;
    bool remainIndex = true;
    std::string outputPath = "rl/datasets/xrz_imitation.jsonl";
    std::vector<std::string> teachers = {"XiaruizeBot"};
    std::vector<std::string> opponents = {"GcBot", "SmartRandomBot", "oimbot"};
    std::vector<BoardSize> randomBoardSizes;
    std::vector<std::string> mapPaths;
    std::vector<LoadedBoard> customBoards;
};

struct DumpStats {
    std::atomic<long long> gamesFinished{0};
    std::atomic<long long> teacherWins{0};
    std::atomic<long long> samplesConsidered{0};
    std::atomic<long long> samplesRecorded{0};
    std::atomic<long long> samplesOutOfSet{0};
};

bool parsePositiveInt(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool parseBoardSize(const std::string& text, BoardSize& value) {
    const std::size_t delimiter = text.find_first_of("xX");
    if (delimiter == std::string::npos || delimiter == 0 ||
        delimiter + 1 >= text.size()) {
        return false;
    }

    int width = 0;
    int height = 0;
    const std::string widthText = text.substr(0, delimiter);
    const std::string heightText = text.substr(delimiter + 1);
    if (!parsePositiveInt(widthText.c_str(), width) ||
        !parsePositiveInt(heightText.c_str(), height)) {
        return false;
    }

    value.width = width;
    value.height = height;
    return true;
}

std::string boardSizeLabel(const BoardSize& boardSize) {
    std::ostringstream out;
    out << boardSize.width << 'x' << boardSize.height << " random";
    return out.str();
}

void printUsage() {
    std::cout
        << "Usage: LocalGen-bot-imitation-dump [options]\n"
        << "  --games N            Number of games to dump (default: 64)\n"
        << "  --width N            Random map width (default: 18)\n"
        << "  --height N           Random map height (default: 18)\n"
        << "  --steps N            Maximum half-turn steps per game (default: "
           "600)\n"
        << "  --threads N          CPU worker threads (default: auto)\n"
        << "  --players N          Total players per game including the "
           "teacher (default: 2)\n"
        << "  --teacher NAME       Teacher bot name (single-teacher "
           "compatibility mode)\n"
        << "  --teachers A B ...   Teacher bot pool\n"
        << "  --output PATH        JSONL output path (default: "
           "rl/datasets/xrz_imitation.jsonl)\n"
        << "  --size WxH           Add a random-map size to the rotation\n"
        << "  --sizes A B ...      Replace the random-map rotation with "
           "explicit sizes\n"
        << "  --map PATH           Add a custom .lgmp map to the rotation\n"
        << "  --maps A B ...       Replace the custom-map rotation with "
           "explicit paths\n"
        << "  --shuffle            Randomize player index mapping\n"
        << "  --opponents A B ...  Opponent pool (default: GcBot "
           "SmartRandomBot oimbot)\n";
}

bool parseArgs(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--games" || arg == "--width" || arg == "--height" ||
            arg == "--players" || arg == "--steps" || arg == "--threads") {
            if (i + 1 >= argc) return false;
            int value = 0;
            if (!parsePositiveInt(argv[++i], value)) return false;
            if (arg == "--games")
                options.games = value;
            else if (arg == "--width")
                options.width = value;
            else if (arg == "--height")
                options.height = value;
            else if (arg == "--players")
                options.playerCount = value;
            else if (arg == "--threads")
                options.threads = value;
            else
                options.maxSteps = value;
        } else if (arg == "--teacher") {
            if (i + 1 >= argc) return false;
            options.teachers = {argv[++i]};
        } else if (arg == "--teachers") {
            options.teachers.clear();
            while (i + 1 < argc &&
                   std::string(argv[i + 1]).rfind("--", 0) != 0) {
                options.teachers.emplace_back(argv[++i]);
            }
            if (options.teachers.empty()) return false;
        } else if (arg == "--output") {
            if (i + 1 >= argc) return false;
            options.outputPath = argv[++i];
        } else if (arg == "--size") {
            if (i + 1 >= argc) return false;
            BoardSize boardSize;
            if (!parseBoardSize(argv[++i], boardSize)) return false;
            options.randomBoardSizes.push_back(boardSize);
        } else if (arg == "--sizes") {
            options.randomBoardSizes.clear();
            while (i + 1 < argc &&
                   std::string(argv[i + 1]).rfind("--", 0) != 0) {
                BoardSize boardSize;
                if (!parseBoardSize(argv[++i], boardSize)) return false;
                options.randomBoardSizes.push_back(boardSize);
            }
            if (options.randomBoardSizes.empty()) return false;
        } else if (arg == "--map") {
            if (i + 1 >= argc) return false;
            options.mapPaths.emplace_back(argv[++i]);
        } else if (arg == "--maps") {
            options.mapPaths.clear();
            while (i + 1 < argc &&
                   std::string(argv[i + 1]).rfind("--", 0) != 0) {
                options.mapPaths.emplace_back(argv[++i]);
            }
            if (options.mapPaths.empty()) return false;
        } else if (arg == "--shuffle") {
            options.remainIndex = false;
        } else if (arg == "--opponents") {
            options.opponents.clear();
            while (i + 1 < argc &&
                   std::string(argv[i + 1]).rfind("--", 0) != 0) {
                options.opponents.emplace_back(argv[++i]);
            }
            if (options.opponents.empty()) return false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            return false;
        }
    }
    if (options.playerCount < 2) return false;
    if (options.randomBoardSizes.empty() && options.mapPaths.empty()) {
        options.randomBoardSizes.push_back({options.width, options.height});
    }
    return !options.teachers.empty() && !options.opponents.empty();
}

int detectWorkerCount(const Options& options) {
    const unsigned detected = std::thread::hardware_concurrency();
    const int preferred =
        options.threads > 0 ? options.threads : static_cast<int>(detected);
    return std::max(1, std::min(std::max(1, preferred), options.games));
}

std::string serializeSample(
    const std::array<xrz_policy::CandidateAction,
                     xrz_policy::kDatasetActionCount>& actions,
    std::size_t label) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\"action\":" << label << ",\"legal_mask\":[";
    for (std::size_t i = 0; i < actions.size(); ++i) {
        if (i > 0) out << ',';
        out << (actions[i].legal ? "true" : "false");
    }
    out << "],\"action_features\":[";
    for (std::size_t i = 0; i < actions.size(); ++i) {
        if (i > 0) out << ',';
        out << '[';
        for (std::size_t j = 0; j < actions[i].features.size(); ++j) {
            if (j > 0) out << ',';
            out << actions[i].features[j];
        }
        out << ']';
    }
    out << "]}\n";
    return out.str();
}

class RecordingTeacherBot : public BasicBot {
   public:
    RecordingTeacherBot(std::string teacherName, std::ofstream* output,
                        std::mutex* outputMutex, DumpStats* stats)
        : teacherName(std::move(teacherName)),
          output(output),
          outputMutex(outputMutex),
          stats(stats) {
        inner.reset(BotFactory::instance().create(this->teacherName));
        if (!inner) {
            throw std::runtime_error("Failed to create teacher bot: " +
                                     this->teacherName);
        }
    }

    void init(index_t playerId, const GameConstantsPack& constants) override {
        state.init(playerId, constants);
        inner->init(playerId, constants);
    }

    void requestMove(const BoardView& boardView,
                     const std::vector<RankItem>& rank) override {
        moveQueue.clear();
        state.observe(boardView, rank);
        inner->requestMove(boardView, rank);
        const Move teacherMove = inner->step();

        if (teacherMove.type == MoveType::MOVE_ARMY) {
            stats->samplesConsidered.fetch_add(1);
            const auto actions = state.enumerateFixedCandidateActions();
            const std::optional<std::size_t> label =
                state.labelForMove(actions, teacherMove);
            if (label) {
                const std::string payload = serializeSample(actions, *label);
                {
                    std::lock_guard<std::mutex> lock(*outputMutex);
                    (*output) << payload;
                }
                stats->samplesRecorded.fetch_add(1);
            } else {
                stats->samplesOutOfSet.fetch_add(1);
            }
            state.commitSelectedMove(teacherMove);
        }

        if (teacherMove.type != MoveType::EMPTY)
            moveQueue.push_back(teacherMove);
    }

    void onGameEvent(const GameEvent& event) override {
        inner->onGameEvent(event);
    }

   private:
    std::string teacherName;
    std::unique_ptr<BasicBot> inner;
    xrz_policy::PolicyState state;
    std::ofstream* output = nullptr;
    std::mutex* outputMutex = nullptr;
    DumpStats* stats = nullptr;
};

struct GameResult {
    std::string winnerName;
    std::string teacherDisplayName;
};

bool loadCustomMap(const std::string& mapPath, LoadedBoard& board,
                   std::string& errorMessage) {
    const std::filesystem::path mapFilePath(mapPath);
    std::string extension = mapFilePath.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (extension != ".lgmp") {
        errorMessage =
            "Only v6 .lgmp maps are supported by --map/--maps: " + mapPath;
        return false;
    }

    QString qtErrorMessage;
    const MapDocument document =
        openMap_v6(QString::fromStdString(mapPath), qtErrorMessage);
    if (!qtErrorMessage.isEmpty()) {
        errorMessage = "Failed to load map '" + mapPath +
                       "': " + qtErrorMessage.toStdString();
        return false;
    }

    if (document.board.getWidth() <= 0 || document.board.getHeight() <= 0) {
        errorMessage = "Failed to load map '" + mapPath + "': empty board.";
        return false;
    }

    board.path = mapPath;
    board.board = document.board;
    return true;
}

bool loadCustomMaps(Options& options, std::string& errorMessage) {
    options.customBoards.clear();
    options.customBoards.reserve(options.mapPaths.size());
    for (const std::string& mapPath : options.mapPaths) {
        LoadedBoard loadedBoard;
        if (!loadCustomMap(mapPath, loadedBoard, errorMessage)) return false;
        options.customBoards.push_back(std::move(loadedBoard));
    }
    return true;
}

std::mt19937 buildGameRng(int gameNumber) {
    const unsigned seed =
        0xC001CAFEu ^ (static_cast<unsigned>(gameNumber) * 2654435761u);
    return std::mt19937(seed);
}

template <typename T>
const T& chooseFromPool(const std::vector<T>& pool, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, pool.size() - 1);
    return pool[dist(rng)];
}

std::vector<SelectedBot> chooseOpponents(const Options& options,
                                         const std::string& teacherName,
                                         std::mt19937& rng) {
    std::vector<std::string> candidatePool;
    candidatePool.reserve(options.opponents.size());
    for (const std::string& name : options.opponents) {
        if (name != teacherName) candidatePool.push_back(name);
    }
    if (candidatePool.empty()) candidatePool = options.opponents;

    std::vector<SelectedBot> selected;
    selected.reserve(std::max(1, options.playerCount - 1));
    std::vector<std::string> shuffledPool = candidatePool;
    int shuffleRound = 0;
    while (static_cast<int>(selected.size()) < options.playerCount - 1) {
        if (shuffledPool.empty()) break;
        std::shuffle(shuffledPool.begin(), shuffledPool.end(), rng);
        for (const std::string& botName : shuffledPool) {
            if (static_cast<int>(selected.size()) >= options.playerCount - 1) {
                break;
            }
            const int opponentSlot = static_cast<int>(selected.size()) + 1;
            std::ostringstream displayName;
            displayName << botName << " [op" << opponentSlot << ']';
            selected.push_back({botName, displayName.str()});
        }
        ++shuffleRound;
        if (shuffleRound > options.playerCount + 2) break;
    }
    return selected;
}

GameSetup selectGameSetup(const Options& options, int gameNumber) {
    std::mt19937 rng = buildGameRng(gameNumber);

    GameSetup setup;
    setup.teacher.botName = chooseFromPool(options.teachers, rng);
    setup.teacher.displayName = setup.teacher.botName + " [teacher]";
    setup.opponents = chooseOpponents(options, setup.teacher.botName, rng);

    const std::size_t randomBoardCount = options.randomBoardSizes.size();
    const std::size_t customBoardCount = options.customBoards.size();
    const std::size_t totalBoardCount = randomBoardCount + customBoardCount;
    if (totalBoardCount == 0) {
        setup.board = Board::generate(options.width, options.height);
        setup.boardLabel = boardSizeLabel({options.width, options.height});
        return setup;
    }

    std::uniform_int_distribution<std::size_t> boardDist(0,
                                                         totalBoardCount - 1);
    const std::size_t boardIndex = boardDist(rng);
    if (boardIndex < randomBoardCount) {
        const BoardSize& boardSize = options.randomBoardSizes[boardIndex];
        setup.board = Board::generate(boardSize.width, boardSize.height);
        setup.boardLabel = boardSizeLabel(boardSize);
    } else {
        const LoadedBoard& loadedBoard =
            options.customBoards[boardIndex - randomBoardCount];
        setup.board = loadedBoard.board;
        setup.boardLabel = loadedBoard.path;
    }

    return setup;
}

GameResult runSingleGame(const Options& options, int gameNumber,
                         std::ofstream* output, std::mutex* outputMutex,
                         DumpStats* stats) {
    GameResult result;
    const GameSetup setup = selectGameSetup(options, gameNumber);
    result.teacherDisplayName = setup.teacher.displayName;

    std::vector<Player*> players;
    std::vector<index_t> teams;
    std::vector<std::string> names;
    players.reserve(options.playerCount);
    teams.reserve(options.playerCount);
    names.reserve(options.playerCount);

    players.push_back(new RecordingTeacherBot(setup.teacher.botName, output,
                                              outputMutex, stats));
    teams.push_back(0);
    names.push_back(setup.teacher.displayName);

    for (std::size_t index = 0; index < setup.opponents.size(); ++index) {
        const SelectedBot& opponentSpec = setup.opponents[index];
        BasicBot* opponent =
            BotFactory::instance().create(opponentSpec.botName);
        if (!opponent) {
            for (Player* player : players) delete player;
            throw std::runtime_error("Failed to create opponent bot: " +
                                     opponentSpec.botName);
        }
        players.push_back(opponent);
        teams.push_back(static_cast<index_t>(index + 1));
        names.push_back(opponentSpec.displayName);
    }

    BasicGame game(options.remainIndex, players, teams, names, setup.board);
    const int initResult = game.init();
    if (initResult != 0) {
        std::ostringstream err;
        err << "Failed to initialize game " << gameNumber << " on "
            << setup.boardLabel << " (spawn error code " << initResult << ')';
        throw std::runtime_error(err.str());
    }

    int steps = 0;
    while (static_cast<int>(game.getAlivePlayers().size()) > 1 &&
           steps < options.maxSteps) {
        game.step();
        ++steps;
    }

    const std::vector<RankItem> rank = game.ranklist();
    if (!rank.empty()) result.winnerName = game.getName(rank.front().player);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Options options;
    if (!parseArgs(argc, argv, options)) {
        printUsage();
        return 1;
    }

    const auto registeredBots = BotFactory::instance().list();
    auto isRegistered = [&](const std::string& name) {
        return std::find(registeredBots.begin(), registeredBots.end(), name) !=
               registeredBots.end();
    };
    for (const std::string& teacher : options.teachers) {
        if (!isRegistered(teacher)) {
            std::cerr << "Unknown teacher bot: " << teacher << '\n';
            return 2;
        }
    }
    for (const std::string& opponent : options.opponents) {
        if (!isRegistered(opponent)) {
            std::cerr << "Unknown opponent bot: " << opponent << '\n';
            return 3;
        }
    }

    std::string mapError;
    if (!loadCustomMaps(options, mapError)) {
        std::cerr << mapError << '\n';
        return 4;
    }

    const std::filesystem::path outputPath(options.outputPath);
    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }
    std::ofstream output(outputPath, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        std::cerr << "Failed to open output file: " << options.outputPath
                  << '\n';
        return 5;
    }

    const int workerCount = detectWorkerCount(options);
    DumpStats stats;
    std::mutex outputMutex;
    std::mutex errorMutex;
    std::exception_ptr workerError;
    std::atomic<int> nextGame{1};
    std::atomic<bool> stopRequested{false};
    std::vector<std::thread> workers;
    workers.reserve(workerCount);

    std::cout << "Dumping imitation data to " << options.outputPath << "\n"
              << "Teacher pool:";
    for (const std::string& teacher : options.teachers) {
        std::cout << ' ' << teacher;
    }
    std::cout << "\nOpponents:";
    for (const std::string& opponent : options.opponents) {
        std::cout << ' ' << opponent;
    }
    std::cout << "\nPlayers per game: " << options.playerCount << "\nBoards:";
    if (!options.randomBoardSizes.empty()) {
        for (const BoardSize& boardSize : options.randomBoardSizes) {
            std::cout << ' ' << boardSizeLabel(boardSize);
        }
    }
    for (const LoadedBoard& board : options.customBoards) {
        std::cout << ' ' << board.path;
    }
    std::cout << "\nUsing " << workerCount << " worker thread(s).\n";

    auto worker = [&]() {
        while (!stopRequested.load()) {
            const int gameNumber = nextGame.fetch_add(1);
            if (gameNumber > options.games) return;
            try {
                const GameResult result = runSingleGame(
                    options, gameNumber, &output, &outputMutex, &stats);
                if (result.winnerName == result.teacherDisplayName) {
                    stats.teacherWins.fetch_add(1);
                }
                stats.gamesFinished.fetch_add(1);
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(errorMutex);
                    if (!workerError) workerError = std::current_exception();
                }
                stopRequested.store(true);
                return;
            }
        }
    };

    for (int i = 0; i < workerCount; ++i) {
        workers.emplace_back(worker);
    }
    for (std::thread& thread : workers) thread.join();

    output.flush();
    output.close();

    if (workerError) {
        try {
            std::rethrow_exception(workerError);
        } catch (const std::exception& ex) {
            std::cerr << "Dataset dump failed: " << ex.what() << std::endl;
            return 6;
        }
    }

    const double coverage =
        stats.samplesConsidered.load() > 0
            ? static_cast<double>(stats.samplesRecorded.load()) /
                  static_cast<double>(stats.samplesConsidered.load())
            : 0.0;

    std::cout << "Finished " << stats.gamesFinished.load() << " games.\n"
              << "Teacher-side wins: " << stats.teacherWins.load() << "/"
              << options.games << "\n"
              << "Samples recorded: " << stats.samplesRecorded.load() << "/"
              << stats.samplesConsidered.load() << " (coverage " << std::fixed
              << std::setprecision(2) << coverage * 100.0 << "%)\n"
              << "Samples outside candidate set: "
              << stats.samplesOutOfSet.load() << std::endl;

    return 0;
}
