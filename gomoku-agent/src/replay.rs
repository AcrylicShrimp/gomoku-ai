use crate::agent::Agent;
use gomoku_core::{
    board::Board,
    game::{Game, GameResult, PlaceStoneResult, Turn},
};
use rand::{seq::SliceRandom, Rng};

#[derive(Debug, Clone)]
pub struct ReplayStep {
    pub turn: Turn,
    pub action: usize,
    pub boards: [(Turn, Board); 4],
    pub next_boards: Option<[(Turn, Board); 4]>,
    pub game_result: Option<GameResult>,
    pub reward: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opponent {
    Random,
    SelfPlay,
}

pub fn sample_replay(
    game: Game,
    agent_turn: Turn,
    agent: &mut dyn Agent,
    opponent: Opponent,
    epsilon: f64,
) -> (Game, Turn, ReplayStep) {
    let mut rng = rand::thread_rng();
    let mut game = game;
    let mut agent_turn = agent_turn;

    // start a new game if the current game is finished
    if game.game_result().is_some() {
        let new_game = Game::new(game.board_size(), game.max_consecutive_stones());
        let new_agent_turn = if rng.gen_bool(0.5) {
            Turn::Black
        } else {
            Turn::White
        };

        game = new_game;
        agent_turn = new_agent_turn;
    }

    // let opponent play if it's not the agent's turn
    // NOTE: there is no case where the opponent wins the game at this point
    if game.turn() != agent_turn {
        let action = match opponent {
            Opponent::Random => RandomPlayer::new().generate_move(&game),
            Opponent::SelfPlay => agent.generate_move(&game),
        };
        game.place_stone(action).unwrap();
    }

    // let agent play
    let boards = generate_history_boards(agent_turn, &game);
    let agent_action = if 1e-4 < epsilon && rng.gen_bool(epsilon) {
        let legal_moves = game.board().legal_moves();
        *legal_moves.choose(&mut rng).unwrap()
    } else {
        agent.generate_move(&game)
    };
    let result_after_agent = game.place_stone(agent_action).unwrap();

    // return immediately if the game is finished (agent wins)
    if result_after_agent.game_result.is_some() {
        return (
            game,
            agent_turn,
            ReplayStep {
                turn: result_after_agent.turn_was,
                action: agent_action,
                boards,
                next_boards: None,
                game_result: result_after_agent.game_result,
                reward: 100f32,
            },
        );
    }

    // let opponent play
    let opponent_action = match opponent {
        Opponent::Random => RandomPlayer::new().generate_move(&game),
        Opponent::SelfPlay => agent.generate_move(&game),
    };
    let result_after_opponent = game.place_stone(opponent_action).unwrap();

    // return immediately if the game is finished (opponent wins)
    if result_after_opponent.game_result.is_some() {
        return (
            game,
            agent_turn,
            ReplayStep {
                turn: result_after_opponent.turn_was,
                action: opponent_action,
                boards,
                next_boards: None,
                game_result: result_after_opponent.game_result,
                reward: -100f32,
            },
        );
    }

    // compute reward
    let reward = compute_nonterminal_reward(&result_after_agent);
    let next_boards = Some(generate_history_boards(game.turn(), &game));

    (
        game,
        agent_turn,
        ReplayStep {
            turn: result_after_agent.turn_was,
            action: agent_action,
            boards,
            next_boards,
            game_result: result_after_agent.game_result,
            reward,
        },
    )
}

pub fn generate_history_boards(player: Turn, game: &Game) -> [(Turn, Board); 4] {
    let mut boards = game
        .history()
        .iter()
        .rev()
        .filter(|(turn, _)| *turn == player)
        .take(4)
        .map(|(_, board)| (player, board.clone()))
        .collect::<Vec<_>>();

    while boards.len() < 4 {
        boards.insert(0, (player, Board::new(game.board_size())));
    }

    boards.try_into().unwrap()
}

trait Player {
    fn generate_move(&mut self, game: &Game) -> usize;
}

impl<T> Player for T
where
    T: ?Sized + Agent,
{
    fn generate_move(&mut self, game: &Game) -> usize {
        self.next_move(game).unwrap()
    }
}

struct RandomPlayer {
    rng: rand::rngs::ThreadRng,
}

impl RandomPlayer {
    fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl Player for RandomPlayer {
    fn generate_move(&mut self, game: &Game) -> usize {
        let legal_moves = game.board().legal_moves();
        debug_assert!(!legal_moves.is_empty());
        legal_moves.choose(&mut self.rng).copied().unwrap()
    }
}

fn compute_nonterminal_reward(result: &PlaceStoneResult) -> f32 {
    // +1 for 3-5 consecutive stones (offensive)
    if let Some(n) = result.consecutive_stones.first().copied() {
        if (3..=5).contains(&n) {
            return 1f32;
        }
    }

    // +1 for defensive move (blocking opponent's 4-5 consecutive stones)
    let mut virtual_board = result.board_was.clone();
    virtual_board.set_cell(result.index, result.turn_was.next().into());

    let opponent_consecutive_stones =
        virtual_board.count_consecutive_cells(result.index, result.turn_was.next());
    if let Some(n) = opponent_consecutive_stones.first().copied() {
        if (4..=5).contains(&n) {
            return 1f32;
        }
    }

    0f32
}
