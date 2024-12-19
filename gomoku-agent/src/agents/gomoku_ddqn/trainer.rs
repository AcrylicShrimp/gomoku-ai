use super::{agent::GomokuDDQNAgent, model::Model};
use crate::{
    agent::Agent,
    replay::{sample_replay, Opponent},
};
use figment::Figment;
use gomoku_core::game::{Game, Turn};
use rand::{seq::IteratorRandom, Rng};
use serde::Deserialize;
use std::{collections::VecDeque, error::Error};
use tch::nn::{Adam, OptimizerConfig};

pub struct GomokuDDQNTrainer;

#[derive(Deserialize)]
pub struct TrainOptions {
    save_path: Option<String>,
    replay_buffer_size: usize,
    batch_size: usize,
    iterations: usize,
    training_steps: usize,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
    gamma: f64,
    learning_rate: f64,
    max_grad_norm: f64,
    tau: f64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            save_path: None,
            replay_buffer_size: 10000,
            batch_size: 32,
            iterations: 100,
            training_steps: 10,
            epsilon: 0.5,
            epsilon_decay: 0.99,
            epsilon_min: 0.01,
            gamma: 0.9,
            learning_rate: 0.0001,
            max_grad_norm: 1.0,
            tau: 0.001,
        }
    }
}

impl GomokuDDQNTrainer {
    pub fn train(
        &mut self,
        agent: &mut GomokuDDQNAgent,
        epoches: usize,
        options: Figment,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let train_options: TrainOptions = options.extract().unwrap_or_default();

        let mut target = Model::new(
            agent.var_store().root().sub("train-target"),
            agent.model().config().clone(),
        );
        target.copy_weights_from(agent.model(), None);

        let mut optimizer =
            Adam::default().build(agent.var_store(), train_options.learning_rate)?;

        let mut rng = rand::thread_rng();
        let mut game = Game::new(15, 5);
        let mut agent_turn = if rng.gen_bool(0.5) {
            Turn::Black
        } else {
            Turn::White
        };
        let mut epsilon = train_options.epsilon;
        let mut replay_buffer = VecDeque::with_capacity(train_options.replay_buffer_size);
        let mut loss_visualizer = loss_visualizer::LossVisualizer::new();

        for epoch in 0..epoches {
            println!("epoches: {}", epoch + 1);

            let mut iteration = 0;

            while iteration < train_options.iterations {
                let (new_game, new_agent_turn, replay_step) =
                    sample_replay(game, agent_turn, agent, Opponent::Random, epsilon);

                game = new_game;
                agent_turn = new_agent_turn;

                // skip if the turn is not the agent's turn
                if replay_step.turn != agent_turn {
                    continue;
                }

                if !replay_buffer.is_empty()
                    && train_options.replay_buffer_size <= replay_buffer.len()
                {
                    replay_buffer.pop_front();
                }

                replay_buffer.push_back(replay_step);

                epsilon *= train_options.epsilon_decay;
                epsilon = epsilon.max(train_options.epsilon_min);

                iteration += 1;
            }

            for _ in 0..train_options.training_steps {
                let batch = if train_options.batch_size <= replay_buffer.len() {
                    replay_buffer
                        .iter()
                        .choose_multiple(&mut rng, train_options.batch_size)
                } else {
                    replay_buffer.iter().collect()
                };

                optimizer.zero_grad();

                let loss = loss::compute_loss(agent.model(), &target, &batch, train_options.gamma);
                loss.backward();

                optimizer.clip_grad_norm(train_options.max_grad_norm);
                optimizer.step();

                target.copy_weights_from(agent.model(), Some(train_options.tau));

                loss_visualizer.add(loss.double_value(&[]));
            }

            println!("loss: {}", loss_visualizer.mean());

            if let Some(save_path) = &train_options.save_path {
                if let Err(err) = agent.save(save_path) {
                    eprintln!("failed to save agent: {:#?}", err);
                }
            }

            let (agent_wins, opponent_wins, draws) = eval::evaluate_many(agent, 10);
            println!(
                "agent wins: {}, opponent wins: {}, draws: {}",
                agent_wins, opponent_wins, draws
            );

            if epoch % 10 == 0 {
                let (agent_turn, recent_game, _) = eval::evaluate(agent);
                println!(
                    "recent game [agent={}]:\n{}",
                    agent_turn.name(),
                    recent_game
                );
            }
        }

        Ok(())
    }
}

mod loss {
    use crate::{
        agents::gomoku_ddqn::model::{encode_batched_board, Model},
        replay::ReplayStep,
    };
    use tch::{nn::ModuleT, Device, Kind, Tensor};

    pub fn compute_loss(
        agent: &Model,
        target: &Model,
        batch: &[&ReplayStep],
        gamma: f64,
    ) -> Tensor {
        let td_target = compute_td_target(agent, target, batch, gamma);

        let boards = Vec::from_iter(batch.iter().map(|step| &step.boards));
        let boards = encode_batched_board(&boards);
        let q = agent.forward_t(&boards, false).to_device(Device::Cpu);

        let actions = Vec::from_iter(batch.iter().map(|step| step.action as i64));
        let actions = Tensor::from_slice(&actions);
        let q = q.index_select(1, &actions);

        (td_target - q).square().mean(Kind::Float)
    }

    fn compute_td_target(
        agent: &Model,
        target: &Model,
        batch: &[&ReplayStep],
        gamma: f64,
    ) -> Tensor {
        let r = Vec::from_iter(batch.iter().map(|step| step.reward as f64));
        let r = Tensor::from_slice(&r).view([-1, 1]);

        // NOTE: it is safe to fall back to the current board if the next board is not available,
        // because those wrong q values will be masked out by flags later
        let next_boards = encode_batched_board(&Vec::from_iter(
            batch
                .iter()
                .map(|step| step.next_boards.as_ref().unwrap_or(&step.boards)),
        ));
        let action_values = agent.forward_t(&next_boards, false).to_device(Device::Cpu);
        let action_values: Vec<f64> = action_values.flatten(0, -1).try_into().unwrap();

        let mut legal_actions = Vec::with_capacity(batch.len());

        // apply argmax only to legal moves
        for (i, step) in batch.iter().enumerate() {
            let board = &step.boards.last().unwrap().1;
            let action_values = &action_values[i * board.board_size() * board.board_size()
                ..(i + 1) * board.board_size() * board.board_size()];
            let pairs = Vec::from_iter(
                board
                    .legal_moves()
                    .into_iter()
                    .map(|action| (action, action_values[action])),
            );
            let best_action = pairs
                .iter()
                .max_by(|(_, lhs), (_, rhs)| f64::total_cmp(lhs, rhs))
                .unwrap()
                .0;
            legal_actions.push(best_action as i64);
        }

        let actions = Tensor::from_slice(&legal_actions).view([-1, 1]);
        let target_qs = target.forward_t(&next_boards, false).to_device(Device::Cpu);
        let target_q = target_qs.gather(1, &actions, false);

        // flag for whether the game is done to mask out the future q values
        let is_done =
            Vec::from_iter(
                batch
                    .iter()
                    .map(|step| if step.game_result.is_some() { 1.0 } else { 0.0 }),
            );
        let is_done = Tensor::from_slice(&is_done).view([-1, 1]);

        r + (1.0 - is_done) * gamma * target_q
    }
}

mod loss_visualizer {
    pub struct LossVisualizer {
        losses: Vec<f64>,
    }

    impl LossVisualizer {
        pub fn new() -> Self {
            Self { losses: vec![] }
        }

        pub fn add(&mut self, loss: f64) {
            if 100 <= self.losses.len() {
                self.losses.swap_remove(0);
            }

            self.losses.push(loss);
        }

        pub fn mean(&self) -> f64 {
            if self.losses.is_empty() {
                return 0.0;
            }

            self.losses.iter().sum::<f64>() / self.losses.len() as f64
        }
    }
}

mod eval {
    use crate::{agent::Agent, agents::gomoku_ddqn::agent::GomokuDDQNAgent};
    use gomoku_core::game::{Game, GameResult, Turn};
    use rand::{seq::SliceRandom, Rng};

    pub fn evaluate_many(agent: &mut GomokuDDQNAgent, n: usize) -> (usize, usize, usize) {
        let mut agent_wins = 0;
        let mut opponent_wins = 0;
        let mut draws = 0;

        for _ in 0..n {
            let (agent_turn, _, game_result) = evaluate(agent);

            match game_result {
                GameResult::Win(winner) => {
                    if winner == agent_turn {
                        agent_wins += 1;
                    } else {
                        opponent_wins += 1;
                    }
                }
                GameResult::Draw => {
                    draws += 1;
                }
            }
        }

        (agent_wins, opponent_wins, draws)
    }

    pub fn evaluate(agent: &mut GomokuDDQNAgent) -> (Turn, Game, GameResult) {
        let mut rng = rand::thread_rng();
        let mut game = Game::new(15, 5);
        let agent_turn = if rng.gen_bool(0.5) {
            Turn::Black
        } else {
            Turn::White
        };

        while game.game_result().is_none() {
            let action = if game.turn() == agent_turn {
                agent.next_move(&game).unwrap()
            } else {
                *game.board().legal_moves().choose(&mut rng).unwrap()
            };

            let result = game.place_stone(action).unwrap();

            if let Some(game_result) = result.game_result {
                return (agent_turn, game, game_result);
            }
        }

        (agent_turn, game, GameResult::Draw)
    }
}
