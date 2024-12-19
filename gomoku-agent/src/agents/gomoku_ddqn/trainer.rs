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

        for epoch in 0..epoches {
            println!("epoches: {}", epoch + 1);

            let mut iteration = 0;

            while iteration < train_options.iterations {
                println!("iteration: {}", iteration + 1);

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

                println!("loss: {}", loss.double_value(&[]));
            }

            if let Some(save_path) = &train_options.save_path {
                if let Err(err) = agent.save(save_path) {
                    eprintln!("failed to save agent: {:#?}", err);
                }
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
        let actions = agent.forward_t(&next_boards, false).argmax(1, true);
        let target_qs = target.forward_t(&next_boards, false);
        let target_q = target_qs.gather(1, &actions, false).to_device(Device::Cpu);

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
