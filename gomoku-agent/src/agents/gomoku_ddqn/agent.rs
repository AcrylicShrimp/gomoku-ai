use super::{
    model::{encode_batched_board, Model, ModelConfig},
    trainer::GomokuDDQNTrainer,
};
use crate::{agent::Agent, replay::generate_history_boards};
use figment::Figment;
use gomoku_core::game::Game;
use std::error::Error;
use tch::{
    nn::{ModuleT, VarStore},
    utils::{has_cuda, has_mps, has_vulkan},
    Device, Tensor,
};

#[derive(Debug)]
pub struct GomokuDDQNAgent {
    var_store: VarStore,
    model: Model,
}

impl GomokuDDQNAgent {
    pub fn new(model_config: ModelConfig) -> Self {
        let device = if has_cuda() {
            Device::Cuda(0)
        } else if has_mps() {
            Device::Mps
        } else if has_vulkan() {
            Device::Vulkan
        } else {
            Device::Cpu
        };
        let var_store = VarStore::new(device);
        let model = Model::new(var_store.root().sub("gomoku-ddqn-agent"), model_config);

        Self { var_store, model }
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
}

impl Agent for GomokuDDQNAgent {
    fn save(&self, path: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.var_store.save(path)?;
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.var_store.load(path)?;
        Ok(())
    }

    fn train(
        &mut self,
        epoch: usize,
        options: Figment,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut trainer = GomokuDDQNTrainer;
        trainer.train(self, epoch, options)?;
        Ok(())
    }

    fn next_move(&mut self, game: &Game) -> Result<usize, Box<dyn Error + Send + Sync>> {
        let boards = generate_history_boards(game.turn(), game);
        let input = encode_batched_board(&[&boards]).to_device(self.var_store.device());
        let output = self.model.forward_t(&input, false).to_device(Device::Cpu);

        // filter-out illegal moves
        let legal_moves = Tensor::from_slice(
            &game
                .board()
                .legal_moves()
                .iter()
                .map(|m| *m as i64)
                .collect::<Vec<_>>(),
        );
        let legal_q_values = output.index_select(1, &legal_moves);
        let action = legal_q_values.argmax(1, false).int64_value(&[0]);
        Ok(action as usize)
    }
}
