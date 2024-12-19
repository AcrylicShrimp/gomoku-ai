use super::{agent::GomokuDDQNAgent, model::ModelConfig};
use crate::{agent::Agent, agent_provider::AgentProvider};

pub struct GomokuDDQNProvider;

impl AgentProvider for GomokuDDQNProvider {
    fn name(&self) -> String {
        "gomoku-ddqn".to_owned()
    }

    fn create_agent(&self) -> Box<dyn Agent> {
        Box::new(GomokuDDQNAgent::new(ModelConfig {
            board_size: 15,
            residual_blocks: 10,
            residual_block_channels: 128,
            fc0_channels: 128,
        }))
    }
}
