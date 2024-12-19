use figment::{
    providers::{Format, Toml},
    Figment,
};
use gomoku_agent::{agent_provider::AgentProvider, agents::gomoku_ddqn::GomokuDDQNProvider};

const AGENT_PATH: &str = "agents/test";

fn main() {
    let mut agent = GomokuDDQNProvider.create_agent();

    // if std::fs::exists(format!("{AGENT_PATH}/agent.safetensors")).unwrap() {
    //     agent.load(AGENT_PATH).unwrap();
    // }

    let config = Figment::new().merge(Toml::file(format!("{AGENT_PATH}/config.toml")));
    agent.train(1000000, config).unwrap();
    agent.save(AGENT_PATH).unwrap();
}
