use crate::agent::Agent;

pub trait AgentProvider {
    fn name(&self) -> String;
    fn create_agent(&self) -> Box<dyn Agent>;
}
