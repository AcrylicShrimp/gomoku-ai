use figment::Figment;
use gomoku_core::game::Game;
use std::error::Error;

pub trait Agent {
    fn save(&self, path: &str) -> Result<(), Box<dyn Error + Send + Sync>>;
    fn load(&mut self, path: &str) -> Result<(), Box<dyn Error + Send + Sync>>;
    fn train(&mut self, epoch: usize, options: Figment)
        -> Result<(), Box<dyn Error + Send + Sync>>;
    fn next_move(&mut self, game: &Game) -> Result<usize, Box<dyn Error + Send + Sync>>;
}
