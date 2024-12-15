use crate::board::{Board, Cell};
use std::fmt::Display;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Turn {
    Black,
    White,
}

impl Turn {
    pub fn name(self) -> &'static str {
        match self {
            Turn::Black => "black",
            Turn::White => "white",
        }
    }

    pub fn symbol(self) -> char {
        match self {
            Turn::Black => 'X',
            Turn::White => 'O',
        }
    }

    pub fn next(self) -> Self {
        match self {
            Turn::Black => Turn::White,
            Turn::White => Turn::Black,
        }
    }
}

impl From<Turn> for Cell {
    fn from(turn: Turn) -> Self {
        match turn {
            Turn::Black => Cell::Black,
            Turn::White => Cell::White,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GameResult {
    Draw,
    Win(Turn),
}

#[derive(Debug, Clone)]
pub struct Game {
    size: usize,
    max_consecutive_stones: usize,
    turn: Turn,
    turn_count: usize,
    game_result: Option<GameResult>,
    board: Board,
}

impl Game {
    pub fn new(size: usize, max_consecutive_stones: usize) -> Self {
        Self {
            size,
            max_consecutive_stones,
            turn: Turn::Black,
            turn_count: 0,
            game_result: None,
            board: Board::new(size),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn max_consecutive_stones(&self) -> usize {
        self.max_consecutive_stones
    }

    pub fn turn(&self) -> Turn {
        self.turn
    }

    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    pub fn game_result(&self) -> Option<GameResult> {
        self.game_result
    }

    pub fn board(&self) -> &Board {
        &self.board
    }
}

pub struct PlaceStoneResult {
    pub index: usize,
    pub stone: Cell,
    pub turn_was: Turn,
    /// The number of consecutive stones placed by the current player.
    pub consecutive_stones: Vec<usize>,
    pub game_result: Option<GameResult>,
}

#[derive(Error, Debug, Clone)]
pub enum PlaceStoneError {
    #[error("invalid index {index}")]
    InvalidIndex {
        index: usize,
        max_allowed_index: usize,
    },
    #[error("stone already placed at index {index}")]
    StoneAlreadyPlaced { index: usize, stone: Cell },
}

impl Game {
    pub fn place_stone(&mut self, index: usize) -> Result<PlaceStoneResult, PlaceStoneError> {
        let max_allowed_index = self.board.size() * self.board.size();
        let cell = match self.board.get_cell(index) {
            Some(cell) => cell,
            None => {
                return Err(PlaceStoneError::InvalidIndex {
                    index,
                    max_allowed_index,
                });
            }
        };

        if !cell.is_empty() {
            return Err(PlaceStoneError::StoneAlreadyPlaced { index, stone: cell });
        }

        self.board.set_cell(index, self.turn.into());

        let consecutive_stones = self.board.count_consecutive_cells(index, self.turn);
        let is_winning_move =
            consecutive_stones.first().copied() == Some(self.max_consecutive_stones);

        let turn_was = self.turn;
        self.turn = self.turn.next();
        self.turn_count += 1;

        if is_winning_move {
            self.game_result = Some(GameResult::Win(turn_was));
        } else if self.turn_count == max_allowed_index {
            self.game_result = Some(GameResult::Draw);
        }

        Ok(PlaceStoneResult {
            index,
            stone: self.turn.into(),
            turn_was,
            consecutive_stones,
            game_result: self.game_result,
        })
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "turn: {} ({:3})", self.turn.name(), self.turn_count + 1)?;
        writeln!(
            f,
            "state: {}",
            match self.game_result {
                Some(GameResult::Win(turn)) => format!("{} wins", turn.name()),
                Some(GameResult::Draw) => "draw".to_string(),
                None => "in progress".to_string(),
            }
        )?;
        write!(f, "{}", self.board)
    }
}
