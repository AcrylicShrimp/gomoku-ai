mod index_parser;

use crate::game::Turn;
use index_parser::IndexParser;
use std::{cmp::Reverse, fmt::Display};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Cell {
    Empty,
    Black,
    White,
}

impl Cell {
    pub fn is_empty(self) -> bool {
        matches!(self, Cell::Empty)
    }

    pub fn is_black(self) -> bool {
        matches!(self, Cell::Black)
    }

    pub fn is_white(self) -> bool {
        matches!(self, Cell::White)
    }

    pub fn name(self) -> &'static str {
        match self {
            Cell::Empty => "empty",
            Cell::Black => "black",
            Cell::White => "white",
        }
    }

    pub fn symbol(self) -> char {
        match self {
            Cell::Empty => '.',
            Cell::Black => 'X',
            Cell::White => 'O',
        }
    }
}

#[derive(Debug, Clone)]
pub struct Board {
    board_size: usize,
    cells: Vec<Cell>,
}

impl Board {
    pub fn new(board_size: usize) -> Self {
        let cells = vec![Cell::Empty; board_size * board_size];
        Self { board_size, cells }
    }

    pub fn board_size(&self) -> usize {
        self.board_size
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn legal_moves(&self) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| if cell.is_empty() { Some(index) } else { None })
            .collect()
    }

    pub fn illegal_moves(&self) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| if cell.is_empty() { None } else { Some(index) })
            .collect()
    }

    pub fn get_cell(&self, index: usize) -> Option<Cell> {
        self.cells.get(index).copied()
    }

    pub fn set_cell(&mut self, index: usize, cell: Cell) {
        self.cells[index] = cell;
    }

    /// Parses a string index into a board index.
    ///
    /// The string index is in the format of:
    /// - a1
    /// - 3c
    /// - A 10
    /// - 15 O
    /// - 3 15
    pub fn parse_index(&self, index: &str) -> Option<usize> {
        let mut parser = IndexParser::new(self.board_size, index);
        let index = parser.parse()?;
        Some(index.to_index(self.board_size))
    }

    /// Converts a board index to a position string.
    ///
    /// The position string is in the format of:
    /// - {column}{row}
    ///
    /// Example:
    /// - 0 -> A1
    /// - 1 -> B1
    /// - 15 -> A15
    /// - 16 -> B1
    /// - 25 -> Z1
    pub fn index_to_position(&self, index: usize) -> Option<String> {
        if self.board_size * self.board_size <= index {
            return None;
        }

        let mut x = index % self.board_size;
        let y = index / self.board_size;

        let mut alpha = String::new();

        loop {
            alpha.push((b'A' + (x % 26) as u8) as char);
            x /= 26;

            if x == 0 {
                break;
            }
        }

        Some(format!("{}{}", alpha, y + 1))
    }
}

impl Display for Board {
    /// Returns a string representation of the board with chess-like headers.
    ///
    /// The board is displayed with:
    /// - Column headers: A-O (depending on board size)
    /// - Row numbers: 1-15 (depending on board size)
    /// - "." for empty cells
    /// - "X" for black stones
    /// - "O" for white stones
    ///
    /// Example output for a 3x3 board:
    ///   A B C
    /// 1 . . .
    /// 2 . X .
    /// 3 . . O
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::with_capacity(self.board_size * (self.board_size + 1) * 2);

        // Add column headers (A, B, C, ...)
        result.push_str("   "); // Initial spacing for row numbers
        for x in 0..self.board_size {
            result.push((b'A' + x as u8) as char);
            result.push(' ');
        }
        result.push('\n');

        // Add rows with numbers and cells
        for y in 0..self.board_size {
            // Add row number
            result.push_str(&format!("{:2} ", y + 1));

            // Add cells
            for x in 0..self.board_size {
                let cell = self.cells[y * self.board_size + x];
                result.push(cell.symbol());
                result.push(' '); // Add space between cells
            }
            if y < self.board_size - 1 {
                result.push('\n');
            }
        }

        write!(f, "{}", result)
    }
}

impl Board {
    /// Count the number of consecutive cells in all directions for a given position and turn.
    ///
    /// Returns a vector of counts, sorted in descending order. It is useful to check
    /// how many stones are connected in a given direction.
    ///
    /// The maximum number of elements in the returned vector is 4, which is the number
    /// of directions. Note that zero or one is not included in the returned vector, as they
    /// are not considered as a connection.
    pub fn count_consecutive_cells(&self, index: usize, turn: Turn) -> Vec<usize> {
        let cell = match self.cells.get(index).copied() {
            Some(cell) => cell,
            None => {
                return vec![];
            }
        };

        if cell != turn.into() {
            return vec![];
        }

        let x = (index % self.board_size) as isize;
        let y = (index / self.board_size) as isize;

        let mut results = vec![
            // case 1: horizontal
            1 + self.count_consecutive_cells_in_direction(x + 1, y, cell, 1, 0)
                + self.count_consecutive_cells_in_direction(x - 1, y, cell, -1, 0),
            // case 2: vertical
            1 + self.count_consecutive_cells_in_direction(x, y + 1, cell, 0, 1)
                + self.count_consecutive_cells_in_direction(x, y - 1, cell, 0, -1),
            // case 3: diagonal left-up
            1 + self.count_consecutive_cells_in_direction(x + 1, y - 1, cell, 1, -1)
                + self.count_consecutive_cells_in_direction(x - 1, y + 1, cell, -1, 1),
            // case 4: diagonal right-up
            1 + self.count_consecutive_cells_in_direction(x + 1, y + 1, cell, 1, 1)
                + self.count_consecutive_cells_in_direction(x - 1, y - 1, cell, -1, -1),
        ];

        results.sort_unstable_by_key(|&count| Reverse(count));

        while let Some(&count) = results.last() {
            if count < 2 {
                results.pop();
            } else {
                break;
            }
        }

        results
    }

    fn count_consecutive_cells_in_direction(
        &self,
        x: isize,
        y: isize,
        cell: Cell,
        x_delta: isize,
        y_delta: isize,
    ) -> usize {
        let mut count = 0;
        let mut x = x;
        let mut y = y;

        while x >= 0 && x < self.board_size as isize && y >= 0 && y < self.board_size as isize {
            let index = (y * self.board_size as isize + x) as usize;

            if self.cells[index] != cell {
                return count;
            }

            count += 1;
            x += x_delta;
            y += y_delta;
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_consecutive_cells() {
        // Create a board with some stones placed in various patterns
        let mut board = Board::new(15);

        // Place a horizontal line of black stones
        board.cells[0] = Cell::Black;
        board.cells[1] = Cell::Black;
        board.cells[2] = Cell::Black;
        board.cells[3] = Cell::Black;

        // Place a vertical line of white stones
        board.cells[15] = Cell::White;
        board.cells[30] = Cell::White;
        board.cells[45] = Cell::White;

        // Place a diagonal line of black stones
        board.cells[16] = Cell::Black;
        board.cells[32] = Cell::Black;
        board.cells[48] = Cell::Black;

        println!("{}", board);

        // Test horizontal black line
        let results = board.count_consecutive_cells(0, Turn::Black);
        assert_eq!(results, vec![4, 4]);

        // Test vertical white line
        let results = board.count_consecutive_cells(15, Turn::White);
        assert_eq!(results, vec![3]);

        // Test diagonal black line
        let results = board.count_consecutive_cells(16, Turn::Black);
        assert_eq!(results, vec![4, 2, 2]);

        // Test empty position
        let results = board.count_consecutive_cells(230, Turn::Black);
        assert_eq!(results, vec![]);
    }
}
