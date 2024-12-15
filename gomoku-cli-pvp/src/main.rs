use gomoku_core::game::{Game, GameResult, PlaceStoneError, PlaceStoneResult};
use std::io::Write;

fn main() {
    let mut game = Game::new(15, 5);

    while game.game_result().is_none() {
        println!("===========================");
        println!("{}", game);
        place_stone(&mut game);
    }

    println!("===========================");
    println!("{}", game);
    println!(
        "game result: {}",
        match game.game_result().unwrap() {
            GameResult::Draw => "draw".to_owned(),
            GameResult::Win(winner) => format!("{} wins", winner.name()),
        }
    );
}

fn read_position(game: &Game) -> usize {
    loop {
        println!();
        print!(
            "enter position to place stone for {} ({}): ",
            game.turn().name(),
            game.turn().symbol()
        );
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        let index = match game.board().parse_index(&input) {
            Some(index) => index,
            None => {
                println!("invalid position");
                continue;
            }
        };

        return index;
    }
}

fn place_stone(game: &mut Game) -> PlaceStoneResult {
    loop {
        let index = read_position(game);
        let result = game.place_stone(index);

        match result {
            Ok(result) => {
                return result;
            }
            Err(err) => match err {
                PlaceStoneError::InvalidIndex {
                    index,
                    max_allowed_index,
                } => {
                    println!(
                        "invalid index: {} (max allowed: {})",
                        index, max_allowed_index
                    );
                }
                PlaceStoneError::StoneAlreadyPlaced { index, .. } => {
                    println!(
                        "stone already placed at index: {}",
                        game.board().index_to_position(index).unwrap()
                    );
                }
            },
        }
    }
}
