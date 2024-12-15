use std::{iter::Peekable, str::Chars};

#[derive(Debug, Clone)]
pub struct IndexParser<'a> {
    size: usize,
    chars: Peekable<Chars<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index {
    pub row: usize,
    pub column: usize,
}

impl Index {
    pub fn is_valid(self, size: usize) -> bool {
        self.row < size && self.column < size
    }

    pub fn to_index(self, size: usize) -> usize {
        self.row * size + self.column
    }
}

impl<'a> IndexParser<'a> {
    pub fn new(size: usize, index: &'a str) -> Self {
        Self {
            size,
            chars: index.chars().peekable(),
        }
    }

    pub fn parse(&mut self) -> Option<Index> {
        self.skip_whitespace();

        if let Some(alpha) = self.read_alpha() {
            return self.parse_pattern_alpha_number(alpha);
        }

        if let Some(number) = self.read_number() {
            self.skip_whitespace();

            if let Some(alpha) = self.read_alpha() {
                return self.parse_pattern_number_alpha(number, alpha);
            }

            if let Some(second_number) = self.read_number() {
                return self.parse_pattern_number_number(number, second_number);
            }

            if self.is_end() {
                return self.parse_pattern_number(number);
            }
        }

        None
    }

    fn parse_pattern_alpha_number(&mut self, alpha: String) -> Option<Index> {
        self.skip_whitespace();

        let number = self.read_number()?;

        self.skip_whitespace();

        if !self.is_end() {
            return None;
        }

        let alpha_index = alpha_to_index(alpha.as_str());
        let number_index = number_to_index(number.as_str()) - 1; // 1-indexed to 0-indexed
        let index = Index {
            row: number_index,
            column: alpha_index,
        };

        if index.is_valid(self.size) {
            return Some(index);
        }

        None
    }

    fn parse_pattern_number_alpha(&mut self, number: String, alpha: String) -> Option<Index> {
        self.skip_whitespace();

        if !self.is_end() {
            return None;
        }

        let number_index = number_to_index(number.as_str()) - 1; // 1-indexed to 0-indexed
        let alpha_index = alpha_to_index(alpha.as_str());
        let index = Index {
            row: number_index,
            column: alpha_index,
        };

        if index.is_valid(self.size) {
            return Some(index);
        }

        None
    }

    fn parse_pattern_number_number(
        &mut self,
        number: String,
        second_number: String,
    ) -> Option<Index> {
        self.skip_whitespace();

        if !self.is_end() {
            return None;
        }

        let number_index = number_to_index(number.as_str()) - 1; // 1-indexed to 0-indexed
        let second_number_index = number_to_index(second_number.as_str()) - 1; // 1-indexed to 0-indexed
        let index = Index {
            row: second_number_index,
            column: number_index,
        };

        if index.is_valid(self.size) {
            return Some(index);
        }

        None
    }

    fn parse_pattern_number(&mut self, number: String) -> Option<Index> {
        self.skip_whitespace();

        if !self.is_end() {
            return None;
        }

        let number_index = number_to_index(number.as_str()) - 1; // 1-indexed to 0-indexed
        let row = number_index / self.size;
        let column = number_index % self.size;
        let index = Index { row, column };

        if index.is_valid(self.size) {
            return Some(index);
        }

        None
    }

    fn is_end(&mut self) -> bool {
        self.chars.peek().is_none()
    }

    /// Reads a string of alphabetic characters from the index.
    ///
    /// It converts the alphabetic characters to lowercase.
    fn read_alpha(&mut self) -> Option<String> {
        let mut alpha = String::new();

        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_alphabetic() {
                alpha.push(c.to_ascii_lowercase());
                self.chars.next();
            } else {
                break;
            }
        }

        if !alpha.is_empty() {
            Some(alpha)
        } else {
            None
        }
    }

    /// Reads a string of numeric characters from the index.
    fn read_number(&mut self) -> Option<String> {
        let mut number = String::new();

        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() {
                number.push(c);
                self.chars.next();
            } else {
                break;
            }
        }

        if !number.is_empty() {
            Some(number)
        } else {
            None
        }
    }

    /// Skips any whitespace characters in the index.
    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() {
                self.chars.next();
            } else {
                break;
            }
        }
    }
}

fn alpha_to_index(lowercased_alpha: &str) -> usize {
    let mut index = 0;

    for c in lowercased_alpha.chars() {
        let c_index = c as usize - b'a' as usize;
        index = index * 26 + c_index;
    }

    index
}

fn number_to_index(number: &str) -> usize {
    number.parse::<usize>().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_alpha_single_digit() {
        let board_size = 15;
        let test_cases = vec![
            ("a1", Index { row: 0, column: 0 }),
            ("A1", Index { row: 0, column: 0 }),
            ("b1", Index { row: 0, column: 1 }),
            ("B1", Index { row: 0, column: 1 }),
            ("a15", Index { row: 14, column: 0 }),
            ("b15", Index { row: 14, column: 1 }),
            ("B15", Index { row: 14, column: 1 }),
            ("b2", Index { row: 1, column: 1 }),
            ("B2", Index { row: 1, column: 1 }),
            (
                "o15",
                Index {
                    row: 14,
                    column: 14,
                },
            ),
            (
                "O15",
                Index {
                    row: 14,
                    column: 14,
                },
            ),
            ("1a", Index { row: 0, column: 0 }),
            ("1A", Index { row: 0, column: 0 }),
            ("2b", Index { row: 1, column: 1 }),
            ("2B", Index { row: 1, column: 1 }),
            (
                "15o",
                Index {
                    row: 14,
                    column: 14,
                },
            ),
            (
                "15O",
                Index {
                    row: 14,
                    column: 14,
                },
            ),
        ];

        for (input, expected) in test_cases {
            let mut parser = IndexParser::new(board_size, input);
            assert_eq!(parser.parse(), Some(expected));
        }
    }

    #[test]
    fn test_parse_numeric_coordinates() {
        let board_size = 15;
        let test_cases = vec![
            ("1 1", Index { row: 0, column: 0 }),
            ("2 2", Index { row: 1, column: 1 }),
            (
                "15 15",
                Index {
                    row: 14,
                    column: 14,
                },
            ),
            ("8 8", Index { row: 7, column: 7 }),
        ];

        for (input, expected) in test_cases {
            let mut parser = IndexParser::new(board_size, input);
            assert_eq!(parser.parse(), Some(expected));
        }
    }

    #[test]
    fn test_parse_single_digit() {
        let board_size = 15;
        let test_cases = vec![
            ("1", Index { row: 0, column: 0 }),
            ("15", Index { row: 0, column: 14 }),
            (
                "225",
                Index {
                    row: 14,
                    column: 14,
                },
            ),
        ];

        for (input, expected) in test_cases {
            let mut parser = IndexParser::new(board_size, input);
            assert_eq!(parser.parse(), Some(expected));
        }
    }

    #[test]
    fn test_parse_with_extra_whitespace() {
        let board_size = 15;
        let test_cases = vec![
            ("  a1  ", Index { row: 0, column: 0 }),
            ("  1  1  ", Index { row: 0, column: 0 }),
            ("B2\t", Index { row: 1, column: 1 }),
            (
                "\t15\t15\t",
                Index {
                    row: 14,
                    column: 14,
                },
            ),
        ];

        for (input, expected) in test_cases {
            let mut parser = IndexParser::new(board_size, input);
            assert_eq!(parser.parse(), Some(expected));
        }
    }

    #[test]
    fn test_parse_invalid_inputs() {
        let board_size = 15;
        let test_cases = vec![
            "",                  // Empty string
            " ",                 // Only whitespace
            "a",                 // Missing number
            "hello, world! 15a", // Invalid format
            "15 1 15",           // Invalid format
        ];

        for input in test_cases {
            let mut parser = IndexParser::new(board_size, input);
            assert_eq!(parser.parse(), None);
        }
    }

    #[test]
    fn test_parse_edge_cases() {
        let board_size = 15;
        let test_cases = vec![
            ("a15", Index { row: 14, column: 0 }),  // Max row
            ("o1", Index { row: 0, column: 14 }),   // Max column
            ("O1", Index { row: 0, column: 14 }),   // Max column uppercase
            ("1 15", Index { row: 14, column: 0 }), // Max row numeric
            ("15 1", Index { row: 0, column: 14 }), // Max column numeric
        ];

        for (input, expected) in test_cases {
            let mut parser = IndexParser::new(board_size, input);
            assert_eq!(parser.parse(), Some(expected));
        }
    }
}
