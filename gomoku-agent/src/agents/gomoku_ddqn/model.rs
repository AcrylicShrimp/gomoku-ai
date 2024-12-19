use crate::nn_utils::{
    copy_weights_batch_norm2d, copy_weights_conv2d, copy_weights_linear,
    copy_weights_residual_block, residual_block, ResidualBlock,
};
use gomoku_core::{
    board::{Board, Cell},
    game::Turn,
};
use std::borrow::Borrow;
use tch::{
    nn::{batch_norm2d, conv2d, linear, BatchNorm, Conv2D, ConvConfig, Linear, ModuleT, Path},
    no_grad, Device, Tensor,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelConfig {
    pub board_size: usize,
    pub residual_blocks: usize,
    pub residual_block_channels: usize,
    pub fc0_channels: usize,
}

#[derive(Debug)]
pub struct Model {
    device: Device,
    config: ModelConfig,
    match_channel_conv: Conv2D,
    match_channel_bn: BatchNorm,
    residual_blocks: Vec<ResidualBlock>,
    fc0: Linear,
    fc1: Linear,
}

impl Model {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: ModelConfig) -> Self {
        let vs = vs.borrow();
        let match_channel_conv = conv2d(
            vs,
            16,
            config.residual_block_channels as i64,
            3,
            ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let match_channel_bn = batch_norm2d(
            vs,
            config.residual_block_channels as i64,
            Default::default(),
        );
        let mut residual_blocks = Vec::with_capacity(config.residual_blocks);

        for _ in 0..config.residual_blocks {
            residual_blocks.push(residual_block(vs, config.residual_block_channels as i64));
        }

        let fc0 = linear(
            vs,
            config.residual_block_channels as i64
                * config.board_size as i64
                * config.board_size as i64,
            config.fc0_channels as i64,
            Default::default(),
        );
        let fc1 = linear(
            vs,
            config.fc0_channels as i64,
            config.board_size as i64 * config.board_size as i64,
            Default::default(),
        );

        Self {
            device: vs.device(),
            config,
            match_channel_conv,
            match_channel_bn,
            residual_blocks,
            fc0,
            fc1,
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Copy weights from another model.
    ///
    /// If `weight` is provided, the weights will be scaled by the given value.
    /// In that case, `1.0` means the weights will be copied as is, and `0.0` means the weights will be
    /// ignored.
    ///
    /// Otherwise, the weights will be blended with the current weights:
    ///
    /// `current_weights * (1 - weight) + from_weights * weight`
    ///
    /// If `weight` is not provided, the weights will be copied as is.
    pub fn copy_weights_from(&mut self, from: &Model, weight: Option<f64>) {
        let weight = weight.unwrap_or(1.0);

        copy_weights_conv2d(
            &mut self.match_channel_conv,
            &from.match_channel_conv,
            weight,
        );
        copy_weights_batch_norm2d(&mut self.match_channel_bn, &from.match_channel_bn, weight);

        for (block_to, block_from) in self
            .residual_blocks
            .iter_mut()
            .zip(from.residual_blocks.iter())
        {
            copy_weights_residual_block(block_to, block_from, weight);
        }

        copy_weights_linear(&mut self.fc0, &from.fc0, weight);
        copy_weights_linear(&mut self.fc1, &from.fc1, weight);
    }
}

impl ModuleT for Model {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut x = xs
            .to_device(self.device)
            .view([
                -1,
                16,
                self.config.board_size as i64,
                self.config.board_size as i64,
            ])
            .apply(&self.match_channel_conv)
            .apply_t(&self.match_channel_bn, train)
            .relu();

        for block in self.residual_blocks.iter() {
            x = x.apply_t(block, train);
        }

        x.flatten(1, -1).apply(&self.fc0).relu().apply(&self.fc1)
    }
}

pub fn encode_batched_board(boards: &[&[(Turn, Board); 4]]) -> Tensor {
    no_grad(|| {
        let encoded = Tensor::zeros([boards.len() as i64, 16, 15, 15], tch::kind::FLOAT_CPU);

        for (i, boards) in boards.iter().enumerate() {
            let board_tensor = create_board_tensor(boards);
            encoded
                .slice(0, i as i64, (i + 1) as i64, 1)
                .copy_(&board_tensor);
        }

        encoded
    })
}

fn create_board_tensor(boards: &[(Turn, Board); 4]) -> Tensor {
    let encoded = Tensor::zeros([1, 16, 15, 15], tch::kind::FLOAT_CPU);

    for (i, (turn, board)) in boards.iter().enumerate() {
        let encoded = encoded.slice(1, i as i64 * 4, (i as i64 + 1) * 4, 1);

        let point_of_view = (*turn).into();
        let turn = match turn {
            Turn::Black => 1f64,
            Turn::White => -1f64,
        };
        let _ = encoded.slice(1, 0, 1, 1).fill_(turn);

        let mut data = vec![0f32; 3 * 15 * 15];

        for (i, cell) in board.cells().iter().enumerate() {
            let offset = match cell {
                Cell::Empty => 0,
                &cell => {
                    if cell == point_of_view {
                        1
                    } else {
                        2
                    }
                }
            };
            data[(offset * 15 * 15) + i] = 1f32;
        }

        encoded
            .slice(1, 1, 4, 1)
            .copy_(&Tensor::from_slice(&data).view([1, 3, 15, 15]));
    }

    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use gomoku_core::game::Game;
    use tch::nn::VarStore;

    #[test]
    fn test_encode_batched_board() {
        let mut boards = Vec::with_capacity(4);
        let mut game = Game::new(15, 5);

        let result = game.place_stone(0).unwrap();
        boards.push((result.turn_was, result.board_was));

        let result = game.place_stone(1).unwrap();
        boards.push((result.turn_was, result.board_was));

        let result = game.place_stone(2).unwrap();
        boards.push((result.turn_was, result.board_was));

        let result = game.place_stone(3).unwrap();
        boards.push((result.turn_was, result.board_was));

        let boards = boards.try_into().unwrap();
        let encoded = encode_batched_board(&[&boards]);
        encoded.print();
    }

    #[test]
    fn test_model_cpu() {
        let vs = VarStore::new(tch::Device::Cpu);
        let model = Model::new(
            vs.root(),
            ModelConfig {
                board_size: 15,
                residual_blocks: 2,
                residual_block_channels: 32,
                fc0_channels: 32,
            },
        );

        let batch = 16;
        let xs =
            Tensor::randn([batch, 16 * 15 * 15], tch::kind::FLOAT_CPU).to_device(tch::Device::Cpu);
        let q = model.forward_t(&xs, true);

        assert_eq!(q.size(), &[batch, 15 * 15]);
        q.to_device(tch::Device::Cpu).print();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_model_mps() {
        let vs = VarStore::new(tch::Device::Mps);
        let model = Model::new(
            vs.root(),
            ModelConfig {
                board_size: 15,
                residual_blocks: 2,
                residual_block_channels: 32,
                fc0_channels: 32,
            },
        );

        let batch = 16;
        let xs =
            Tensor::randn([batch, 16 * 15 * 15], tch::kind::FLOAT_CPU).to_device(tch::Device::Mps);
        let q = model.forward_t(&xs, true);

        assert_eq!(q.size(), &[batch, 15 * 15]);
        q.to_device(tch::Device::Cpu).print();
    }
}
