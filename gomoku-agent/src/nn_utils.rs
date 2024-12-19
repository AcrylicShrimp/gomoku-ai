use std::{borrow::Borrow, ops::Add};
use tch::{
    nn::{batch_norm2d, conv2d, BatchNorm, Conv2D, ConvConfig, Linear, ModuleT, Path},
    Tensor,
};

#[derive(Debug)]
pub struct ResidualBlock {
    pub conv1: Conv2D,
    pub bn1: BatchNorm,
    pub conv2: Conv2D,
    pub bn2: BatchNorm,
}

pub fn residual_block<'a>(vs: impl Borrow<Path<'a>>, channels: i64) -> ResidualBlock {
    let vs = vs.borrow();
    let conv1 = conv2d(
        vs,
        channels,
        channels,
        3,
        ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let bn1 = batch_norm2d(vs, channels, Default::default());
    let conv2 = conv2d(
        vs,
        channels,
        channels,
        3,
        ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let bn2 = batch_norm2d(vs, channels, Default::default());

    ResidualBlock {
        conv1,
        bn1,
        conv2,
        bn2,
    }
}

impl ModuleT for ResidualBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, train)
            .add(xs)
            .relu()
    }
}

fn blend_weights(lhs: &Tensor, rhs: &Tensor, weight: f64) -> Tensor {
    lhs * (1.0 - weight) + rhs * weight
}

pub fn copy_weights_conv2d(to: &mut Conv2D, from: &Conv2D, weight: f64) {
    to.ws.copy_(&blend_weights(&to.ws, &from.ws, weight));

    if let (Some(bs), Some(from_bs)) = (&mut to.bs, &from.bs) {
        bs.copy_(&blend_weights(bs, from_bs, weight));
    }
}

pub fn copy_weights_batch_norm2d(to: &mut BatchNorm, from: &BatchNorm, weight: f64) {
    to.running_mean
        .copy_(&blend_weights(&to.running_mean, &from.running_mean, weight));
    to.running_var
        .copy_(&blend_weights(&to.running_var, &from.running_var, weight));

    if let (Some(bs), Some(from_bs)) = (&mut to.bs, &from.bs) {
        bs.copy_(&blend_weights(bs, from_bs, weight));
    }

    if let (Some(ws), Some(from_ws)) = (&mut to.ws, &from.ws) {
        ws.copy_(&blend_weights(ws, from_ws, weight));
    }
}

pub fn copy_weights_residual_block(to: &mut ResidualBlock, from: &ResidualBlock, weight: f64) {
    copy_weights_conv2d(&mut to.conv1, &from.conv1, weight);
    copy_weights_batch_norm2d(&mut to.bn1, &from.bn1, weight);
    copy_weights_conv2d(&mut to.conv2, &from.conv2, weight);
    copy_weights_batch_norm2d(&mut to.bn2, &from.bn2, weight);
}

pub fn copy_weights_linear(to: &mut Linear, from: &Linear, weight: f64) {
    to.ws.copy_(&blend_weights(&to.ws, &from.ws, weight));

    if let (Some(bs), Some(from_bs)) = (&mut to.bs, &from.bs) {
        bs.copy_(&blend_weights(bs, from_bs, weight));
    }
}
