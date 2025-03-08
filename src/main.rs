use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    data::dataset::Dataset,
    optim::AdamConfig,
};
mod data;
mod inference;
mod model;
mod training;
use data::xor_dataset;
use model::ModelConfig;
use training::TrainingConfig;
type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn train(artifact_dir: &str, device: WgpuDevice) {
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );
}
fn test(artifact_dir: &str, device: WgpuDevice) {
    let dataset = xor_dataset(4);
    for idx in 0..dataset.len() {
        crate::inference::infer::<MyBackend>(
            artifact_dir,
            device.clone(),
            dataset.get(idx).unwrap(),
        );
    }
}

#[allow(unused_variables)]
fn debug(artifact_dir: &str, device: WgpuDevice) {}

fn main() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    let args = std::env::args();
    if let Some(kind) = args.skip(1).next() {
        match kind.as_str() {
            "train" => train(artifact_dir, device),
            "test" => test(artifact_dir, device),
            "debug" => debug(artifact_dir, device),
            _ => println!("Unknown execution type, choose either \"train\" or \"test\"."),
        }
    } else {
        println!("Please enter execution type as argument, choose either \"train\" or \"test\".");
    }
}
