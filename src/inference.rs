use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
};

use crate::{
    data::{XorBatcher, XorItem},
    training::TrainingConfig,
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: XorItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let correct = item.output;
    let batcher = XorBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.items);

    println!("Predicted {} Expected {}", output, correct);
}
