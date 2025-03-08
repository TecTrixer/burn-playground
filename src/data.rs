use burn::{
    data::{dataloader::batcher::Batcher, dataset::InMemDataset},
    prelude::Backend,
    tensor::{Int, Tensor},
};

#[derive(Clone)]
pub struct XorBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> XorBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct XorBatch<B: Backend> {
    pub items: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct XorItem {
    pub a: i32,
    pub b: i32,
    pub output: i32,
}

impl<B: Backend> Batcher<XorItem, XorBatch<B>> for XorBatcher<B> {
    fn batch(&self, input: Vec<XorItem>) -> XorBatch<B> {
        let items: Vec<_> = input
            .iter()
            .map(|item| Tensor::<B, 1>::from_data([item.a, item.b], &self.device))
            .map(|tensor| tensor.reshape([1, 2]))
            .collect();
        let targets = input
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([item.output], &self.device))
            .collect();
        let items = Tensor::cat(items, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);
        XorBatch { items, targets }
    }
}

pub fn xor_dataset(elements: usize) -> InMemDataset<XorItem> {
    assert!(elements % 4 == 0);
    let mut items = vec![];
    for _ in 0..elements / 4 {
        items.push(XorItem {
            a: 0,
            b: 0,
            output: 0,
        });
        items.push(XorItem {
            a: 0,
            b: 1,
            output: 1,
        });
        items.push(XorItem {
            a: 1,
            b: 0,
            output: 1,
        });
        items.push(XorItem {
            a: 1,
            b: 1,
            output: 0,
        });
    }
    InMemDataset::new(items)
}
