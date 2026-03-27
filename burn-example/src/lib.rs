use std::marker::PhantomData;
use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, Optimizer as _};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};
use common::{BenchmarkConfig, HIDDEN_SIZE_1, INPUT_DIM, OUTPUT_SIZE, RunableModel, SEED};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    relu1: Relu,
    linear2: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(INPUT_DIM, HIDDEN_SIZE_1).init(device);
        let relu1 = Relu::new();
        let linear2 = LinearConfig::new(HIDDEN_SIZE_1, OUTPUT_SIZE).init(device);

        Self {
            linear1,
            relu1,
            linear2,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.relu1.forward(x);
        self.linear2.forward(x)
    }
}

#[derive(Clone, Default)]
pub struct BurnBatcher<B: Backend> {
    _marker: PhantomData<B>,
}

#[derive(Debug, Clone, Copy)]
pub struct BurnItem {
    pub query: [f32; INPUT_DIM],
    pub label: usize,
}

#[derive(Clone, Debug)]
pub struct BurnBatch<B: Backend> {
    pub queries: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, BurnItem, BurnBatch<B>> for BurnBatcher<B> {
    fn batch(&self, items: Vec<BurnItem>, device: &B::Device) -> BurnBatch<B> {
        let queries = items
            .iter()
            .map(|item| TensorData::from(item.query).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let queries = Tensor::stack(queries, 0);
        let targets = Tensor::cat(targets, 0);

        BurnBatch { queries, targets }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        queries: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(queries);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = BurnBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: BurnBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.queries, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = BurnBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: BurnBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.queries, batch.targets)
    }
}

pub struct BurnDataset {
    dataset: Vec<BurnItem>,
}

impl Dataset<BurnItem> for BurnDataset {
    fn get(&self, index: usize) -> Option<BurnItem> {
        self.dataset.get(index).copied()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl BurnDataset {
    pub fn new(dataset: Vec<BurnItem>) -> Self {
        Self { dataset }
    }
}

pub struct BurnModel<B: AutodiffBackend> {
    config: BenchmarkConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> BurnModel<B> {
    pub fn new(config: BenchmarkConfig, device: B::Device) -> Self {
        Self { device, config }
    }
}

impl<B: AutodiffBackend> RunableModel for BurnModel<B> {
    type TrainModel = Model<B>;
    type Model = Model<B::InnerBackend>;
    type Dataset = Arc<dyn DataLoader<B, BurnBatch<B>>>;
    type Batch = BurnBatch<B>;
    type Optimizer = OptimizerAdaptor<Adam, Model<B>, B>;

    fn optimizer(&self) -> Self::Optimizer {
        AdamConfig::new().init()
    }

    fn dataset(&self, xs: &[f32], ys: &[usize]) -> Self::Dataset {
        let dataset = xs
            .chunks(INPUT_DIM)
            .zip(ys)
            .map(|(query, &label)| BurnItem {
                query: query.try_into().unwrap(),
                label,
            })
            .collect::<Vec<_>>();
        let batcher = BurnBatcher::<B>::default();
        DataLoaderBuilder::new(batcher)
            .batch_size(self.config.batch_size)
            .shuffle(SEED)
            .num_workers(self.config.num_workers)
            .build(BurnDataset::new(dataset.to_vec()))
    }

    fn batch(&self, xs: &[f32], ys: &[usize]) -> Self::Batch {
        let queries = xs
            .chunks_exact(INPUT_DIM)
            .map(|chunk| TensorData::from(chunk).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .collect();

        let targets = ys
            .iter()
            .map(|&label| {
                Tensor::<B, 1, Int>::from_data([(label as i64).elem::<B::IntElem>()], &self.device)
            })
            .collect();

        let queries = Tensor::stack(queries, 0);
        let targets = Tensor::cat(targets, 0);

        BurnBatch { queries, targets }
    }

    fn train_batch(
        &self,
        train_model: Self::TrainModel,
        optimizer: &mut Self::Optimizer,
        batch: Self::Batch,
    ) -> Self::TrainModel {
        let output = TrainStep::step(&train_model, batch);
        optimizer.step(self.config.learning_rate, train_model, output.grads)
    }

    fn train(&self, dataset: &Self::Dataset, epochs: usize) -> Self::Model {
        let device = B::Device::default();
        let mut burn_model = Model::<B>::new(&device);
        let mut optim = AdamConfig::new().init();
        for _ in 0..epochs {
            for batch in dataset.iter() {
                let output = TrainStep::step(&burn_model, batch);
                burn_model = optim.step(self.config.learning_rate, burn_model, output.grads);
            }
        }
        burn_model.valid()
    }

    fn predict_single(&self, model: &Self::Model, x: &[f32]) -> usize {
        let input_data = TensorData::from(x).convert::<<B::InnerBackend as Backend>::FloatElem>();
        let input_tensor = Tensor::<B::InnerBackend, 1>::from_data(input_data, &self.device)
            .reshape([1, INPUT_DIM]);
        let output = model.forward(input_tensor);
        let predicted_tensor = output.argmax(1);
        let class_id = predicted_tensor.into_scalar().elem::<i64>();
        class_id as usize
    }

    fn predict_many(&self, model: &Self::Model, x: &[f32]) -> Vec<usize> {
        let batch_size = x.len() / INPUT_DIM;
        let input_data = TensorData::from(x).convert::<B::FloatElem>();
        let input_tensor = Tensor::<B::InnerBackend, 1>::from_data(input_data, &self.device)
            .reshape([batch_size, INPUT_DIM]);
        let output = model.forward(input_tensor);
        let predictions = output.argmax(1);
        let preds_vec: Vec<i64> = predictions
            .into_data()
            .convert::<i64>()
            .to_vec()
            .expect("Failed to convert batched predictions to vector");
        preds_vec.into_iter().map(|id| id as usize).collect()
    }

    fn train_model(&self) -> Self::TrainModel {
        Model::<B>::new(&self.device)
    }

    fn model(&self) -> Self::Model {
        self.train_model().valid()
    }
}

pub fn model_runnner(config: BenchmarkConfig) -> impl RunableModel {
    use burn::backend::{
        Autodiff,
        ndarray::{NdArray, NdArrayDevice},
    };
    let device = NdArrayDevice::Cpu;
    BurnModel::<Autodiff<NdArray<f32, u32>>>::new(config, device)
}
// pub fn model_runnner(config: BenchmarkConfig) -> impl RunableModel {
//     use burn::backend::{
//         Autodiff,
//         Cpu, // 1. Import the CubeCL-powered Cpu backend instead of ndarray
//     };
    
//     // 2. The Cpu backend maps perfectly to your system's processor using Default
//     let device = Default::default(); 
    
//     // 3. Swap the generic type. `Cpu` automatically handles the float/int types 
//     // (defaulting to f32/i32) so you don't need the `<f32, u32>` boilerplate anymore.
//     BurnModel::<Autodiff<Cpu>>::new(config.clone(), device)
// }