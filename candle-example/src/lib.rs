use std::rc::Rc;

use candle_core::{DType, Device, IndexOp as _, Tensor, D};
use candle_nn::{linear, loss, ops, Module, Optimizer as _, Sequential, VarBuilder, VarMap};
use common::{BenchmarkConfig, RunableModel, HIDDEN_SIZE_1, INPUT_DIM, OUTPUT_SIZE};

pub struct CandleRunner {
    config: BenchmarkConfig,
    device: Device,
}

impl CandleRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            device: Device::Cpu,
        }
    }
}

pub struct CandleModel {
    network: Sequential,
}

#[derive(Clone)]
pub struct CandleTrainModel {
    network: Rc<Sequential>,
    varmap: VarMap,
}

pub struct CandleDataset {
    xs: Tensor,
    ys: Tensor,
}

#[derive(Clone)]
pub struct CandleBatch {
    xs: Tensor,
    ys: Tensor,
}

fn build_inference_network(device: &Device) -> Sequential {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);

    candle_nn::seq()
        .add(linear(INPUT_DIM, HIDDEN_SIZE_1, vs.pp("layer1")).unwrap())
        .add_fn(|xs| xs.relu())
        .add(linear(HIDDEN_SIZE_1, OUTPUT_SIZE, vs.pp("layer2")).unwrap())
}

fn build_train_network(device: &Device) -> (Sequential, VarMap) {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let network = candle_nn::seq()
        .add(linear(INPUT_DIM, HIDDEN_SIZE_1, vs.pp("layer1")).unwrap())
        .add_fn(|xs| xs.relu())
        .add(linear(HIDDEN_SIZE_1, OUTPUT_SIZE, vs.pp("layer2")).unwrap());

    (network, varmap)
}

impl RunableModel for CandleRunner {
    type TrainModel = CandleTrainModel;
    type Model = CandleModel;
    type Dataset = CandleDataset;
    type Batch = CandleBatch;
    type Optimizer = candle_nn::AdamW;

    fn model(&self) -> Self::Model {
        CandleModel {
            network: build_inference_network(&self.device),
        }
    }

    fn train_model(&self) -> Self::TrainModel {
        let (network, varmap) = build_train_network(&self.device);
        CandleTrainModel {
            network: Rc::new(network),
            varmap,
        }
    }

    fn dataset(&self, xs: &[f32], ys: &[usize]) -> Self::Dataset {
        let xs = Tensor::from_slice(xs, (xs.len() / INPUT_DIM, INPUT_DIM), &self.device).unwrap();
        let labels = ys.len();
        let ys = ys.iter().map(|y| *y as i64).collect::<Vec<_>>();
        let ys = Tensor::from_vec(ys, (labels,), &self.device).unwrap();
        CandleDataset { xs, ys }
    }

    fn batch(&self, xs: &[f32], ys: &[usize]) -> Self::Batch {
        let num_samples = xs.len() / INPUT_DIM;
        let xs_tensor = Tensor::from_slice(xs, (num_samples, INPUT_DIM), &self.device)
            .expect("Failed to create input tensor");
        let ys = ys.iter().map(|y| *y as i64).collect::<Vec<_>>();
        let ys_tensor = Tensor::from_vec(ys, (num_samples,), &self.device)
            .and_then(|t| t.to_dtype(DType::I64))
            .expect("Failed to create label tensor");

        CandleBatch {
            xs: xs_tensor,
            ys: ys_tensor,
        }
    }

    fn train_batch(
        &self,
        train_model: Self::TrainModel,
        optimizer: &mut Self::Optimizer,
        batch: Self::Batch,
    ) -> Self::TrainModel {
        let logits = train_model
            .network
            .forward(&batch.xs)
            .expect("Forward pass failed");
        let loss =
            candle_nn::loss::cross_entropy(&logits, &batch.ys).expect("Loss computation failed");
        optimizer
            .backward_step(&loss)
            .expect("Backward step failed");

        train_model
    }

    fn train(&self, dataset: &Self::Dataset, epochs: usize) -> Self::Model {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);

        let model = candle_nn::seq()
            .add(linear(INPUT_DIM, HIDDEN_SIZE_1, vs.pp("layer1")).unwrap())
            .add_fn(|xs| xs.relu())
            .add(linear(HIDDEN_SIZE_1, OUTPUT_SIZE, vs.pp("layer2")).unwrap());
        let optim_config = candle_nn::ParamsAdamW {
            lr: self.config.learning_rate,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), optim_config)
            .expect("Failed to create optimizer");
        let total_samples = dataset.ys.dim(0).unwrap();

        for _ in 0..epochs {
            let mut start = 0;
            while start < total_samples {
                let end = std::cmp::min(start + self.config.batch_size, total_samples);

                let batch_xs = dataset.xs.i(start..end).unwrap();
                let batch_ys = dataset.ys.i(start..end).unwrap();

                let logits = model.forward(&batch_xs).unwrap();
                let loss = candle_nn::loss::cross_entropy(&logits, &batch_ys).unwrap();
                optimizer.backward_step(&loss).unwrap();

                start = end;
            }
        }

        CandleModel { network: model }
    }

    fn predict_single(&self, model: &Self::Model, x: &[f32]) -> usize {
        let xs_tensor = Tensor::from_slice(x, (1, INPUT_DIM), &self.device)
            .expect("Failed to create input tensor");
        let logits = model
            .network
            .forward(&xs_tensor)
            .expect("Forward pass failed");
        let predictions = logits
            .argmax(1)
            .expect("Argmax failed")
            .to_vec1::<u32>()
            .expect("Failed to convert to vec");

        predictions[0] as usize
    }

    fn predict_many(&self, model: &Self::Model, x: &[f32]) -> Vec<usize> {
        let num_samples = x.len() / INPUT_DIM;
        let batch_size = 4096;
        let mut predictions = Vec::with_capacity(num_samples);

        for chunk_start in (0..num_samples).step_by(batch_size) {
            let chunk_end = std::cmp::min(chunk_start + batch_size, num_samples);
            let chunk_size = chunk_end - chunk_start;
            let chunk_xs = &x[chunk_start * INPUT_DIM..chunk_end * INPUT_DIM];

            let xs_tensor = Tensor::from_slice(chunk_xs, (chunk_size, INPUT_DIM), &self.device)
                .expect("Failed to create input tensor");
            let logits = model
                .network
                .forward(&xs_tensor)
                .expect("Forward pass failed");
            let preds = logits
                .argmax(1)
                .expect("Argmax failed")
                .to_vec1::<u32>()
                .expect("Failed to convert to vec");

            predictions.extend(preds.into_iter().map(|p| p as usize));
        }

        predictions
    }

    fn optimizer(&self) -> Self::Optimizer {
        let optim_config = candle_nn::ParamsAdamW {
            lr: self.config.learning_rate,
            weight_decay: 0.0,
            ..Default::default()
        };
        candle_nn::AdamW::new(self.train_model().varmap.all_vars(), optim_config)
            .expect("Failed to create optimizer")
    }
}
