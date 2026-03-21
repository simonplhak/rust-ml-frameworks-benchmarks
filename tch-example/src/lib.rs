use common::{BenchmarkConfig, HIDDEN_SIZE_1, INPUT_DIM, OUTPUT_SIZE, RunableModel};
use tch::{Device, Kind, Tensor, nn};

pub struct TchRunner {
    config: BenchmarkConfig,
}

pub struct TchModel {
    model: nn::Sequential,
    device: Device,
}

#[derive(Clone)]
pub struct TchTrainModel {
    model: nn::Sequential,
    vs: std::sync::Arc<std::sync::Mutex<nn::VarStore>>,
    device: Device,
}

pub struct TchDataset {
    train_xs: Vec<f32>,
    train_ys: Vec<usize>,
}

#[derive(Clone)]
pub struct TchBatch {
    xs: Tensor,
    ys: Tensor,
}

pub struct TchOptimizer {
    inner: Option<nn::Optimizer>,
}

impl TchRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        TchRunner { config }
    }

    fn build_model(vs: &nn::Path, device: Device) -> nn::Sequential {
        nn::seq()
            .add(nn::linear(
                vs / "layer1",
                INPUT_DIM as i64,
                HIDDEN_SIZE_1 as i64,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "layer2",
                HIDDEN_SIZE_1 as i64,
                OUTPUT_SIZE as i64,
                Default::default(),
            ))
    }
}

impl RunableModel for TchRunner {
    type TrainModel = TchTrainModel;
    type Model = TchModel;
    type Dataset = TchDataset;
    type Batch = TchBatch;
    type Optimizer = TchOptimizer;

    fn model(&self) -> Self::Model {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let model = Self::build_model(&vs.root(), device);

        TchModel { model, device }
    }

    fn train_model(&self) -> Self::TrainModel {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let model = Self::build_model(&vs.root(), device);

        TchTrainModel {
            model,
            vs: std::sync::Arc::new(std::sync::Mutex::new(vs)),
            device,
        }
    }

    fn optimizer(&self) -> Self::Optimizer {
        TchOptimizer { inner: None }
    }

    fn dataset(&self, xs: &[f32], ys: &[usize]) -> Self::Dataset {
        TchDataset {
            train_xs: xs.to_vec(),
            train_ys: ys.to_vec(),
        }
    }

    fn batch(&self, xs: &[f32], ys: &[usize]) -> Self::Batch {
        let batch_size = ys.len() as i64;
        let device = Device::Cpu;
        let xs_tensor = Tensor::from_slice(xs)
            .view((batch_size, INPUT_DIM as i64))
            .to_kind(Kind::Float)
            .to(device);
        let ys_tensor = Tensor::of_slice(ys)
            .view((batch_size,))
            .to_kind(Kind::Int64)
            .to(device);

        TchBatch {
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
        let vs_guard = train_model.vs.lock().unwrap();
        let mut opt = if optimizer.inner.is_none() {
            nn::Adam::default()
                .build(&vs_guard, self.config.learning_rate)
                .unwrap()
        } else {
            optimizer.inner.take().unwrap()
        };
        drop(vs_guard);

        let logits = train_model.model.forward(&batch.xs);
        let loss = logits.cross_entropy_for_logits(&batch.ys);

        opt.backward_step(&loss);
        optimizer.inner = Some(opt);

        train_model
    }

    fn train(&self, dataset: &Self::Dataset, epochs: usize) -> Self::Model {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let model = Self::build_model(&vs.root(), device);
        let mut opt = nn::Adam::default()
            .build(&vs, self.config.learning_rate)
            .unwrap();

        for _epoch in 0..epochs {
            for batch_start in (0..dataset.train_xs.len()).step_by(self.config.batch_size) {
                let batch_end =
                    std::cmp::min(batch_start + self.config.batch_size, dataset.train_xs.len());
                let batch_size_actual = batch_end - batch_start;

                let xs = &dataset.train_xs[batch_start * INPUT_DIM..batch_end * INPUT_DIM];
                let ys = &dataset.train_ys[batch_start..batch_end];

                let xs_tensor = Tensor::from_slice(xs)
                    .view((batch_size_actual as i64, INPUT_DIM as i64))
                    .to(device);
                let ys_tensor = Tensor::of_slice(ys)
                    .view((batch_size_actual as i64,))
                    .to_kind(Kind::Int64)
                    .to(device);

                let logits = model.forward(&xs_tensor);
                let loss = logits.cross_entropy_for_logits(&ys_tensor);

                opt.backward_step(&loss);
            }
        }

        TchModel { model, device }
    }

    fn predict_single(&self, model: &Self::Model, x: &[f32]) -> usize {
        let xs_tensor = Tensor::from_slice(x)
            .view((1, INPUT_DIM as i64))
            .to(model.device);
        let logits = model.model.forward(&xs_tensor);
        let prediction = logits.argmax(1, false);
        let vec: Vec<i64> = prediction.try_into().unwrap();
        vec[0] as usize
    }

    fn predict_many(&self, model: &Self::Model, x: &[f32]) -> Vec<usize> {
        let num_samples = x.len() / INPUT_DIM;
        let xs_tensor = Tensor::from_slice(x)
            .view((num_samples as i64, INPUT_DIM as i64))
            .to(model.device);
        let logits = model.model.forward(&xs_tensor);
        let predictions = logits.argmax(1, false);
        let vec: Vec<i64> = predictions.try_into().unwrap();
        vec.iter().map(|&v| v as usize).collect()
    }
}
