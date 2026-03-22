use common::{BenchmarkConfig, HIDDEN_SIZE_1, INPUT_DIM, OUTPUT_SIZE, RunableModel};
use tch::{
    Device, Kind, Tensor,
    nn::{self, Module, OptimizerConfig as _},
    vision::dataset::Dataset,
};

pub struct TchRunner {
    config: BenchmarkConfig,
}

pub struct TchModel {
    model: nn::Sequential,
    device: Device,
}

#[derive(Clone)]
pub struct TchTrainModel {
    model: std::sync::Arc<nn::Sequential>,
    vs: std::sync::Arc<std::sync::Mutex<nn::VarStore>>,
}

#[derive(Clone)]
pub struct TchBatch {
    xs: std::sync::Arc<Tensor>,
    ys: std::sync::Arc<Tensor>,
}

impl TchRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        TchRunner { config }
    }

    fn build_model(vs: &nn::Path, _device: Device) -> nn::Sequential {
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
    type Dataset = Dataset;
    type Batch = TchBatch;
    type Optimizer = tch::nn::Optimizer;

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
            model: std::sync::Arc::new(model),
            vs: std::sync::Arc::new(std::sync::Mutex::new(vs)),
        }
    }

    fn optimizer(&self) -> Self::Optimizer {
        let model = self.train_model();
        let vs_guard = model.vs.lock().unwrap();
        nn::Adam::default()
            .build(&vs_guard, self.config.learning_rate)
            .unwrap()
    }

    fn dataset(&self, xs: &[f32], ys: &[usize]) -> Dataset {
        let xs = Tensor::from_slice(xs);
        let xs = xs.view((xs.size()[0] / INPUT_DIM as i64, INPUT_DIM as i64));
        let ys = ys.iter().map(|y| *y as i64).collect::<Vec<_>>();
        let ys = Tensor::from_slice(&ys).to_kind(tch::Kind::Int64);
        let options = (xs.kind(), xs.device());
        Dataset {
            train_images: xs,
            train_labels: ys,
            test_images: Tensor::empty(0, options),
            test_labels: Tensor::empty(0, options),
            labels: OUTPUT_SIZE as i64,
        }
    }

    fn batch(&self, xs: &[f32], ys: &[usize]) -> Self::Batch {
        let batch_size = ys.len() as i64;
        let device = Device::Cpu;
        let xs = Tensor::from_slice(xs)
            .view((batch_size, INPUT_DIM as i64))
            .to_kind(Kind::Float)
            .to(device);
        let ys = ys.iter().map(|y| *y as i64).collect::<Vec<_>>();
        let ys = Tensor::from_slice(&ys)
            .view((batch_size,))
            .to_kind(Kind::Int64)
            .to(device);

        TchBatch {
            xs: std::sync::Arc::new(xs),
            ys: std::sync::Arc::new(ys),
        }
    }

    fn train_batch(
        &self,
        train_model: Self::TrainModel,
        optimizer: &mut Self::Optimizer,
        batch: Self::Batch,
    ) -> Self::TrainModel {
        let logits = train_model.model.forward(&batch.xs);
        let loss = logits.cross_entropy_for_logits(&batch.ys);
        optimizer.backward_step(&loss);
        train_model
    }

    fn train(&self, dataset: &Self::Dataset, epochs: usize) -> Self::Model {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let model = self.model();
        let mut opt = nn::Adam::default()
            .build(&vs, self.config.learning_rate)
            .unwrap();

        for _ in 0..epochs {
            for (xs, ys) in dataset.train_iter(self.config.batch_size as i64).shuffle() {
                let loss = model.model.forward(&xs).cross_entropy_for_logits(&ys);
                opt.backward_step(&loss);
            }
        }

        model
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
