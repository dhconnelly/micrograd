use rand::Rng;

use crate::value::Value;

fn random_value() -> Value {
    let mut gen = rand::thread_rng();
    let val = gen.sample(rand::distributions::Uniform::new(-1.0, 1.0));
    Value::of(val)
}

pub type Tensor = Vec<Value>;

#[derive(Debug)]
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Neuron {
        let w = std::iter::repeat_with(random_value).take(nin).collect();
        let b = random_value();
        Neuron { w, b }
    }

    pub fn forward(&self, x: &Tensor) -> Value {
        assert_eq!(x.len(), self.w.len());
        let dot_product = x
            .iter()
            .zip(&self.w)
            .map(|(xi, wi)| xi.mul(&wi))
            .fold(Value::of(0.0), |acc, comp| acc.add(&comp));
        let biased = dot_product.add(&self.b);
        biased.tanh()
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Layer {
        let neurons = std::iter::repeat_with(|| Neuron::new(nin))
            .take(nout)
            .collect();
        Layer { neurons }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, first: usize, rest: Vec<usize>) -> MLP {
        let mut layers = Vec::with_capacity(rest.len() + 1);
        layers.push(Layer::new(nin, first));
        let mut nin = first;
        for nout in rest {
            layers.push(Layer::new(nin, nout));
            nin = nout;
        }
        MLP { layers }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn float_equal(x: f64, y: f64) -> bool {
        (x - y).abs() < 0.001
    }

    fn tensor1(xs: &[f64]) -> Tensor {
        xs.iter().map(|x| Value::of(*x)).collect()
    }

    type Tensor2 = Vec<Vec<Value>>;

    fn predict(n: &MLP, xs: &Tensor2) -> Tensor {
        xs.iter().map(|x| n.forward(x)[0].clone()).collect()
    }

    fn rmse(ypred: &Tensor, ys: &Tensor) -> Value {
        ypred
            .iter()
            .zip(ys.iter())
            .map(|(yout, ygt)| yout.sub(ygt).pow(2.0))
            .fold(Value::of(0.0), |acc, l| acc.add(&l))
    }

    fn train(n: &mut MLP, err: f64, step: f64, xs: &Tensor2, ys: &Tensor) {
        let mut ypred = predict(n, xs);
        let mut loss = rmse(&ypred, ys);
        while loss.val() >= err {
            loss.backward();
            for mut p in n.parameters() {
                p.adjust_val(step * -p.grad());
            }
            ypred = predict(n, xs);
            loss = rmse(&ypred, ys);
        }
    }

    #[test]
    fn test_train() {
        let mut n = MLP::new(3, 4, vec![4, 1]);
        let xs = Tensor2::from([
            tensor1(&[2.0, 3.0, -1.0]),
            tensor1(&[3.0, -1.0, 0.5]),
            tensor1(&[0.5, 1.0, 1.0]),
            tensor1(&[1.0, 1.0, -1.0]),
        ]);
        let ys = tensor1(&[1.0, -1.0, -1.0, 1.0]);
        train(&mut n, 0.000000001, 0.01, &xs, &ys);
        let ypred = predict(&n, &xs);
        for i in 0..ys.len() {
            println!("{}", ypred[i]);
            assert!(float_equal(ys[i].val(), ypred[i].val()));
        }
    }
}
