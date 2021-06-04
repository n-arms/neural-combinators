use std::convert::TryInto;

pub trait Neuron<const I: usize, const O: usize, const P: usize> {
    fn eval(&self, input: [f64;I], params: [f64;P]) -> [f64;O];
    fn derivative(&self, input: [f64;I], params: [f64;P]) -> [f64;O];
    fn gradient(&self, input: [f64;I], params: [f64;P]) -> [f64;P];
    fn train(&self, data: Vec<([f64;I], [f64;O])>, eta: f64, iterations: usize) -> [f64;P] {
        vec![0.0;P].try_into().unwrap()
    }
}