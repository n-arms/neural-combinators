#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr) => {
        if !(($a - $b).abs() < 0.01) {
            panic!("assertion failed: {} != {}", $a, $b);
        }
    }
}

pub trait Neuron<const I: usize, const O: usize, const P: usize> {
    fn eval(&self, input: [f64;I], params: [f64;P]) -> [f64;O];
    fn derivative(&self, input: [f64;I], params: [f64;P]) -> [[f64;O];I];
    fn gradient(&self, input: [f64;I], params: [f64;P]) -> [[f64;O];P];
    #[allow(unused_variables)]
    fn train(&self, data: Vec<([f64;I], [f64;O])>, eta: f64, iterations: usize) -> [f64;P] {
        let mut params = [1.0;P];
        for _ in 0..iterations {
            for d in &data {
                let e = self.eval(d.0, params.clone());
                let g = self.gradient(d.0, params.clone());
                let e2: Vec<_> = e.iter()
                    .zip(d.1.iter())
                    .map(|(x, y)| 2.0 * x - 2.0 * y)
                    .collect();
                let loss_prime: Vec<_> = g.iter()
                    .map(|x| e2.iter()
                         .zip(x.iter())
                         .map(|(x, y)| x * y)
                         .fold(0.0, |acc, x| acc + x))
                    .collect();
                for i in 0..params.len() {
                    params[i] -= loss_prime[i] * eta;
                }
            }
        }
        params
    }
}
