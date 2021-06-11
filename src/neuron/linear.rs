pub struct LinearReg;
use crate::neuron::neuron::*;

impl Neuron<1, 1, 2> for LinearReg {
    fn eval(&self, input: [f64;1], params: [f64;2]) -> [f64;1] {
        [params[0] * input[0] + params[1]]
    }

    fn derivative(&self, _: [f64;1], params: [f64;2]) -> [f64;1] {
        [params[0]]
    }

    fn gradient(&self, input: [f64;1], _: [f64;2]) -> [[f64;1];2] {
        [[input[0]], [1.0]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linear_eval() {
        assert_eq!(LinearReg{}.eval([2.0], [3.0, 2.0]), [8.0]); 
        assert_eq!(LinearReg{}.eval([5.0], [7.0, 11.0]), [46.0]);
    }

    #[test]
    fn linear_derivative() {
        assert_eq!(LinearReg{}.derivative([2.0], [3.0, 2.0]), [3.0]); 
        assert_eq!(LinearReg{}.derivative([5.0], [7.0, 11.0]), [7.0]);
    }

    #[test]
    fn linear_gradient() {
        assert_eq!(LinearReg{}.gradient([2.0], [3.0, 2.0]), [[2.0], [1.0]]);
        assert_eq!(LinearReg{}.gradient([5.0], [7.0, 11.0]), [[5.0], [1.0]]);
    }

    #[test]
    fn linear_train() {
        let d = vec![
            ([-1.0], [-2.0]),
            ([0.0], [1.0]),
            ([1.0], [4.0])
        ];
        let r = LinearReg{}.train(d, 0.1, 100);
        println!("{:?}", r);
        assert_close!(r[0], 3.0);
        assert_close!(r[1], 1.0);
    }
}
