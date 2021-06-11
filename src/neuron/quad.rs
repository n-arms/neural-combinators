pub struct QuadReg;
use crate::neuron::neuron::*;

impl Neuron<1, 1, 3> for QuadReg {
    fn eval(&self, input: [f64;1], params: [f64;3]) -> [f64;1] {
        [params[0] * input[0] * input[0] + params[1] * input[0] + params[2]]
    }

    fn derivative(&self, input: [f64;1], params: [f64;3]) -> [f64;1] {
        [params[0] *2.0 * input[0] + params[1]]
    }

    fn gradient(&self, input: [f64;1], _: [f64;3]) -> [[f64;1];3] {
        [[input[0] * input[0]], [input[0]], [1.0]]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn quad_eval() {
        assert_eq!(QuadReg{}.eval([2.0], [2.0, 3.0, 5.0]), [19.0]);
        assert_eq!(QuadReg{}.eval([5.0], [7.0, 11.0, 13.0]), [243.0]);
    }

    #[test]
    fn quad_derivative() {
        assert_eq!(QuadReg{}.derivative([2.0], [2.0, 3.0, 5.0]), [11.0]);
        assert_eq!(QuadReg{}.derivative([5.0], [7.0, 11.0, 13.0]), [81.0]);
    }

    #[test]
    fn quad_gradient() {
        assert_eq!(QuadReg{}.gradient([2.0], [2.0, 3.0, 5.0]), [[4.0], [2.0], [1.0]]);
        assert_eq!(QuadReg{}.gradient([5.0], [7.0, 11.0, 13.0]), [[25.0], [5.0], [1.0]]);
    }

    #[test]
    fn linear_train() {
        let d = vec![
            ([-1.0], [10.0]),
            ([0.0], [5.0]),
            ([1.0], [4.0])
        ];
        let r = QuadReg{}.train(d, 0.1, 100);
        println!("{:?}", r);
        assert_close!(r[0], 2.0);
        assert_close!(r[1], -3.0);
        assert_close!(r[2], 5.0);
    }
}
