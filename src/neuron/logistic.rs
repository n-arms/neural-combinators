pub struct LogisticReg;
use crate::neuron::neuron::*;


impl Neuron<1, 1, 0> for LogisticReg {
    fn eval(&self, input: [f64;1], _: [f64;0]) -> [f64;1] {
        [1.0/(1.0+(-input[0]).exp())]
    }

    fn derivative(&self, input: [f64;1], _: [f64;0]) -> [f64;1] {
        let [e] = self.eval(input, []);
        [e * (1.0 - e)]
    }

    fn gradient(&self, _: [f64;1], _: [f64;0]) -> [[f64;1];0] {
        []
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logistic_eval() {
        assert_close!(LogisticReg{}.eval([2.0], [])[0], 0.88079); 
        assert_close!(LogisticReg{}.eval([5.0], [])[0], 0.99330);
    }

    #[test]
    fn logistic_derivative() {
        assert_close!(LogisticReg{}.derivative([2.0], [])[0], 0.10499); 
        assert_close!(LogisticReg{}.derivative([5.0], [])[0], 0.00664);
    }

    #[test]
    fn logistic_gradient() {
        assert_eq!(LogisticReg{}.gradient([2.0], []), [] as [[f64;1];0]);
        assert_eq!(LogisticReg{}.gradient([5.0], []), [] as [[f64;1];0]);
    }

    #[test]
    fn logistic_train() {
        let d = vec![
            ([-1.0], [-2.0]),
            ([0.0], [1.0]),
            ([1.0], [4.0])
        ];
        let r = LogisticReg{}.train(d, 0.1, 100);
        println!("{:?}", r);
        assert_eq!(r, [] as [f64;0]);
    }
}
