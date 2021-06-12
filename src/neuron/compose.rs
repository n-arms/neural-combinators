use crate::{Neuron};
use std::convert::TryInto;

pub struct Composition<N1, N2, const I: usize, const O: usize, const O2: usize, const P1: usize, const P2: usize> 
where
    N1: Neuron<I, O, P1>,
    N2: Neuron<O, O2, P2>,
{
    f: N2,
    g: N1,
}

fn dot<const L: usize>(a: [f64;L], b: [f64;L]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .fold(0.0, |acc, x| acc + x)
}

fn transpose<const M: usize, const N: usize>(m: [[f64;M];N]) -> [[f64;N];M] {
    let mut out = [[0.0;N];M];
    for (i, row) in m.iter().enumerate() {
        for (j, e) in row.iter().enumerate() {
            out[j][i] = *e;
        }
    }
    out
}

fn matrix_mul<const M1: usize, const N: usize, const M2: usize>(a: [[f64;N];M1], b: [[f64;M2];N]) -> [[f64;M2];M1] {
    let b_trans = transpose(b);
    a.iter()
        .map(|x| b_trans.iter()
            .map(|y| dot(*x, *y))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap())
        .collect::<Vec<[f64;M2]>>()
        .try_into()
        .unwrap()
}

impl<N1, N2, const I: usize, const O: usize, const O2: usize, const P1: usize, const P2: usize> Neuron<I, O2, {P1 + P2}> for Composition<N1, N2, I, O, O2, P1, P2>
where
    N1: Neuron<I, O, P1>,
    N2: Neuron<O, O2, P2>,
{
    fn eval(&self, input: [f64;I], params: [f64;P1+P2]) -> [f64;O2] {
        let lhs = &params[..P1];
        let rhs = &params[P1..];
        self.f.eval(self.g.eval(input, lhs.try_into().unwrap()), rhs.try_into().unwrap())
    }

    fn derivative(&self, input: [f64;I], params: [f64;P1+P2]) -> [[f64;O2];I] { // a matrix of O2 rows and I columns
        let lhs: [f64;P1] = params[..P1].try_into().unwrap();
        let rhs: [f64;P2] = params[P1..].try_into().unwrap();
        let f_d: [[f64;O2];O] = self.f.derivative(self.g.eval(input, lhs), rhs);
        let g_d: [[f64;O];I] = self.g.derivative(input, lhs);
        transpose(matrix_mul(transpose(f_d), transpose(g_d)))
    }


    fn gradient(&self, input: [f64;I], params: [f64;P1+P2]) -> [[f64;O2];P1+P2] {
        let lhs: [f64;P1] = params[..P1].try_into().unwrap();
        let rhs: [f64;P2] = params[P1..].try_into().unwrap();
        let f_d: [[f64;O2];O] = self.f.derivative(self.g.eval(input, lhs), rhs);
        let g_d: [[f64;O];P1] = self.g.gradient(input, lhs);
        let p1 = transpose(matrix_mul(transpose(f_d), transpose(g_d)));
        let p2 = self.f.gradient(self.g.eval(input, lhs), rhs);
        p1.iter()
            .chain(p2.iter())
            .map(|x| *x)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::neuron::linear::LinearReg;
    use crate::neuron::logistic::LogisticReg;

    #[test]
    fn compose_eval() {
        let c = Composition{f: LogisticReg{}, g: LinearReg{}};
        assert_close!(c.eval([-2.0], [2.0, 3.0])[0], 0.2689414213699951);
        assert_close!(c.eval([-3.0], [5.0, 7.0])[0], 0.0);
    }

    #[test]
    fn compose_derivative() {
        let c = Composition{f: LogisticReg{}, g: LinearReg{}};
        assert_close!(c.derivative([-2.0], [2.0, 3.0])[0][0], 0.3932238664829637);
        assert_close!(c.derivative([-3.0], [5.0, 7.0])[0][0], 0.0);
    }

    #[test]
    fn compose_gradient() {
        let c = Composition{f: LogisticReg{}, g: LinearReg{}};
        let r1 = c.gradient([-2.0], [2.0, 3.0]);
        assert_close!(r1[0][0], -0.3932238664829637);
        assert_close!(r1[1][0], 0.19661193324148185);
        let r2 = c.gradient([-3.0], [5.0, 7.0]);
        assert_close!(r2[0][0], -0.0010042);
        assert_close!(r2[1][0], 0.00033540);
    }

    #[test]
    fn compose_train() {
        let d = vec![
            ([-1.0], [0.119]),
            ([0.0], [0.731]),
            ([1.0], [0.982])
        ];
        let r = Composition{f: LogisticReg{}, g: LinearReg{}}.train(d, 0.2, 5000);
        println!("{:?}", r);
        assert_close!(r[0], 3.0);
        assert_close!(r[1], 1.0);
    }

    #[test]
    fn matrix_mul_test() {
        let m = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let n = [[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(matrix_mul(m, n), [[7.0, 10.0], [15.0, 22.0], [23.0, 34.0]]);
    }
}
