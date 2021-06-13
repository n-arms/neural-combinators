use crate::Neuron;

pub struct Split<N1, N2, const I: usize, const O1: usize, const O2: usize, const P1: usize, const P2: usize>
where
    N1: Neuron<I, O1, P1>,
    N2: Neuron<I, O2, P2>,
{
    left: N1,
    right: N2,
}

impl<N1, N2, const I: usize, const O1: usize, const O2: usize, const P1: usize, const P2: usize> Neuron<I, {O1 + O2}, {P1 + P2}> for Split<N1, N2, I, O1, O2, P1, P2> 
where
    N1: Neuron<I, O1, P1>,
    N2: Neuron<I, O2, P2>
{
    fn eval(&self, input: [f64;I], params: [f64;P1+P2]) -> [f64;O1+O2] {
        todo!();
    }

    fn derivative(&self, input: [f64;I], params: [f64;P1+P2]) -> [[f64;O1+O2];I] {
        todo!();
    }

    fn gradient(&self, input: [f64;I], params: [f64;P1+P2]) -> [[f64;O1+O2];P1+P2] {
        todo!();
    }
}
