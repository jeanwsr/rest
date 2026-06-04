use tensors::MatrixFull;

pub trait HessAPI {
    /// Returns the full Hessian matrix as [natm*3, natm*3] in a.u.
    fn get_hessian(&self) -> MatrixFull<f64>;

    /// Returns the electronic contribution e1 + ej - ek (without CP-HF)
    fn get_partial_hessian(&self) -> MatrixFull<f64>;
}
