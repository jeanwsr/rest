//! Utility trait for analytical derivative module.

/// Stable permutation of the MO index that puts occupied orbitals (occupation > 0)
/// before virtual ones.
///
/// The analytical-derivative CP-SCF and generalized-Fock kernels use `0..nocc` as the
/// occupied block. This is the identity for an Aufbau occupation.
pub(crate) fn occupied_first_permutation(mo_occ: &[f64]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..mo_occ.len()).collect();
    perm.sort_by_key(|&i| mo_occ[i] <= f64::EPSILON);
    perm
}

pub trait AnalDrvBaseAPI {
    /// Get the type name of the implementor.
    ///
    /// The full type name is too verbose. We will use the short type name for display usage.
    fn get_type_name(&self) -> String {
        // the full type name is probably something like
        // crate::mod::struct<lifetime, generic>
        // we will only preserve the `struct` part.
        std::any::type_name::<Self>().split("::").last().unwrap().split('<').next().unwrap().to_string()
    }

    /// Get the full type name of the implementor. For debugging usage.
    fn get_full_type_name(&self) -> String {
        std::any::type_name::<Self>().to_string()
    }
}
