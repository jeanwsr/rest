//! Utility trait for Hessian computation.

pub trait HessUtilAPI {
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
