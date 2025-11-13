use serde::{Deserialize, Serialize};

/// Normalize input string by removing spaces, hyphens, underscores, and converting to lowercase.
pub fn normalize_input_string(input: &str) -> String {
    input.trim().replace([' ', '-', '_'], "").to_lowercase()
}

/// Deserialize a serde_json::Value into a specified type T.
///
/// This will panic if deserialization fails, instead of giving a default value.
pub fn serde_from_value<T: serde::de::DeserializeOwned>(v: &serde_json::Value) -> T {
    serde_json::from_value(v.clone()).unwrap()
}

/// Flag for algorithms of Coulomb contribution (J) to Fock operator.
///
/// - `default`: use default algorithm (currently same to `ri`).
/// - `ri`: use RI algorithm with incore fitting integrals; depending to the memory available, it
///   will decide whether to use `ri-incore` or `ri-direct`.
/// - `ri-incore`: use RI algorithm with Cholesky decomposed 3c-2e ERI stored in DRAM.
/// - `ri-direct`: use RI algorithm with on-the-fly computation of 3c-2e ERI, no need to store them
///   in DRAM.
#[non_exhaustive]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum AlgJ {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
}

impl std::fmt::Debug for AlgJ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AlgJ::Default => "default",
            AlgJ::Ri => "ri",
            AlgJ::RiIncore => "ri-incore",
            AlgJ::RiDirect => "ri-direct",
        };
        write!(f, "{s}")
    }
}

impl<'de> Deserialize<'de> for AlgJ {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match normalize_input_string(&s).as_str() {
            "ri" => Ok(AlgJ::Ri),
            "riincore" => Ok(AlgJ::RiIncore),
            "ridirect" => Ok(AlgJ::RiDirect),
            "default" | "" => Ok(AlgJ::Default),
            _ => Err(serde::de::Error::custom(format!("unknown AlgJ variant: {s}",))),
        }
    }
}

impl Serialize for AlgJ {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{self:?}"))
    }
}

/// Flag for algorithms of Exchange contribution (K) to Fock operator.
///
/// - `default`: use default algorithm (currently same to `ri`).
/// - `ri`: use RI algorithm with incore fitting integrals; depending to the memory available, it
///   will decide whether to use `ri-incore` or `ri-direct`.
/// - `ri-incore`: use RI algorithm with Cholesky decomposed 3c-2e ERI stored in DRAM.
/// - `ri-direct`: use RI algorithm with on-the-fly computation of 3c-2e ERI, no need to store them
///   in DRAM.
#[non_exhaustive]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum AlgK {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
}

impl std::fmt::Debug for AlgK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AlgK::Default => "default",
            AlgK::Ri => "ri",
            AlgK::RiIncore => "ri-incore",
            AlgK::RiDirect => "ri-direct",
        };
        write!(f, "{s}")
    }
}

impl<'de> Deserialize<'de> for AlgK {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match normalize_input_string(&s).as_str() {
            "ri" => Ok(AlgK::Ri),
            "riincore" => Ok(AlgK::RiIncore),
            "ridirect" => Ok(AlgK::RiDirect),
            "default" | "" => Ok(AlgK::Default),
            _ => Err(serde::de::Error::custom(format!("unknown AlgK variant: {s}",))),
        }
    }
}

impl Serialize for AlgK {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{self:?}"))
    }
}

/// Flag for combined algorithms of Coulomb and Exchange contributions (J and K) to Fock operator.
///
/// - `default`: use default algorithm (currently same to `ri`).
/// - `ri`: use RI algorithm with incore fitting integrals; depending to the memory available, it
///   will decide whether to use `ri-incore` or `ri-direct`.
/// - `ri-incore`: use RI algorithm with Cholesky decomposed 3c-2e ERI stored in DRAM.
/// - `ri-direct`: use RI algorithm with on-the-fly computation of 3c-2e ERI, no need to store them
///   in DRAM.
/// - `separated(alg_j, alg_k)`: use separate algorithms for J and K, specified by `alg_j` (of type
///   [`AlgJ`]) and `alg_k` (of type [`AlgK`]).
#[non_exhaustive]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum AlgJK {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
    Separated(AlgJ, AlgK),
}

impl std::fmt::Debug for AlgJK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AlgJK::Default => "default",
            AlgJK::Ri => "ri",
            AlgJK::RiIncore => "ri-incore",
            AlgJK::RiDirect => "ri-direct",
            AlgJK::Separated(alg_j, alg_k) => &format!("separated({alg_j:?}, {alg_k:?})"),
        };
        write!(f, "{s}")
    }
}

impl<'de> Deserialize<'de> for AlgJK {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match normalize_input_string(&s).as_str() {
            "ri" => Ok(AlgJK::Ri),
            "default" | "" => Ok(AlgJK::Default),
            "riincore" => Ok(AlgJK::RiIncore),
            "ridirect" => Ok(AlgJK::RiDirect),
            _ => Err(serde::de::Error::custom(format!("unknown AlgJK variant: {s}",))),
        }
    }
}

impl Serialize for AlgJK {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{self:?}"))
    }
}
