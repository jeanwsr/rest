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
pub enum AlgorithmJ {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
}

impl std::fmt::Debug for AlgorithmJ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AlgorithmJ::Default => "default",
            AlgorithmJ::Ri => "ri",
            AlgorithmJ::RiIncore => "ri-incore",
            AlgorithmJ::RiDirect => "ri-direct",
        };
        write!(f, "{s}")
    }
}

impl<'de> Deserialize<'de> for AlgorithmJ {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match normalize_input_string(&s).as_str() {
            "ri" => Ok(AlgorithmJ::Ri),
            "riincore" => Ok(AlgorithmJ::RiIncore),
            "ridirect" => Ok(AlgorithmJ::RiDirect),
            "default" | "" => Ok(AlgorithmJ::Default),
            _ => Err(serde::de::Error::custom(format!("unknown AlgorithmJ variant: {s}",))),
        }
    }
}

impl Serialize for AlgorithmJ {
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
pub enum AlgorithmK {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
}

impl std::fmt::Debug for AlgorithmK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AlgorithmK::Default => "default",
            AlgorithmK::Ri => "ri",
            AlgorithmK::RiIncore => "ri-incore",
            AlgorithmK::RiDirect => "ri-direct",
        };
        write!(f, "{s}")
    }
}

impl<'de> Deserialize<'de> for AlgorithmK {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match normalize_input_string(&s).as_str() {
            "ri" => Ok(AlgorithmK::Ri),
            "riincore" => Ok(AlgorithmK::RiIncore),
            "ridirect" => Ok(AlgorithmK::RiDirect),
            "default" | "" => Ok(AlgorithmK::Default),
            _ => Err(serde::de::Error::custom(format!("unknown AlgorithmK variant: {s}",))),
        }
    }
}

impl Serialize for AlgorithmK {
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
/// - `separated(algorithm_j, algorithm_k)`: use separate algorithms for J and K, specified by
///   `algorithm_j` (of type [`AlgorithmJ`]) and `algorithm_k` (of type [`AlgorithmK`]).
#[non_exhaustive]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum AlgorithmJK {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
    Separated(AlgorithmJ, AlgorithmK),
}

impl std::fmt::Debug for AlgorithmJK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AlgorithmJK::Default => "default",
            AlgorithmJK::Ri => "ri",
            AlgorithmJK::RiIncore => "ri-incore",
            AlgorithmJK::RiDirect => "ri-direct",
            AlgorithmJK::Separated(algorithm_j, algorithm_k) => &format!("separated({algorithm_j:?}, {algorithm_k:?})"),
        };
        write!(f, "{s}")
    }
}

impl<'de> Deserialize<'de> for AlgorithmJK {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match normalize_input_string(&s).as_str() {
            "ri" => Ok(AlgorithmJK::Ri),
            "default" | "" => Ok(AlgorithmJK::Default),
            "riincore" => Ok(AlgorithmJK::RiIncore),
            "ridirect" => Ok(AlgorithmJK::RiDirect),
            _ => Err(serde::de::Error::custom(format!("unknown AlgorithmJK variant: {s}",))),
        }
    }
}

impl Serialize for AlgorithmJK {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{self:?}"))
    }
}
