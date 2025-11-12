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
            _ => Err(serde::de::Error::custom(format!(
                "unknown AlgJ variant: {s}",
            ))),
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
            _ => Err(serde::de::Error::custom(format!(
                "unknown AlgK variant: {s}",
            ))),
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
            _ => Err(serde::de::Error::custom(format!(
                "unknown AlgJK variant: {s}",
            ))),
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

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub alg_jk: AlgJK,
    #[serde(default)]
    pub alg_j: AlgJ,
    #[serde(default)]
    pub alg_k: AlgK,
}