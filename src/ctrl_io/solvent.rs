use crate::constants::solvent::{solvent_data, SUPPORTED_SOLVENT_NAMES};

/// Parse the `solvent` keyword and resolve it to a name, epsilon, and descriptors.
///
/// Panics if:
/// - Both `solvent` and `solv_epsilon` are explicitly set (conflict)
/// - Both `solvent` and `solvent_descriptors` are explicitly set (conflict)
/// - The solvent name is not recognized
///
/// Returns `Some((name, eps, [n, n25, alpha, beta, gamma, eps, phi, psi]))`
/// if the `solvent` keyword is present, `None` if absent or empty.
pub fn parse_solvent_name(
    tmp_ctrl: &serde_json::Map<String, serde_json::Value>,
) -> Option<(String, f64, [f64; 8])> {
    let name = match tmp_ctrl.get("solvent").unwrap_or(&serde_json::Value::Null) {
        serde_json::Value::String(s) => s.trim().to_string(),
        _ => return None,
    };
    if name.is_empty() {
        return None;
    }

    let has_explicit_epsilon = matches!(
        tmp_ctrl.get("solv_epsilon").unwrap_or(&serde_json::Value::Null),
        serde_json::Value::String(_) | serde_json::Value::Number(_)
    );
    if has_explicit_epsilon {
        panic!(
            "Error: Both `solvent = \"{}\"` and `solv_epsilon` are specified. \
             Please use only one: either `solvent` (name-based lookup) or `solv_epsilon` (explicit value).",
            name
        );
    }

    let has_explicit_descriptors = matches!(
        tmp_ctrl.get("solvent_descriptors").unwrap_or(&serde_json::Value::Null),
        serde_json::Value::Array(_)
    );
    if has_explicit_descriptors {
        panic!(
            "Error: Both `solvent = \"{}\"` and `solvent_descriptors` are specified. \
             Please use only one: either `solvent` (name-based lookup) or `solvent_descriptors` (explicit array).",
            name
        );
    }

    

    match solvent_data(&name) {
        Some(data) => Some((name, data[5], data)),
        None => panic!(
            "Error: Unrecognized solvent name \"{}\". Supported solvents include:\n  {}",
            name, SUPPORTED_SOLVENT_NAMES
        ),
    }
}