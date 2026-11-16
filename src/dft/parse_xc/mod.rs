pub mod parse;
mod xc_helper;
mod dispersion;

pub use parse::{func_list_from_components, parse_and_derive, xc_part_of_name, DFAdef, DFAComponent};
pub use xc_helper::{
    ComponentType, get_name, MULTISTEP, MULTISTEP_ALIAS, MULTISTEP_NAME_WITH_DASH,
    NAME_WITH_DASH, NAME_WITHOUT_UNDERSCORE, NSCF_COMPONENTS, WHITELIST_NONLIBXC,
};


// fn main() {
//     // println!("Hello, world!");
//     // get a string from command line
//     let args: Vec<String> = std::env::args().collect();
//     let name = &args[1];
//     // println!("parsing xc: {}", name);
//     let final_results = parse::parse_and_derive(name, 1);
//     // println!("{}", final_results.formatted_output());
//     // final_results.init_libxc();
//     final_results.summary();
// }