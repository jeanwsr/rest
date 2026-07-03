pub mod vsap;
pub mod c2s;
pub mod solvent;
mod cartesian_gto;

use std::collections::HashMap;

use lazy_static::lazy_static;

pub use crate::constants::vsap::*;
pub use crate::constants::c2s::*;
pub use crate::constants::cartesian_gto::*;

//struct Matrix3x3 {
//    size
//
//}

pub const SPECIES_NAME: [&str; 118] = ["H", "He",
        "Li","Be","B", "C","N", "O", "F", "Ne",
        "Na","Mg","Al","Si","P", "S", "Cl","Ar",
        "K", "Ca","Sc","Ti","V", "Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
        "Rb","Sr","Y", "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I", "Xe",
        "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W", "Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
        "Fr","Ra","Ac","Th","Pa","U", "Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
        ];

// IUPAC 2021 standard atomic weights (Prohaska et al., Pure Appl. Chem., 2022)
// Table 1 column 7: abridged to 5 significant figures; interval elements use conventional single values
// 14 interval elements: H, Li, B, C, N, O, Mg, Si, S, Cl, Ar, Br, Tl, Pb
// Tc, Pm, Po-At, Rn-Ac, Np and beyond: most stable isotope mass
//   AME2020 / NUBASE2020 via IUPAC 2021 Table 2

pub const MASS_CHARGE: [(f64,f64);118] = [
    (1.0080,1.0),       (4.0026,2.0),

    (6.94,3.0),         (9.0122,4.0),
    (10.81,5.0),        (12.011,6.0),       (14.007,7.0),       (15.999,8.0),
    (18.998,9.0),       (20.180,10.0),

    (22.990,11.0),      (24.305,12.0),
    (26.982,13.0),      (28.085,14.0),      (30.974,15.0),      (32.06,16.0),
    (35.45,17.0),       (39.95,18.0),

    (39.098,19.0),      (40.078,20.0),
    (44.956,21.0),      (47.867,22.0),      (50.942,23.0),      (51.996,24.0),
    (54.938,25.0),      (55.845,26.0),      (58.933,27.0),      (58.693,28.0),
    (63.546,29.0),      (65.38,30.0),

    (69.723,31.0),      (72.630,32.0),
    (74.922,33.0),      (78.971,34.0),      (79.904,35.0),      (83.798,36.0),

    (85.468,37.0),      (87.62,38.0),       (88.906,39.0),      (91.224,40.0),
    (92.906,41.0),      (95.95,42.0),       (97.90721,43.0),    (101.07,44.0),
    (102.91,45.0),      (106.42,46.0),      (107.87,47.0),      (112.41,48.0),

    (114.82,49.0),      (118.71,50.0),      (121.76,51.0),      (127.60,52.0),
    (126.90,53.0),      (131.29,54.0),

    (132.91,55.0),      (137.33,56.0),

    (138.91,57.0),      (140.12,58.0),      (140.91,59.0),      (144.24,60.0),
    (144.91276,61.0),   (150.36,62.0),      (151.96,63.0),      (157.25,64.0),
    (158.93,65.0),      (162.50,66.0),      (164.93,67.0),      (167.26,68.0),
    (168.93,69.0),      (173.05,70.0),

    (174.97,71.0),      (178.49,72.0),
    (180.95,73.0),      (183.84,74.0),      (186.21,75.0),      (190.23,76.0),
    (192.22,77.0),      (195.08,78.0),      (196.97,79.0),      (200.59,80.0),

    (204.38,81.0),      (207.2,82.0),       (208.98,83.0),      (208.98243,84.0),
    (209.98715,85.0),   (222.01758,86.0),

    (223.01973,87.0),   (226.02541,88.0),
    (227.02775,89.0),   (232.04,90.0),      (231.04,91.0),      (238.03,92.0),

    (237.04817,93.0),   (244.06420,94.0),   (243.06138,95.0),   (247.07035,96.0),
    (247.07031,97.0),   (251.07959,98.0),   (252.08298,99.0),   (257.09511,100.0),
    (258.09843,101.0),  (259.10100,102.0),

    (262.10962,103.0),  (267.12179,104.0),
    (268.12567,105.0),  (271.13378,106.0),  (270.13337,107.0),  (269.13365,108.0),
    (278.15649,109.0),  (281.16455,110.0),  (282.16934,111.0),  (285.17723,112.0),

    (286.18246,113.0),  (289.19052,114.0),  (289.19397,115.0),  (293.20458,116.0),
    (294.21084,117.0),  (294.21398,118.0),
];

pub const ELEM1ST: [&str;2]  = ["H", "He"];
pub const ELEM2ND: [&str;8]  = ["Li","Be","B", "C", "N", "O", "F", "Ne"];
pub const ELEM3RD: [&str;8]  = ["Na","Mg","Al","Si","P", "S", "Cl","Ar"];
pub const ELEM4TH: [&str;18] = ["K", "Ca","Sc","Ti","V", "Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr"];
pub const ELEM5TH: [&str;18] = ["Rb","Sr","Y", "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I", "Xe"];
pub const ELEM6TH: [&str;32] = ["Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W", "Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"];
pub const ELEM7TH: [&str;32] = ["Fr","Ra","Ac","Th","Pa","U", "Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"];

pub const ELEMTMS: [&str;40] = [
    "Sc","Ti","V", "Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y", "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "La","Hf","Ta","W", "Re","Os","Ir","Pt","Au","Hg",
    "Ac","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn"
];


lazy_static!{
    pub static ref SPECIES_INFO: HashMap<&'static str, &'static (f64,f64)> = {
        let mut m = HashMap::new();
        SPECIES_NAME.iter().zip(MASS_CHARGE.iter()).for_each(|(name,info)| {
            m.insert(*name,info);
        });
        m
    };
}

// use for the inverse (sqrt inverse) of the auxiliary coulomb matrix
pub const AUXBAS_THRESHOLD: f64 = 1.0e-10;
pub const INVERSE_THRESHOLD: f64 = 1.0e-10;
pub const SQRT_THRESHOLD: f64 = 1.0e-10;

pub const CM: f64 = 8065.541;
pub const ANG:f64 = 0.5291772083;
pub const EV: f64 = 27.2113845;
pub const FQ: f64 = 1822.888;
pub const E:  f64 = std::f64::consts::E;
pub const PI: f64 = std::f64::consts::PI;

pub const E5: f64 = 1.0e5;
pub const E6: f64 = 1.0e6;
pub const E7: f64 = 1.0e7;
pub const E8: f64 = 1.0e8;
pub const E9: f64 = 1.0e9;


pub struct DMatrix<const L: usize> {
    size:[usize;2],
    indicing:[usize;2],
    data: [f64; L]
}
pub const ATOM_CONFIGURATION: [[usize; 4]; 119] = [
    [ 0, 0, 0, 0],     //  0  GHOST
    [ 1, 0, 0, 0],     //  1  H
    [ 2, 0, 0, 0],     //  2  He
    [ 3, 0, 0, 0],     //  3  Li
    [ 4, 0, 0, 0],     //  4  Be
    [ 4, 1, 0, 0],     //  5  B
    [ 4, 2, 0, 0],     //  6  C
    [ 4, 3, 0, 0],     //  7  N
    [ 4, 4, 0, 0],     //  8  O
    [ 4, 5, 0, 0],     //  9  F
    [ 4, 6, 0, 0],     // 10  Ne
    [ 5, 6, 0, 0],     // 11  Na
    [ 6, 6, 0, 0],     // 12  Mg
    [ 6, 7, 0, 0],     // 13  Al
    [ 6, 8, 0, 0],     // 14  Si
    [ 6, 9, 0, 0],     // 15  P
    [ 6,10, 0, 0],     // 16  S
    [ 6,11, 0, 0],     // 17  Cl
    [ 6,12, 0, 0],     // 18  Ar
    [ 7,12, 0, 0],     // 19  K
    [ 8,12, 0, 0],     // 20  Ca
    [ 8,12, 1, 0],     // 21  Sc
    [ 8,12, 2, 0],     // 22  Ti
    [ 8,12, 3, 0],     // 23  V
    [ 7,12, 5, 0],     // 24  Cr
    [ 8,12, 5, 0],     // 25  Mn
    [ 8,12, 6, 0],     // 26  Fe
    [ 8,12, 7, 0],     // 27  Co
    [ 8,12, 8, 0],     // 28  Ni
    [ 7,12,10, 0],     // 29  Cu
    [ 8,12,10, 0],     // 30  Zn
    [ 8,13,10, 0],     // 31  Ga
    [ 8,14,10, 0],     // 32  Ge
    [ 8,15,10, 0],     // 33  As
    [ 8,16,10, 0],     // 34  Se
    [ 8,17,10, 0],     // 35  Br
    [ 8,18,10, 0],     // 36  Kr
    [ 9,18,10, 0],     // 37  Rb
    [10,18,10, 0],     // 38  Sr
    [10,18,11, 0],     // 39  Y
    [10,18,12, 0],     // 40  Zr
    [ 9,18,14, 0],     // 41  Nb
    [ 9,18,15, 0],     // 42  Mo
    [10,18,15, 0],     // 43  Tc
    [ 9,18,17, 0],     // 44  Ru
    [ 9,18,18, 0],     // 45  Rh
    [ 8,18,20, 0],     // 46  Pd
    [ 9,18,20, 0],     // 47  Ag
    [10,18,20, 0],     // 48  Cd
    [10,19,20, 0],     // 49  In
    [10,20,20, 0],     // 50  Sn
    [10,21,20, 0],     // 51  Sb
    [10,22,20, 0],     // 52  Te
    [10,23,20, 0],     // 53  I
    [10,24,20, 0],     // 54  Xe
    [11,24,20, 0],     // 55  Cs
    [12,24,20, 0],     // 56  Ba
    [12,24,21, 0],     // 57  La
    [12,24,21, 1],     // 58  Ce
    [12,24,20, 3],     // 59  Pr
    [12,24,20, 4],     // 60  Nd
    [12,24,20, 5],     // 61  Pm
    [12,24,20, 6],     // 62  Sm
    [12,24,20, 7],     // 63  Eu
    [12,24,21, 7],     // 64  Gd
    [12,24,21, 8],     // 65  Tb
    [12,24,20,10],     // 66  Dy
    [12,24,20,11],     // 67  Ho
    [12,24,20,12],     // 68  Er
    [12,24,20,13],     // 69  Tm
    [12,24,20,14],     // 70  Yb
    [12,24,21,14],     // 71  Lu
    [12,24,22,14],     // 72  Hf
    [12,24,23,14],     // 73  Ta
    [12,24,24,14],     // 74  W
    [12,24,25,14],     // 75  Re
    [12,24,26,14],     // 76  Os
    [12,24,27,14],     // 77  Ir
    [11,24,29,14],     // 78  Pt
    [11,24,30,14],     // 79  Au
    [12,24,30,14],     // 80  Hg
    [12,25,30,14],     // 81  Tl
    [12,26,30,14],     // 82  Pb
    [12,27,30,14],     // 83  Bi
    [12,28,30,14],     // 84  Po
    [12,29,30,14],     // 85  At
    [12,30,30,14],     // 86  Rn
    [13,30,30,14],     // 87  Fr
    [14,30,30,14],     // 88  Ra
    [14,30,31,14],     // 89  Ac
    [14,30,32,14],     // 90  Th
    [14,30,31,16],     // 91  Pa
    [14,30,31,17],     // 92  U
    [14,30,31,18],     // 93  Np
    [14,30,30,20],     // 94  Pu
    [14,30,30,21],     // 95  Am
    [14,30,31,21],     // 96  Cm
    [14,30,31,22],     // 97  Bk
    [14,30,30,24],     // 98  Cf
    [14,30,30,25],     // 99  Es
    [14,30,30,26],     //100  Fm
    [14,30,30,27],     //101  Md
    [14,30,30,28],     //102  No
    [14,30,31,28],     //103  Lr
    [14,30,32,28],     //104  Rf
    [14,30,33,28],     //105  Db
    [14,30,34,28],     //106  Sg
    [14,30,35,28],     //107  Bh
    [14,30,36,28],     //108  Hs
    [14,30,37,28],     //109  Mt
    [14,30,38,28],     //110  Ds
    [14,30,39,28],     //111  Rg
    [14,30,40,28],     //112  Cn
    [14,31,40,28],     //113  Nh
    [14,32,40,28],     //114  Fl
    [14,33,40,28],     //115  Mc
    [14,34,40,28],     //116  Lv
    [14,35,40,28],     //117  Ts
    [14,36,40,28],     //118  Og
];

// =========== libcint ===================================
// for the bas index - libcint
pub const BAS_ATM: usize = 0;
pub const BAS_ANG: usize = 1;
pub const BAS_PRM: usize = 2;
pub const BAS_CTR: usize = 3;
pub const BAS_SLOTS: usize = 6;

// for the atm index - libcint 
pub const ATM_NUC: usize = 0;
pub const ATM_ENV: usize = 1;
pub const ATM_NUC_MOD_OF: usize = 2;
pub const ATM_FRAC_CHARGE_OF: usize = 3;
pub const ATM_SLOTS: usize = 6;

// for ECP - libcint
pub const ECP_LMAX: i32 = 5;
pub const NUC_ECP:  i32 = 4;

// for exp cutoff -libcint
pub const PTR_EXPCUTOFF: i32 = 0;
// for dipole - libcint
pub const PTR_COMMON_ORG: i32 = 1;
// for Gauge origin
pub const PTR_RINV_ORIG: i32 = 4;

pub const NUC_MOD_OF: i32 = 2;

pub const NUC_STAD_CHARGE: i32 = 1;
pub const NUC_GAUS_CHARGE: i32 = 2;
pub const NUC_FRAC_CHARGE: i32 = 3;
// =========== libcint ===================================


//// for the atm index - libcint
//pub const ATM_CHARGE_OF: usize = 0;
//pub const ATM_PRT_COORD: usize = 1;
//pub const ATM_NUC_MOD_OF: usize = 2;
//pub const ATM_PRT_ZETA: usize = 3;

pub const ENV_PRT_START: usize = 20;


// SAD and ECP configuration
pub const S_SHELL: [f64; 1] = [2.0];
pub const P_SHELL: [f64; 3] = [2.0, 2.0, 2.0];
pub const D_SHELL: [f64; 5] = [2.0, 2.0, 2.0, 2.0, 2.0];
pub const F_SHELL: [f64; 7] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
pub const XE_SHELL: [f64; 27] =
//   1s   2s   2p   2p   2p   3s   3p   3p   3p   4s   3d   3d   3d   3d   3d   4p   4p   4p   5s   4d   4d   4d   4d   4d   5p   5p   5p  
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
pub const KR_SHELL: [f64; 18] =
//   1s   2s   2p   2p   2p   3s   3p   3p   3p   4s   3d   3d   3d   3d   3d   4p   4p   4p  
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

//                                    1s   2s   2p   3s   3p   4s   3d    4p   5s   4d    5p   4f    5d    6s   6p 
pub const NELE_IN_SHELLS: [f64; 15] = [2.0, 2.0, 6.0, 2.0, 6.0, 2.0, 10.0, 6.0, 2.0, 10.0, 6.0, 14.0, 10.0, 2.0, 6.0];

//
pub const LIGHT_SPEED: f64 = 137.03599967994;   // http://physics.nist.gov/cgi-bin/cuu/Value?alph
// BOHR = .529 177 210 92(17) e-10m  // http://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
pub const BOHR: f64 = 0.52917721092;  // Angstroms
pub const BOHR_SI: f64 = BOHR * 1e-10;

pub const G_ELECTRON: f64 = 2.00231930436182;  // http://physics.nist.gov/cgi-bin/cuu/Value?gem
pub const E_MASS: f64 = 9.10938356e-31;         // kg https://physics.nist.gov/cgi-bin/cuu/Value?me
pub const AVOGADRO: f64 = 6.022140857e23;       // https://physics.nist.gov/cgi-bin/cuu/Value?na
pub const PLANCK: f64 = 6.626070040e-34;        // J*s http://physics.nist.gov/cgi-bin/cuu/Value?h
pub const BOLTZMANN: f64 = 1.380649e-23;        // J/K https://physics.nist.gov/cgi-bin/cuu/Value?k
pub const CLIGHT_CMS: f64 = 2.99792458e10;      // speed of light, cm/s
pub const R_GAS: f64 = BOLTZMANN * AVOGADRO;    // J/(mol*K) ideal gas constant
pub const E_CHARGE: f64 = 1.6021766208e-19;
pub const DEBYE:f64 = 3.335641e-30;            // C*m = 1e-18/LIGHT_SPEED_SI https://cccbdb.nist.gov/debye.asp
pub const AU2DEBYE:f64 = E_CHARGE * BOHR*1e-10 / DEBYE; // 2.541746


pub const MPI_CHUNK:usize = 134217728; // around 1 GB

lazy_static!{
    pub static ref ATOMIC_RADII: HashMap<&'static str, &'static f64> = {
    /// 获取元素周期表中所有元素的原子半径（单位：Å）
    /// 数据来源：Clementi-Raimondi半径、共价半径和范德华半径的综合
        let mut radii = HashMap::new();
        
        // 第1周期
        radii.insert( "H", &0.53);  // 氢
        radii.insert("He", &0.31);  // 氦

        // 第2周期
        radii.insert("Li", &1.67);  // 锂
        radii.insert("Be", &1.12);  // 铍
        radii.insert( "B", &0.87);  // 硼
        radii.insert( "C", &0.67);  // 碳
        radii.insert( "N", &0.56);  // 氮
        radii.insert( "O", &0.48);  // 氧
        radii.insert( "F", &0.42);  // 氟
        radii.insert("Ne", &0.38);  // 氖

        // 第3周期
        radii.insert("Na", &1.90);  // 钠
        radii.insert("Mg", &1.45);  // 镁
        radii.insert("Al", &1.18);  // 铝
        radii.insert("Si", &1.11);  // 硅
        radii.insert( "P", &0.98);  // 磷
        radii.insert( "S", &0.88);  // 硫
        radii.insert("Cl", &0.79);  // 氯
        radii.insert("Ar", &0.71);  // 氩

        // 第4周期
        radii.insert( "K", &2.43);  // 钾
        radii.insert("Ca", &1.94);  // 钙
        radii.insert("Sc", &1.84);  // 钪
        radii.insert("Ti", &1.76);  // 钛
        radii.insert( "V", &1.71);  // 钒
        radii.insert("Cr", &1.66);  // 铬
        radii.insert("Mn", &1.61);  // 锰
        radii.insert("Fe", &1.56);  // 铁
        radii.insert("Co", &1.52);  // 钴
        radii.insert("Ni", &1.49);  // 镍
        radii.insert("Cu", &1.45);  // 铜
        radii.insert("Zn", &1.42);  // 锌
        radii.insert("Ga", &1.36);  // 镓
        radii.insert("Ge", &1.25);  // 锗
        radii.insert("As", &1.14);  // 砷
        radii.insert("Se", &1.03);  // 硒
        radii.insert("Br", &0.94);  // 溴
        radii.insert("Kr", &0.88);  // 氪

        // 第5周期
        radii.insert("Rb", &2.65);  // 铷
        radii.insert("Sr", &2.19);  // 锶
        radii.insert( "Y", &2.12);  // 钇
        radii.insert("Zr", &2.06);  // 锆
        radii.insert("Nb", &1.98);  // 铌
        radii.insert("Mo", &1.90);  // 钼
        radii.insert("Tc", &1.83);  // 锝
        radii.insert("Ru", &1.78);  // 钌
        radii.insert("Rh", &1.73);  // 铑
        radii.insert("Pd", &1.69);  // 钯
        radii.insert("Ag", &1.65);  // 银
        radii.insert("Cd", &1.61);  // 镉
        radii.insert("In", &1.56);  // 铟
        radii.insert("Sn", &1.45);  // 锡
        radii.insert("Sb", &1.33);  // 锑
        radii.insert("Te", &1.23);  // 碲
        radii.insert( "I", &1.15);  // 碘
        radii.insert("Xe", &1.08);  // 氙

        // 第6周期
        radii.insert("Cs", &2.98);  // 铯
        radii.insert("Ba", &2.53);  // 钡
        radii.insert("La", &2.50);  // 镧
        radii.insert("Ce", &2.48);  // 铈
        radii.insert("Pr", &2.47);  // 镨
        radii.insert("Nd", &2.45);  // 钕
        radii.insert("Pm", &2.43);  // 钷
        radii.insert("Sm", &2.42);  // 钐
        radii.insert("Eu", &2.40);  // 铕
        radii.insert("Gd", &2.38);  // 钆
        radii.insert("Tb", &2.37);  // 铽
        radii.insert("Dy", &2.35);  // 镝
        radii.insert("Ho", &2.33);  // 钬
        radii.insert("Er", &2.32);  // 铒
        radii.insert("Tm", &2.30);  // 铥
        radii.insert("Yb", &2.28);  // 镱
        radii.insert("Lu", &2.27);  // 镥
        radii.insert("Hf", &2.25);  // 铪
        radii.insert("Ta", &2.20);  // 钽
        radii.insert( "W", &2.10);  // 钨
        radii.insert("Re", &2.05);  // 铼
        radii.insert("Os", &2.00);  // 锇
        radii.insert("Ir", &1.97);  // 铱
        radii.insert("Pt", &1.92);  // 铂
        radii.insert("Au", &1.87);  // 金
        radii.insert("Hg", &1.75);  // 汞
        radii.insert("Tl", &1.70);  // 铊
        radii.insert("Pb", &1.54);  // 铅
        radii.insert("Bi", &1.43);  // 铋
        radii.insert("Po", &1.35);  // 钋
        radii.insert("At", &1.27);  // 砹
        radii.insert("Rn", &1.20);  // 氡

        // 第7周期
        radii.insert("Fr", &3.00);  // 钫
        radii.insert("Ra", &2.70);  // 镭
        radii.insert("Ac", &2.60);  // 锕
        radii.insert("Th", &2.50);  // 钍
        radii.insert("Pa", &2.40);  // 镤
        radii.insert( "U",  &2.30); // 铀
        radii.insert("Np", &2.30);  // 镎
        radii.insert("Pu", &2.30);  // 钚
        radii.insert("Am", &2.30);  // 镅
        radii.insert("Cm", &2.30);  // 锔
        radii.insert("Bk", &2.30);  // 锫
        radii.insert("Cf", &2.30);  // 锎
        radii.insert("Es", &2.30);  // 锿
        radii.insert("Fm", &2.30);  // 镄
        radii.insert("Md", &2.30);  // 钔
        radii.insert("No", &2.30);  // 锘
        radii.insert("Lr", &2.30);  // 铹
        radii.insert("Rf", &2.30);  // 卢瑟福
        radii.insert("Db", &2.30);  // 𨧀
        radii.insert("Sg", &2.30);  // 𨭎
        radii.insert("Bh", &2.30);  // 𨨏
        radii.insert("Hs", &2.30);  // 𨭆
        radii.insert("Mt", &2.30);  // 䥑
        radii.insert("Ds", &2.30);  // 鐽
        radii.insert("Rg", &2.30);  // 錀
        radii.insert("Cn", &2.30);  // 鎶
        radii.insert("Nh", &2.30);  // 鉨
        radii.insert("Fl", &2.30);  // 鈇
        radii.insert("Mc", &2.30);  // 鏌
        radii.insert("Lv", &2.30);  // 鉝
        radii.insert("Ts", &2.30);  // 鿬
        radii.insert("Og", &2.30);  // 鿫

        // 添加一些常见的同位素和特殊表示
        radii.insert("D", &0.53);   // 氘
        radii.insert("T", &0.53);   // 氚
        
        radii
    };
}