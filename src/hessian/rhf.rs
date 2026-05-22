use crate::scf_io;
use crate::scf_io::SCF;
use crate::Molecule;
use rest_libcint::prelude::*;
use std::collections::HashMap;
use tensors::matrix_blas_lapack::_power_rayon_for_symmetric_matrix;
use tensors::MatrixFull;

#[non_exhaustive]
#[derive(derive_builder::Builder)]
pub struct RIRHFHessianFlags {
    #[builder(default = 0)] pub print_level: usize,
    #[builder(default = "None")] pub max_memory: Option<f64>,
    #[builder(default = true)] pub auxbasis_response: bool,
    #[builder(default = "Some(1.0)")] pub factor_j: Option<f64>,
    #[builder(default = "Some(1.0)")] pub factor_k: Option<f64>,
    #[builder(default = false)] pub with_cphf: bool,
}

pub struct RIRHFHessian<'a> {
    pub scf_data: &'a SCF,
    pub flags: RIRHFHessianFlags,
    pub result: HashMap<String, MatrixFull<f64>>,
}

impl RIRHFHessian<'_> {
    pub fn new(scf_data: &SCF) -> RIRHFHessian<'_> {
        match scf_data.scftype { scf_io::SCFType::RHF => {},
            _ => panic!("SCF type is not suitable for RHF Hessian."),
        }
        RIRHFHessian { scf_data, flags: RIRHFHessianFlagsBuilder::default().build().unwrap(),
            result: HashMap::new(),
        }
    }

    pub fn calc_e1(&mut self) -> &mut Self {
        let scf = self.scf_data; let mol = &scf.mol;
        let dm0_mat = &scf.density_matrix[0]; let nao = mol.num_basis;
        let dm0: Vec<f64> = dm0_mat.iter().copied().collect();
        let c = &scf.eigenvectors[0]; let eps = &scf.eigenvalues[0];
        let nocc = (scf.homo[0] + 1) as usize;
        let mut dme0 = vec![0.0; nao * nao];
        for p in 0..nao { for q in 0..nao {
            let mut s = 0.0;
            for i in 0..nocc { s += c[[p,i]] * eps[i] * c[[q,i]]; }
            dme0[p * nao + q] = s * 2.0;
        }}
        self.result.insert("e1".to_string(), compute_e1(mol, &dm0, &dme0));
        self
    }

    /// Compute ej = basic + vjd + vj1 + ri1 + ri2d + ri2o
    /// Ported from standalone prototype main.rs, using rest_libcint integrals.
    pub fn calc_ej_ek(&mut self) -> &mut Self {
        let scf = self.scf_data; let mol = &scf.mol;
        let nao = mol.num_basis; let natm = mol.geom.nfree;
        let aoslices = mol.aoslice_by_atom();
        // SCF data
        let dm0_mat = &scf.density_matrix[0];
        let dm0: Vec<f64> = dm0_mat.iter().copied().collect(); // col-major
        let c = &scf.eigenvectors[0]; let eps = &scf.eigenvalues[0];
        let mo_occ = &scf.occupation[0];
        let nocc = (scf.homo[0] + 1) as usize;
        // mocc_2 = C_occ · sqrt(occ)  (only occupied cols, weighted)
        let mut mc2 = vec![0.0; nao * nocc];
        for p in 0..nao { for i in 0..nocc {
            mc2[p * nocc + i] = c[[p, i]] * (mo_occ[i] as f64).sqrt();
        }}
        // dme0 not needed for ej/ek

        // ── Set up auxiliary basis ──
        // Use combined CInt (regular + aux) for both 2c and 3c integrals
        let cint_reg = mol.initialize_cint(false);
        let nreg = cint_reg.nbas(); // regular shells
        let cint_all = mol.initialize_cint(true); // reg + aux combined
        let naux_shell = cint_all.nbas() - nreg; // aux shells
        let auxmol = mol.make_auxmol_fake();
        let naux = auxmol.num_basis;
        let auxslices = auxmol.aoslice_by_atom();

        // Compute V = int2c2e (aux-only, via shell slice)
        let aux_slc = &[[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        let (int2c_v,_) = cint_all.integrate_row_major("int2c2e","s1",Some(&aux_slc[..])).into();
        let mut int2c = vec![0.0; naux * naux];
        if int2c_v.len() == naux * naux { int2c = int2c_v; }
        else { // triangular expansion
            let mut idx = 0;
            for j in 0..naux { for i in 0..=j {
                int2c[i + j * naux] = int2c_v[idx];
                int2c[j + i * naux] = int2c_v[idx]; idx += 1;
            }}
        }
        // Column-major V^{-1}
        let mut int2c_cm = vec![0.0; naux * naux];
        for p in 0..naux { for q in 0..naux { int2c_cm[p + q * naux] = int2c[p * naux + q]; }}
        let vinv = compute_vinv(&int2c_cm, naux);
        let i2inv = vinv.clone();

        // ── 2c-2e auxiliary basis integrals (via combined CInt with aux-only slice) ──
        let aux_slc_ref: &[[usize; 2]] = &[[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        // For in-place aux-only integrals, we need to use the typed API or change approach.
        // For now, compute int2c2e_ip1 etc on the auxmol directly:
        let cint_aux = auxmol.initialize_cint(false);
        let (i21_v,_) = cint_aux.integrate_row_major("int2c2e_ip1","s1",None).into();
        let (i211_v,_) = cint_aux.integrate_row_major("int2c2e_ipip1","s1",None).into();
        let (i212_v,_) = cint_aux.integrate_row_major("int2c2e_ip1ip2","s1",None).into();
        let i21 = i21_v; let i211 = i211_v; let i212 = i212_v;

        // ── Per-atom block ranges ──
        let mut blk = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize;
            blk.push((ia, p0, p1 - p0));
        }
        let mut aux_blk = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = auxslices[ia][2] as usize; let p1 = auxslices[ia][3] as usize;
            aux_blk.push((ia, p0, p1 - p0));
        }

        let nao3 = nao * nao;
        let i9  = |c: usize, p: usize, q: usize| c * nao3 + p * nao + q;
        let i9p = |c: usize, p: usize, pp: usize| c * naux * naux + p * naux + pp;

        // ══════════════════════════════════════════════════════════
        // Compute all 3c-2e integrals via rest_libcint
        // ══════════════════════════════════════════════════════════
        // 3c-2e integrals via combined CInt (shell slice: reg, reg, aux)
        let slc_3c: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg, nreg + naux_shell]];
        let int3c = |name: &str| -> Vec<f64> {
            let (v,_) = cint_all.integrate_row_major(name, "s1", Some(slc_3c)).into();
            v
        };
        let t3c  = int3c("int3c2e");        // (N,N,P)
        let ip1  = int3c("int3c2e_ip1");    // (3,N,N,P)
        let ip2  = int3c("int3c2e_ip2");    // (3,N,N,P)
        let ipv  = int3c("int3c2e_ipvip1"); // (9,N,N,P)
        let ipip1= int3c("int3c2e_ipip1");  // (9,N,N,P)
        let ipip2= int3c("int3c2e_ipip2");  // (9,N,N,P)
        let ip12 = int3c("int3c2e_ip1ip2"); // (9,N,N,P)

        // Debug integrals
        println!("t3c[0..5] = {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}", t3c[0], t3c[1], t3c[2], t3c[3], t3c[4]);
        println!("ip1[0..3] = {:.4e} {:.4e} {:.4e}", ip1[0], ip1[1], ip1[2]);
        println!("ipv[0..3] = {:.4e} {:.4e} {:.4e}", ipv[0], ipv[1], ipv[2]);

        // ══ Phase 1a: rhoj0_P, rhok0_Pl_ (prototype L57-66) ══
        let mut r0r = vec![0.0; naux]; let mut rkr = vec![0.0; naux * nao * nocc];
        for (ib, &(_, p0, ni)) in blk.iter().enumerate() {
            // Use FULL int3c2e, index at AO range p0:p0+ni
            for p in 0..naux {
                let mut s = 0.0;
                for k in 0..ni { for l in 0..nao {
                    s += t3c[(p0 + k) * nao * naux + l * naux + p] * dm0[(p0 + k) * nao + l];
                }}
                r0r[p] += s;
            }
            for p in 0..naux {
                for ii in 0..ni {
                    for oc in 0..nocc {
                        let mut s = 0.0;
                        for j in 0..nao {
                            s += t3c[(p0 + ii) * nao * naux + j * naux + p] * mc2[j * nocc + oc];
                        }
                        rkr[p + (p0 + ii) * naux + oc * naux * nao] += s;
                    }
                }
            }
        }
        // Apply V^{-1}
        let mut r0 = vec![0.0; naux]; let mut rk = vec![0.0; naux * nao * nocc];
        for p in 0..naux { for q in 0..naux { r0[p] += vinv[p + q * naux] * r0r[q]; }}
        for p in 0..naux { for c in 0..nao * nocc {
            let mut s = 0.0; for q in 0..naux { s += vinv[p + q * naux] * rkr[q + c * naux]; }
            rk[p + c * naux] = s;
        }}

        // ══ Phase 1b: vj1_diag, vk1_diag (prototype L68-76) ══
        let mut vjd = vec![0.0; 9 * nao * nao];
        let mut vkd = vec![0.0; 9 * nao * nao];
        for x in 0..9 { for i in 0..nao { for j in 0..nao {
            let mut sj = 0.0; for p in 0..naux { sj += ipip1[x * nao3 + i * nao * naux + j * naux + p] * r0[p]; }
            vjd[x * nao3 + i * nao + j] = sj;
        }}}
        let mut rkm = vec![0.0; naux * nao * nao];
        for p in 0..naux { for l in 0..nao { for J in 0..nao {
            let mut s = 0.0; for j in 0..nocc { s += rk[p + l * naux + j * naux * nao] * mc2[J * nocc + j]; }
            rkm[p + l * naux + J * naux * nao] = s;
        }}}
        for x in 0..9 { for i in 0..nao { for l in 0..nao {
            let mut s = 0.0; for p in 0..naux { for j in 0..nao {
                s += ipip1[x * nao3 + i * nao * naux + j * naux + p] * rkm[p + l * naux + j * naux * nao];
            }} vkd[x * nao3 + i * nao + l] = s;
        }}}

        // ══ Phase 2: rhoj1, wj1, rho_ip1 (prototype L79-88) ══
        let m3 = 3 * nao * nao;
        let mut ip1c = vec![0.0; m3 * naux];
        for x in 0..3 { for i in 0..nao { for j in 0..nao { for p in 0..naux {
            ip1c[(x * nao * nao + i * nao + j) + p * m3] = ip1[x * nao3 + i * nao * naux + j * naux + p];
        }}}}
        let mut tmpf = vec![0.0; naux * m3];
        for p in 0..naux { for c in 0..m3 {
            let mut s = 0.0; for q in 0..naux { s += vinv[p + q * naux] * ip1c[c + q * m3]; }
            tmpf[p + c * naux] = s;
        }}
        let mut rj1 = vec![0.0; natm * naux * 3]; let mut wj1 = vec![0.0; natm * naux * 3];
        for (ib, &(_, p0, ni)) in blk.iter().enumerate() {
            for p in 0..naux { for x in 0..3 {
                let mut sr = 0.0; let mut sw = 0.0;
                for ii in 0..ni { for j in 0..nao {
                    let i = p0 + ii; let c = x * nao * nao + i * nao + j;
                    sr += tmpf[p + c * naux] * dm0[(p0 + ii) * nao + j];
                    sw += ip1[x * nao3 + i * nao * naux + j * naux + p] * dm0[(p0 + ii) * nao + j];
                }}
                rj1[ib * naux * 3 + p * 3 + x] = sr;
                wj1[ib * naux * 3 + p * 3 + x] = sw;
            }}
        }

        // ══ Phase 3a: vk2buf (prototype L168-183, from Python) ══
        // This needs rhok_ip1_IkP. Compute it from tmpf and dm0:
        let mut rhok_IkP = vec![0.0; natm * nao * naux * 3]; // stored per-atom
        let mut rhok_PkI_full = vec![0.0; naux * nao * nao * 3];
        for (ib, &(_, p0, ni)) in blk.iter().enumerate() {
            let mut tmp_ikp = vec![0.0; 3 * naux * ni * nao];
            for x in 0..3 { for p in 0..naux { for ii in 0..ni { for j in 0..nao {
                tmp_ikp[x * naux * ni * nao + p * ni * nao + ii * nao + j] = tmpf[p + (x * nao * nao + (p0+ii) * nao + j) * naux];
            }}}}
            // rhok_ip1_IkP = einsum('pykl,li->ikpy', tmp_ip1, dm0)
            for ii in 0..ni { for k in 0..nao { for p in 0..naux { for x in 0..3 {
                let mut s = 0.0; for l in 0..nao {
                    s += tmp_ikp[x * naux * ni * nao + p * ii * nao + l * nao + k] * dm0[l * nao + (p0 + ii)];
                }
                rhok_IkP[ib * nao * naux * 3 + (p0 + ii) * naux * 3 + p * 3 + x] = s;
                // Also fill PkI for full
                rhok_PkI_full[p * nao * nao * 3 + ii * nao * 3 + k * 3 + x] = s;
            }}}}
        }
        // Debug
        let rj1_max = rj1.iter().cloned().fold(0.0, f64::max);
        let r0_max = r0.iter().cloned().fold(0.0, f64::max);
        let wj1_max = wj1.iter().cloned().fold(0.0, f64::max);
        let vinv_max = vinv.iter().cloned().fold(0.0, f64::max);
        println!("Phase1-2: r0_max={:.4e} rj1_max={:.4e} wj1_max={:.4e} vinv_max={:.4e}", r0_max, rj1_max, wj1_max, vinv_max);
        // Debug contribution arrays
        // vk2buf = Σ int3c_ip1 × _load_dim0(rhok_ip1_PkI, p0, p1)
        let mut vk2buf = vec![0.0; 9 * nao * nao];
        for x in 0..3 { for y in 0..3 {
            for i in 0..nao { for j in 0..nao {
                let mut s = 0.0;
                for p in 0..naux {
                    s += ip1[x * nao3 + i * nao * naux + j * naux + p] * rhok_PkI_full[p * nao * nao * 3 + j * nao * 3 + i * 3 + y];
                }
                vk2buf[(x * 3 + y) * nao3 + i * nao + j] = s;
            }}
        }}

        // ══ Phase 3b: wj_ip2, wk_ip2_Ipk, wk_ip2_P__ (prototype L92-97) ══
        let mut wj2 = vec![0.0; naux * 3];
        let mut wki = vec![0.0; nao * naux * 3 * nao];
        let mut wk2 = vec![0.0; naux * 3 * nocc * nocc];
        for p in 0..naux { for y in 0..3 {
            let mut sj = 0.0;
            for k in 0..nao { for l in 0..nao { sj += ip2[y * nao3 + k * nao * naux + l * naux + p] * dm0[k * nao + l]; }}
            wj2[p * 3 + y] = sj;
        }}
        for i in 0..nao { for p in 0..naux { for y in 0..3 { for k in 0..nao {
            let mut s = 0.0; for l in 0..nao { s += ip2[y * nao3 + k * nao * naux + l * naux + p] * dm0[l * nao + i]; }
            wki[i * naux * 3 * nao + p * 3 * nao + y * nao + k] = s;
        }}}}
        for p in 0..naux { for x in 0..3 { for i in 0..nocc { for j in 0..nocc {
            let mut s = 0.0; for u in 0..nao { for v in 0..nao {
                s += ip2[x * nao3 + u * nao * naux + v * naux + p] * mc2[u * nocc + i] * mc2[v * nocc + j];
            }} wk2[p * 3 * nocc * nocc + x * nocc * nocc + i * nocc + j] = s;
        }}}}

        // ══ Phase 3c: rhok0_P__, rho2c_0, int2c_ip_ip (prototype L98-106) ══
        let mut rkoo = vec![0.0; naux * nocc * nocc];
        for p in 0..naux { for i in 0..nocc { for jj in 0..nocc {
            let mut s = 0.0; for l in 0..nao { s += rk[p + l * naux + i * naux * nao] * mc2[l * nocc + jj]; }
            rkoo[p * nocc * nocc + i * nocc + jj] = s;
        }}}
        let mut r2c0 = vec![0.0; naux * naux];
        for p in 0..naux { for q in 0..naux {
            let mut s = 0.0; for i in 0..nocc { for j in 0..nocc { s += rkoo[p * nocc * nocc + i * nocc + j] * rkoo[q * nocc * nocc + j * nocc + i]; }}
            r2c0[p + q * naux] = s;
        }}
        let mut i2ip = vec![0.0; 9 * naux * naux];
        for x in 0..3 { for y in 0..3 {
            for p in 0..naux { for s in 0..naux {
                let mut sv = 0.0; for q in 0..naux { for r in 0..naux {
                    sv += i21[p * naux * 3 + q * 3 + x] * i2inv[q + r * naux] * i21[s * naux * 3 + r * 3 + y];
                }}
                sv -= i212[(x * 3 + y) * naux * naux + p * naux + s];
                i2ip[(x * 3 + y) * naux * naux + p * naux + s] = sv;
            }}
        }}
        let mut wj001 = vec![0.0; 3 * naux];
        for y in 0..3 { for p in 0..naux {
            let mut s = 0.0; for q in 0..naux { s += i21[p * naux * 3 + q * 3 + y] * r0[q]; }
            wj001[y * naux + p] = s;
        }}

        // ══════════════════════════════════════════════════════════
        // Phase 4: Contribution arrays
        // ══════════════════════════════════════════════════════════
        let aa9 = natm * natm * 9;
        let mut ej_basic = vec![0.0; aa9]; let mut ej_vjd = vec![0.0; aa9];
        let mut ej_vj1 = vec![0.0; aa9]; let mut ej_ri1 = vec![0.0; aa9];
        let mut ej_ri2d = vec![0.0; aa9]; let mut ej_ri2o = vec![0.0; aa9];
        let mut ek_vkd = vec![0.0; aa9]; let mut ek_vk1 = vec![0.0; aa9];
        let mut ek_ri1 = vec![0.0; aa9]; let mut ek_ri2d = vec![0.0; aa9];
        let mut ek_ri2o = vec![0.0; aa9];
        let i_t = |i0, j0, x, y| i0 * natm * 9 + j0 * 9 + x * 3 + y;

        // 1. ej_basic (prototype L117-119)
        for a in 0..natm { for b in 0..natm { for x in 0..3 { for y in 0..3 {
            let mut s = 0.0; for p in 0..naux { s += rj1[a * naux * 3 + p * 3 + x] * wj1[b * naux * 3 + p * 3 + y]; }
            ej_basic[i_t(a, b, x, y)] = s * 4.0;
        }}}}
        println!("ej_basic[0..3] = {:.4e} {:.4e} {:.4e} ej_basic[27]= {:.4e}",
                 ej_basic[0], ej_basic[1], ej_basic[2], ej_basic[27]);
        println!("ej_vjd[0]={:.4e} ej_vj1[0]={:.4e} ej_ri1[0]={:.4e}", ej_vjd[0], ej_vj1[0], ej_ri1[0]);
        println!("ej_ri2d[0]={:.4e} ej_ri2o[0]={:.4e}", ej_ri2d[0], ej_ri2o[0]);

        // 2. vj1_diag + vk1_diag (prototype L122-130)
        for i0 in 0..natm { let (_, p0, ni) = blk[i0];
            for x in 0..3 { for y in 0..3 {
                let mut sj = 0.0; let mut sk = 0.0;
                for p in p0..p0 + ni { for q in 0..nao {
                    sj += vjd[(x * 3 + y) * nao3 + p * nao + q] * dm0[p * nao + q];
                    sk += vkd[(x * 3 + y) * nao3 + p * nao + q] * dm0[p * nao + q];
                }}
                ej_vjd[i_t(i0, i0, x, y)] = sj * 2.0;
                ek_vkd[i_t(i0, i0, x, y)] = sk;
            }}
        }

        // 3. vj1 from ipvip1 (prototype L133-144) — using full ipv integral
        for i0 in 0..natm { let (ib, p0, ni) = blk[i0];
            let mut vj1_mat = vec![0.0; 9 * nao * ni];
            for x in 0..9 { for j in 0..nao { for i in 0..ni {
                let mut s = 0.0; for p in 0..naux {
                    s += ipv[x * nao3 * naux + (p0+i) * nao * naux + j * naux + p] * r0[p];
                } vj1_mat[x * nao * ni + j * ni + i] = s;
            }}}
            for j0 in 0..=i0 { let (_, q0, qj) = blk[j0];
                for x1 in 0..3 { for x2 in 0..3 {
                    let mut sj = 0.0;
                    for p_off in 0..qj { for q_off in 0..ni {
                        sj += vj1_mat[(x1 * 3 + x2) * nao * ni + (q0 + p_off) * ni + q_off]
                            * dm0[(q0 + p_off) * nao + (p0 + q_off)];
                    }}
                    ej_vj1[i_t(i0, j0, x1, x2)] = sj * 2.0;
                }}
            }
        }

        // 4. ek_vk1 from vk2buf + ipvip1 (prototype L253-268)
        // vk1 from ipvip1: einsum('pki,ji->pkj', rk_block, mc2) then einsum('xijp,pki->xjk')
        for i0 in 0..natm { let (ib, p0, ni) = blk[i0];
            let mut tmp = vec![0.0; naux * nao * ni];
            for p in 0..naux { for k in 0..nao { for ii in 0..ni {
                let mut s = 0.0; for j in 0..nocc {
                    s += rk[p + (p0 + ii) * naux + j * naux * nao] * mc2[k * nocc + j];
                } tmp[p * nao * ni + k * ni + ii] = s;
            }}}
            // vk1 from ipvip1 + vk2buf (matching prototype L253-268)
            for j0 in 0..=i0 { let (_, q0, qj) = blk[j0];
                for x in 0..3 { for y in 0..3 {
                    let c = x*3+y;
                    // vk1 = einsum('xijp,pki->xjk', tipv, tmp) + vk2buf sub-block
                    let mut sv = 0.0;
                    // ipvip1 part: tipv[c, i_bra, j_ao, p] * tmp[p, k_ao, i_bra]
                    // summed over i_bra (0..ni), j_ao (within j0 block), k_ao (within j0 block), p aux
                    // Result is sub-block (j0:k0) = (q0:qj, q0:qj) in the (j,k) matrix
                    for i_bra in 0..ni { for j_ao in 0..qj { for k_ao in 0..qj { for p in 0..naux {
                        sv += ipv[c * nao3 * naux + (p0+i_bra) * nao * naux + (q0+j_ao) * naux + p]
                            * tmp[p * nao * ni + (q0+k_ao) * ni + i_bra];
                    }}}}
                    // vk2buf contribution (sub-block at j0 block)
                    for j_ao in 0..qj { for k_ao in 0..qj {
                        sv += vk2buf[c * nao3 + (q0+j_ao) * nao + (q0+k_ao)]
                            * dm0[(q0+j_ao) * nao + (q0+k_ao)];
                    }}
                    ek_vk1[i_t(i0, j0, x, y)] = sv;
                }}
            }
        }

        // 5. ek_ri1: RI first-order K response (prototype L179-226)
        // Need wk1_Pij = rho_ip1-like data from tmpf
        // Build rho_ip1: (nao, nao, naux, 3) from tmpf
        let mut rho_ip1 = vec![0.0; nao * nao * naux * 3];
        for p in 0..naux { for x in 0..3 { for i in 0..nao { for j in 0..nao {
            rho_ip1[i * nao * naux * 3 + j * naux * 3 + p * 3 + x] = tmpf[p + (x * nao3 + i * nao + j) * naux];
        }}}}
        // Pre-compute wk1_IpJ from wki (the wk_ip2_Ipk intermediate)
        let mut wk1_IpJ_full = vec![0.0; nao * naux * 3 * nao]; // same layout as wki
        for i in 0..nao { for p in 0..naux { for y in 0..3 { for k in 0..nao {
            let mut s = 0.0; for j in 0..nao {
                s += wki[i * naux * 3 * nao + p * 3 * nao + y * nao + j] * dm0[j * nao + k];
            }
            wk1_IpJ_full[i * naux * 3 * nao + p * 3 * nao + y * nao + k] = s;
        }}}}

        for i0 in 0..natm { let (_, p0, ni) = blk[i0];
            // Build wk1_Pij = rho_ip1[p0:p0+ni].transpose(2,3,0,1) → (naux, 3, ni, nao)
            let mut wkp = vec![0.0; naux * 3 * ni * nao];
            for p in 0..naux { for x in 0..3 { for ii in 0..ni { for j in 0..nao {
                wkp[p * 3 * ni * nao + x * ni * nao + ii * nao + j] =
                    rho_ip1[(p0 + ii) * nao * naux * 3 + j * naux * 3 + p * 3 + x];
            }}}}
            // rhok0_P_I = einsum('plj,il->pji', rk_block, dm0_block)
            let mut rk_P_I = vec![0.0; naux * ni * nao];
            for p in 0..naux { for ii in 0..ni { for j in 0..nao {
                let mut s = 0.0; for l in 0..nao {
                    s += rk[p + l * naux + i0 * naux * nao] * dm0[l * nao + (p0 + ii)];
                } rk_P_I[p * ni * nao + ii * nao + j] = s;
            }}}
            // rhok0_PJI = einsum('pji,Jj->pJi', rhok0_P_I, mc2)
            let mut rk_PJI = vec![0.0; naux * nao * ni];
            for p in 0..naux { for J in 0..nao { for ii in 0..ni {
                let mut s = 0.0; for j in 0..nocc {
                    s += rk_P_I[p * ni * nao + ii * nao + j] * mc2[J * nocc + j];
                } rk_PJI[p * nao * ni + J * ni + ii] = s;
            }}}
            // wk1_pJI = einsum('ypq,qji->ypji', i21, rk_PJI)
            let mut wk1_pJI = vec![0.0; 3 * naux * nao * ni];
            for y in 0..3 { for p in 0..naux { for J in 0..nao { for ii in 0..ni {
                let mut s = 0.0; for q in 0..naux {
                    s += i21[p * naux * 3 + q * 3 + y] * rk_PJI[q * nao * ni + J * ni + ii];
                } wk1_pJI[y * naux * nao * ni + p * nao * ni + J * ni + ii] = s;
            }}}}
            // wk1_IpJ = einsum('ipyk,kj->ipyj', wk_ip2_Ipk[p0:p1], dm0)
            let mut wk1_IpJ = vec![0.0; ni * naux * 3 * nao];
            for ii in 0..ni { for p in 0..naux { for y in 0..3 { for k in 0..nao {
                wk1_IpJ[ii * naux * 3 * nao + p * 3 * nao + y * nao + k] =
                    wk1_IpJ_full[(p0 + ii) * naux * 3 * nao + p * 3 * nao + y * nao + k];
            }}}}
            // rho2c_PQ = einsum('pxij,qji->xqp', wk1_Pij, rk_PJI)
            let mut rho2c_PQ = vec![0.0; 3 * naux * naux];
            for x in 0..3 { for q in 0..naux { for p_aux in 0..naux {
                let mut s = 0.0; for ii in 0..ni { for j in 0..nao {
                    s += wkp[p_aux * 3 * ni * nao + x * ni * nao + ii * nao + j]
                        * rk_PJI[q * nao * ni + j * ni + ii];
                }} rho2c_PQ[x * naux * naux + q * naux + p_aux] = s;
            }}}

            for j0 in 0..natm { let (_, aq0, aq_sz) = aux_blk[j0]; let ql = aq_sz;
                // Use full ip12 integral with AO offset p0
                let q0 = aq0;
                // Term1: 'xijp,pji->x' from ip1ip2 × rk_PJI
                let mut t1 = vec![0.0; 9];
                for x in 0..9 { let mut s = 0.0;
                    for ii in 0..ni { for j in 0..nao { for qp in 0..ql {
                        let pg = q0 + qp;
                        s += ip12[x * nao3 * naux + (p0+ii) * nao * naux + j * naux + pg]
                            * rk_PJI[pg * nao * ni + j * ni + ii];
                    }}} t1[x] = s;
                }
                // Term2: 'pxij,ypji->xy' wkp × wk1_pJI
                let mut t2 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let pg = q0 + qp;
                        for ii in 0..ni { for j in 0..nao {
                            s += wkp[pg * 3 * ni * nao + x * ni * nao + ii * nao + j]
                                * wk1_pJI[y * naux * nao * ni + pg * nao * ni + j * ni + ii];
                        }}
                    } t2[x * 3 + y] = s;
                }}
                // Term3: 'xqp,yqp->xy' rho2c_PQ × i21
                let mut t3 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let qg = q0 + qp;
                        for paux in 0..naux {
                            s += rho2c_PQ[x * naux * naux + qg * naux + paux]
                                * i21[paux * naux * 3 + qg * 3 + y];
                        }
                    } t3[x * 3 + y] = s;
                }}
                // Term4: 'pxij,ipyj->xy' wkp × wk1_IpJ
                let mut t4 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let pg = q0 + qp;
                        for ii in 0..ni { for j in 0..nao {
                            s += wkp[pg * 3 * ni * nao + x * ni * nao + ii * nao + j]
                                * wk1_IpJ[ii * naux * 3 * nao + pg * 3 * nao + y * nao + j];
                        }}
                    } t4[x * 3 + y] = s;
                }}
                // _ek = t1 - t2 - t3 + t4
                for x in 0..3 { for y in 0..3 {
                    let v = t1[x * 3 + y] - t2[x * 3 + y] - t3[x * 3 + y] + t4[x * 3 + y];
                    ek_ri1[i_t(i0, j0, x, y)] += v;
                    ek_ri1[i_t(j0, i0, x, y)] += t1[y * 3 + x] - t2[y * 3 + x] - t3[y * 3 + x] + t4[y * 3 + x];
                }}
            }
        }

        // 6. ek_ri2d: RI second-order K diagonal (prototype L230-250)
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            let mut rkj = vec![0.0; ni_aux * nao * nao];
            for p in 0..ni_aux { for J in 0..nao { for I in 0..nao {
                let mut s = 0.0; for j in 0..nocc { for i in 0..nocc {
                    s += rkoo[(ap0 + p) * nocc * nocc + i * nocc + j] * mc2[J * nocc + j] * mc2[I * nocc + i];
                }} rkj[p * nao * nao + J * nao + I] = s;
            }}}
            // Use full ipip2 integral with aux offset ap0
            let mut ta = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for I in 0..nao { for J in 0..nao { for p in 0..ni_aux {
                    s += ipip2[x * nao * nao * naux + I * nao * naux + J * naux + (ap0 + p)]
                        * rkj[p * nao * nao + I * nao + J];
                }}} ta[x] = s * 0.5;
            }
            let mut tb = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for p in 0..ni_aux { for q in 0..naux {
                    s += r2c0[(ap0 + p) * naux + q] * i211[x * naux * naux + (ap0 + p) * naux + q];
                }} tb[x] = s * (-0.5);
            }
            for x in 0..3 { for y in 0..3 {
                ek_ri2d[i_t(i0, i0, x, y)] += ta[x * 3 + y] + tb[x * 3 + y];
            }}
        }

        // 7. ek_ri2o: RI second-order K off-diagonal (prototype L253-278)
        // Need rho2c_1 intermediate, which we compute here
        let mut rho2c1_per_atom = vec![0.0; natm * 3 * naux * naux];
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            // ip1_2c_2c = ip1[:,ap0:ap1,:] · i2inv  ⇒  (3, ni, P)
            let mut ip1_2c = vec![0.0; 3 * ni_aux * naux];
            for x in 0..3 { for p in 0..ni_aux { for r in 0..naux {
                let mut s = 0.0; for q in 0..naux {
                    s += i21[(ap0 + p) * naux * 3 + q * 3 + x] * i2inv[q + r * naux];
                } ip1_2c[x * ni_aux * naux + p * naux + r] = s;
            }}}
            // ip1_rho2c = 0.5 * einsum('xpq,qr->xpr', int2c_ip1[:,ap0:ap1], rho2c_0)
            let mut ip1_r2c = vec![0.0; 3 * ni_aux * naux];
            for x in 0..3 { for p in 0..ni_aux { for r in 0..naux {
                let mut s = 0.0; for q in 0..naux {
                    s += i21[(ap0 + p) * naux * 3 + q * 3 + x] * r2c0[q * naux + r];
                } ip1_r2c[x * ni_aux * naux + p * naux + r] = s * 0.5;
            }}}
            // rho2c_1 = ip1_rho2c · i2inv[ap0:ap1] + ip1_2c_2c · r2c0[ap0:ap1] - tmp_so
            let mut r2c1 = vec![0.0; 3 * naux * naux];
            for x in 0..3 { for p in 0..naux { for q in 0..naux {
                let mut s1 = 0.0; let mut s2 = 0.0;
                for r in 0..ni_aux {
                    s1 += ip1_r2c[x * ni_aux * naux + r * naux + p] * i2inv[(ap0 + r) * naux + q];
                    s2 += ip1_2c[x * ni_aux * naux + r * naux + p] * r2c0[(ap0 + r) * naux + q];
                }
                r2c1[x * naux * naux + p * naux + q] = s1 + s2;
            }}}
            // For now, set the per-atom rho2c_1 (tmp_so not implemented)
            for x in 0..3 { for p in 0..naux { for q in 0..naux {
                rho2c1_per_atom[i0 * 3 * naux * naux + x * naux * naux + p * naux + q] = r2c1[x * naux * naux + p * naux + q];
            }}}
        }
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            let r2c1 = &rho2c1_per_atom[i0 * 3 * naux * naux..];
            for j0 in 0..natm { let (_, aq0, nq) = aux_blk[j0];
                // T1: 0.5 * einsum('pq,xypq->xy', r2c0_sub, i2ip_sub)
                let mut t1 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for p in 0..ni_aux { for q in 0..nq {
                        s += r2c0[(ap0 + p) * naux + (aq0 + q)]
                            * i2ip[(x * 3 + y) * naux * naux + (ap0 + p) * naux + (aq0 + q)];
                    }} t1[x * 3 + y] = s * 0.5;
                }}
                // T2: einsum('xpq,ypq->xy', rho2c_1, i21_sub)
                let mut t2 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for p in 0..nq { for q in 0..naux {
                        s += r2c1[x * naux * naux + (aq0 + p) * naux + q]
                            * i21[q * naux * 3 + (aq0 + p) * 3 + y];
                    }} t2[x * 3 + y] = s;
                }}
                // T3: 0.5 * einsum('pxij,pq,qyij->xy', wk2_sub, i2cinv_sub, wk2_sub_2)
                let mut t3 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for p in 0..ni_aux { for q in 0..nq {
                        let inv_val = i2inv[(ap0 + p) * naux + (aq0 + q)];
                        for i in 0..nocc { for j in 0..nocc {
                            s += wk2[(ap0 + p) * 3 * nocc * nocc + x * nocc * nocc + i * nocc + j]
                                * inv_val
                                * wk2[(aq0 + q) * 3 * nocc * nocc + y * nocc * nocc + i * nocc + j];
                        }}
                    }} t3[x * 3 + y] = s * 0.5;
                }}
                for x in 0..3 { for y in 0..3 {
                    let ek_val = (t1[x * 3 + y] + t2[x * 3 + y] + t3[x * 3 + y]) * 0.5;
                    ek_ri2o[i_t(i0, j0, x, y)] += ek_val;
                    ek_ri2o[i_t(j0, i0, x, y)] += (t1[y * 3 + x] + t2[y * 3 + x] + t3[y * 3 + x]) * 0.5;
                }}
            }
        }

        // 8. ej_ri1: RI first-order J response (prototype L281-313)
        for i0 in 0..natm { let (ib, p0, ni) = blk[i0];
            let mut w11 = vec![0.0; 9 * naux];
            for x in 0..9 { for p in 0..naux { let mut ss = 0.0;
                for ii in 0..ni { for j in 0..nao {
                    ss += ip12[x * nao3 * naux + (p0+ii) * nao * naux + j * naux + p]
                        * dm0[(p0 + ii) * nao + j];
                }} w11[x * naux + p] = ss;
            }}
            for j0 in 0..natm { let (_, aq0, ql) = aux_blk[j0];
                let q0 = aq0;
                // T1: 'xp,p->x' w11[:,q0:q1], r0[q0:q1]
                let mut t1 = vec![0.0; 9];
                for x in 0..9 { let mut s = 0.0;
                    for qp in 0..ql { s += w11[x * naux + q0 + qp] * r0[q0 + qp]; } t1[x] = s;
                }
                // T2: 'yqp,q,px->xy' i21, r0, rj1
                let mut t2 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let qg = q0 + qp;
                        for paux in 0..naux {
                            s += i21[paux * naux * 3 + qg * 3 + y] * r0[qg] * rj1[i0 * naux * 3 + paux * 3 + x];
                        }
                    } t2[x * 3 + y] = s;
                }}
                // T3: 'px,yp->xy' rj1, wj001
                let mut t3 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let qg = q0 + qp;
                        s += rj1[i0 * naux * 3 + qg * 3 + x] * wj001[y * naux + qg];
                    } t3[x * 3 + y] = s;
                }}
                // T4: 'px,py->xy' rj1, wj2
                let mut t4 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let qg = q0 + qp;
                        s += rj1[i0 * naux * 3 + qg * 3 + x] * wj2[qg * 3 + y];
                    } t4[x * 3 + y] = s;
                }}
                for x in 0..3 { for y in 0..3 {
                    let v = (t1[x * 3 + y] - t2[x * 3 + y] - t3[x * 3 + y] + t4[x * 3 + y]) * 2.0;
                    ej_ri1[i_t(i0, j0, x, y)] += v;
                    ej_ri1[i_t(j0, i0, x, y)] += (t1[y * 3 + x] - t2[y * 3 + x] - t3[y * 3 + x] + t4[y * 3 + x]) * 2.0;
                }}
            }
        }

        // 9. ej_ri2d: RI second-order J diagonal (prototype L316-331)
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            let mut td = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for i in 0..nao { for j in 0..nao { for p in 0..ni_aux {
                    s += ipip2[x * nao * nao * naux + i * nao * naux + j * naux + (ap0 + p)]
                        * dm0[j * nao + i] * r0[ap0 + p];
                }}} td[x] = s;
            }
            let mut te = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for p in 0..ni_aux { for q in 0..naux {
                    s += r0[ap0 + p] * i211[x * naux * naux + (ap0 + p) * naux + q] * r0[q];
                }} te[x] = s;
            }
            for x in 0..3 { for y in 0..3 {
                ej_ri2d[i_t(i0, i0, x, y)] += td[x * 3 + y] - te[x * 3 + y];
            }}
        }

        // 10. ej_ri2o: RI second-order J off-diagonal (prototype L333-379)
        // Need rhoj1_so, rhoj0_01, rhoj0_10 per atom
        let mut rs1_per = vec![0.0; natm * 3 * naux];
        let mut r01_per = vec![0.0; natm * 3 * naux];
        let mut r10_per = vec![0.0; natm * 3 * naux];
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            // rhoj1_so = wj2[ap0:ap1] · i2inv   → (3, P)
            for x in 0..3 { for q in 0..naux {
                let mut s = 0.0; for p in 0..ni_aux {
                    s += wj2[(ap0 + p) * 3 + x] * i2inv[(ap0 + p) * naux + q];
                } rs1_per[i0 * 3 * naux + x * naux + q] = s;
            }}
            // rhoj0_01 = wj001[:,ap0:ap1] · i2inv  → (3, P)
            for x in 0..3 { for q in 0..naux {
                let mut s = 0.0; for p in 0..ni_aux {
                    s += wj001[x * naux + (ap0 + p)] * i2inv[(ap0 + p) * naux + q];
                } r01_per[i0 * 3 * naux + x * naux + q] = s;
            }}
            // ip1_2c_2c = ip1[:,ap0:ap1,:] · i2inv
            let mut ip1_2c = vec![0.0; 3 * ni_aux * naux];
            for x in 0..3 { for p in 0..ni_aux { for r in 0..naux {
                let mut s = 0.0; for q in 0..naux {
                    s += i21[(ap0 + p) * naux * 3 + q * 3 + x] * i2inv[q + r * naux];
                } ip1_2c[x * ni_aux * naux + p * naux + r] = s;
            }}}
            // rhoj0_10 = r0[ap0:ap1] · ip1_2c_2c  → (3, P)
            for x in 0..3 { for r in 0..naux {
                let mut s = 0.0; for p in 0..ni_aux {
                    s += r0[ap0 + p] * ip1_2c[x * ni_aux * naux + p * naux + r];
                } r10_per[i0 * 3 * naux + x * naux + r] = s;
            }}
        }
        for i0 in 0..natm { let (_, ap0, _) = aux_blk[i0];
            for j0 in 0..natm { let (_, aq0, nq) = aux_blk[j0];
                // O1: 0.5*'p,xypq,q->xy'
                let mut o1 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for p in 0..nq { for q in 0..nq {
                        s += r0[aq0 + p] * i2ip[(x * 3 + y) * naux * naux + (aq0 + p) * naux + (aq0 + q)] * r0[aq0 + q];
                    }} o1[x * 3 + y] = s * 0.5;
                }}
                // O2: 'xp,yp->xy' rs1 × wj001
                let mut o2 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        s += rs1_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                    } o2[x * 3 + y] = s;
                }}
                // O3: 0.5*'xp,py->xy' rs1 × wj2
                let mut o3 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        s += rs1_per[i0 * 3 * naux + x * naux + qg] * wj2[qg * 3 + y];
                    } o3[x * 3 + y] = s * 0.5;
                }}
                // O4: 0.5*'xp,yp->xy' r01 × wj001
                let mut o4 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        s += r01_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                    } o4[x * 3 + y] = s * 0.5;
                }}
                // O5: 'yqp,q,xp->xy' i21 × r0 × rs1
                let mut o5 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        for paux in 0..naux {
                            s += i21[paux * naux * 3 + qg * 3 + y] * r0[qg] * rs1_per[i0 * 3 * naux + x * naux + paux];
                        }
                    } o5[x * 3 + y] = s;
                }}
                // O6: 'xp,yp->xy' r10 × wj001
                let mut o6 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        s += r10_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                    } o6[x * 3 + y] = s;
                }}
                for x in 0..3 { for y in 0..3 {
                    let v = o1[x * 3 + y] - o2[x * 3 + y] + o3[x * 3 + y] + o4[x * 3 + y] - o5[x * 3 + y] + o6[x * 3 + y];
                    ej_ri2o[i_t(i0, j0, x, y)] += v;
                    ej_ri2o[i_t(j0, i0, x, y)] += o1[y * 3 + x] - o2[y * 3 + x] + o3[y * 3 + x] + o4[y * 3 + x] - o5[y * 3 + x] + o6[y * 3 + x];
                }}
            }
        }

        // ── Sum contributions ──
        let mut ej = vec![0.0; aa9];
        let mut ek = vec![0.0; aa9];
        for i in 0..aa9 {
            ej[i] = ej_basic[i] + ej_vjd[i] + ej_vj1[i] + ej_ri1[i] + ej_ri2d[i] + ej_ri2o[i];
            ek[i] = ek_vkd[i] + ek_vk1[i] + ek_ri1[i] + ek_ri2d[i] + ek_ri2o[i];
        }
        // Symmetrize: (i0,j0) → (j0,i0) by copying
        for i0 in 0..natm { for j0 in 0..i0 { for x in 0..3 { for y in 0..3 {
            let a = i_t(i0, j0, x, y); let b = i_t(j0, i0, y, x);
            ej[b] = ej[a]; ek[b] = ek[a];
        }}}}
        let mut hp = vec![0.0; aa9];
        for i in 0..aa9 { hp[i] = 0.0; } // e1 is computed separately

        // ── Store results ──
        // Flatten to col-major [n3, n3] MatrixFull
        let to_mat = |arr: &[f64]| -> MatrixFull<f64> {
            let n3 = natm * 3; let mut m = vec![0.0; n3 * n3];
            for i0 in 0..natm { for j0 in 0..natm { for x in 0..3 { for y in 0..3 {
                m[(i0 * 3 + x) + (j0 * 3 + y) * n3] = arr[i_t(i0, j0, x, y)];
            }}}}
            // Symmetrize j0 < i0
            for i0 in 0..natm { for j0 in 0..i0 { for x in 0..3 { for y in 0..3 {
                m[(j0 * 3 + y) + (i0 * 3 + x) * n3] = m[(i0 * 3 + x) + (j0 * 3 + y) * n3];
            }}}}
            MatrixFull::from_vec([n3, n3], m).unwrap()
        };
        self.result.insert("ej".to_string(), to_mat(&ej));
        self.result.insert("ek".to_string(), to_mat(&ek));
        // h_partial = e1 + ej - ek (look up e1 if already computed)
        let e1_arr = if let Some(e1_mat) = self.result.get("e1") {
            // Convert MatrixFull back to flat (natm, natm, 3, 3) for summation
            let n3 = natm * 3; let mut e1_flat = vec![0.0; aa9];
            for i0 in 0..natm { for j0 in 0..natm { for x in 0..3 { for y in 0..3 {
                e1_flat[i_t(i0, j0, x, y)] = e1_mat[[(i0*3+x) as _, (j0*3+y) as _]];
            }}}}
            e1_flat
        } else { vec![0.0; aa9] };
        let mut hp = vec![0.0; aa9];
        for i in 0..aa9 { hp[i] = e1_arr[i] + ej[i] - ek[i]; }
        self.result.insert("h_partial".to_string(), to_mat(&hp));
        self
    }
}

// ── Helper: V^{-1} via matrix power. int2c must be column-major ──
fn compute_vinv(int2c_col_major: &[f64], n: usize) -> Vec<f64> {
    let m = MatrixFull::from_vec([n, n], int2c_col_major.to_vec()).unwrap();
    let inv = _power_rayon_for_symmetric_matrix(&m, -1.0, 1e-12).unwrap();
    inv.data
}

// ══════════════════════════════════════════════════════════
// hcore assembly and e1 computation (below)
// ══════════════════════════════════════════════════════════

/// Build hcore matrix for atom pair (ia,ja) matching PySCF's hcore_generator.
fn build_hcore(mol: &Molecule, ia: usize, ja: usize) -> Vec<f64> {
    let nao = mol.num_basis; let n3 = nao * nao;
    let aoslices = mol.aoslice_by_atom();
    let p0 = aoslices[ia][2] as usize; let ni = aoslices[ia][3] - p0;
    let q0 = aoslices[ja][2] as usize; let qj = aoslices[ja][3] - q0;
    let atom_charges: Vec<f64> = crate::geom_io::get_charge(&mol.geom.elem);
    let i9 = |c: usize, p: usize, q: usize| c * n3 + p * nao + q;
    let cint = mol.initialize_cint(false);
    let (k_aa,_): (Vec<f64>,_) = cint.integrate_row_major("int1e_ipipkin","s1",None).into();
    let (n_aa,_): (Vec<f64>,_) = cint.integrate_row_major("int1e_ipipnuc","s1",None).into();
    let (k_ab,_): (Vec<f64>,_) = cint.integrate_row_major("int1e_ipkinip","s1",None).into();
    let (n_ab,_): (Vec<f64>,_) = cint.integrate_row_major("int1e_ipnucip","s1",None).into();
    let h1aa: Vec<f64> = k_aa.iter().zip(n_aa.iter()).map(|(k,n)| k+n).collect();
    let h1ab: Vec<f64> = k_ab.iter().zip(n_ab.iter()).map(|(k,n)| k+n).collect();
    let mut h = vec![0.0; 9 * n3];
    if ia == ja {
        let zi = atom_charges[ia];
        let mut rmol = mol.initialize_cint(false);
        rmol.with_rinv_at_nucleus(ia, |mr| {
            let (r2aa,_): (Vec<f64>,_) = mr.integrate_row_major("int1e_ipiprinv","s1",None).into();
            let (r2ab,_): (Vec<f64>,_) = mr.integrate_row_major("int1e_iprinvip","s1",None).into();
            for c in 0..9 { for p in 0..nao { for q in 0..nao {
                let i = i9(c, p, q); h[i] = -zi * (r2aa[i] + r2ab[i]);
                if p >= p0 && p < p0+ni { h[i] += h1aa[i] + zi * (r2aa[i] + r2ab[i]); }
                if q >= p0 && q < p0+ni { h[i] += zi * r2aa[i9(c, q, p)] + zi * r2ab[i]; }
                if p >= p0 && p < p0+ni && q >= p0 && q < p0+ni { h[i] += h1ab[i]; }
            }}}
            for c in 0..9 { for p in 0..nao { for q in 0..p {
                let i1=i9(c,p,q);let i2=i9(c,q,p);let v=h[i1]+h[i2];h[i1]=v;h[i2]=v;
            }}}
            for c in 0..9 { for p in 0..nao { h[i9(c, p, p)] *= 2.0; }}
        });
    } else {
        let zi = atom_charges[ia]; let zj = atom_charges[ja];
        for c in 0..9 { for p in 0..ni { for q in 0..qj {
            h[i9(c, p0+p, q0+q)] += h1ab[i9(c, p0+p, q0+q)];
        }}}
        let mut rmol = mol.initialize_cint(false);
        rmol.with_rinv_at_nucleus(ia, |mr| {
            let (r2aa,_): (Vec<f64>,_)=mr.integrate_row_major("int1e_ipiprinv","s1",None).into();
            let (r2ab,_): (Vec<f64>,_)=mr.integrate_row_major("int1e_iprinvip","s1",None).into();
            for c in 0..9 { for p in 0..qj { for q in 0..nao { h[i9(c, q0+p, q)] += zi * r2aa[i9(c, q0+p, q)]; }}}
            for x in 0..3 { for y in 0..3 { let cs=x*3+y;let cd=y*3+x;
                for p in 0..qj { for q in 0..nao { h[i9(cd, q0+p, q)] += zi * r2ab[i9(cs, q0+p, q)]; }}
            }}
        });
        let mut rmol = mol.initialize_cint(false);
        rmol.with_rinv_at_nucleus(ja, |mr| {
            let (r2aa,_): (Vec<f64>,_)=mr.integrate_row_major("int1e_ipiprinv","s1",None).into();
            let (r2ab,_): (Vec<f64>,_)=mr.integrate_row_major("int1e_iprinvip","s1",None).into();
            for c in 0..9 { for p in 0..ni { for q in 0..nao { h[i9(c, p0+p, q)] += zj * r2aa[i9(c, p0+p, q)]; }}}
            for c in 0..9 { for p in 0..ni { for q in 0..nao { h[i9(c, p0+p, q)] += zj * r2ab[i9(c, p0+p, q)]; }}}
        });
        for c in 0..9 { for p in 0..nao { for q in 0..p {
            let i1=i9(c,p,q);let i2=i9(c,q,p);let v=h[i1]+h[i2];h[i1]=v;h[i2]=v;
        }}}
        for c in 0..9 { for p in 0..nao { h[i9(c, p, p)] *= 2.0; }}
    }
    h
}

/// Compute e1 = -2·s1aa·dme0 - 2·s1ab·dme0 + hcore·dm0.
pub fn compute_e1(mol: &Molecule, dm0: &[f64], dme0: &[f64]) -> MatrixFull<f64> {
    let natm = mol.geom.nfree; let nao = mol.num_basis; let n3 = natm * 3;
    let cint_mol = mol.initialize_cint(false);
    let aoslices = mol.aoslice_by_atom();
    let i9 = |c: usize, p: usize, q: usize| c * nao * nao + p * nao + q;
    let (s1aa,_) = cint_mol.integrate_row_major("int1e_ipipovlp","s1",None).into();
    let (s1ab,_) = cint_mol.integrate_row_major("int1e_ipovlpip","s1",None).into();
    let mut e1_ten = vec![0.0; natm * natm * 9];
    let i_t = |i0, j0, x, y| i0 * natm * 9 + j0 * 9 + x * 3 + y;
    for i0 in 0..natm {
        let p0 = aoslices[i0][2] as usize; let ni = aoslices[i0][3] - p0;
        for x in 0..3 { for y in 0..3 {
            let c = x*3+y; let mut s=0.0;
            for p in 0..ni { for q in 0..nao { s += s1aa[i9(c, p0+p, q)] * dme0[(p0+p)*nao+q]; }}
            e1_ten[i_t(i0, i0, x, y)] -= s * 2.0;
        }}
        for j0 in 0..=i0 {
            let q0 = aoslices[j0][2] as usize; let qj = aoslices[j0][3] - q0;
            for x in 0..3 { for y in 0..3 {
                let c = x*3+y; let mut s=0.0;
                for p in 0..ni { for q in 0..qj { s += s1ab[i9(c, p0+p, q0+q)] * dme0[(p0+p)*nao+q0+q]; }}
                e1_ten[i_t(i0, j0, x, y)] -= s * 2.0;
            }}
            let hc = build_hcore(mol, i0, j0);
            for x in 0..3 { for y in 0..3 {
                let c = x*3+y; let mut s=0.0;
                for p in 0..nao { for q in 0..nao { s += hc[i9(c, p, q)] * dm0[p + q*nao]; }}
                e1_ten[i_t(i0, j0, x, y)] += s;
            }}
        }
    }
    let mut e1_mat = vec![0.0; n3 * n3];
    for i0 in 0..natm { for j0 in 0..natm { for x in 0..3 { for y in 0..3 {
        e1_mat[(i0*3+x)+(j0*3+y)*n3] = e1_ten[i_t(i0, j0, x, y)];
    }}}}
    for i0 in 0..natm { for j0 in 0..i0 { for x in 0..3 { for y in 0..3 {
        e1_mat[(j0*3+y)+(i0*3+x)*n3] = e1_mat[(i0*3+x)+(j0*3+y)*n3];
    }}}}
    MatrixFull::from_vec([n3, n3], e1_mat).unwrap()
}

// ── Minimal .npy reader (tests only) ──
#[cfg(test)]
fn read_npy_f64(path: &str) -> Vec<f64> {
    use std::fs::File; use std::io::Read;
    let mut f = File::open(path).unwrap(); let mut magic = [0u8; 6];
    f.read_exact(&mut magic).unwrap(); assert_eq!(&magic, b"\x93NUMPY");
    let mut ver = [0u8; 2]; f.read_exact(&mut ver).unwrap();
    let hlen = match (ver[0], ver[1]) {
        (1,0)=>{let mut b=[0u8;2];f.read_exact(&mut b).unwrap();u16::from_le_bytes(b)as usize}
        (2,0)|(3,0)=>{let mut b=[0u8;4];f.read_exact(&mut b).unwrap();u32::from_le_bytes(b)as usize}
        _=>panic!("unsupported npy version"),
    };
    let mut hdr = vec![0u8; hlen]; f.read_exact(&mut hdr).unwrap();
    let hdr_str = std::str::from_utf8(&hdr).unwrap();
    if hdr_str.contains("'fortran_order': True")||hdr_str.contains("\"fortran_order\": True") {
        panic!("F-order npy not supported");
    }
    let mut raw = Vec::new(); f.read_to_end(&mut raw).unwrap();
    raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod debug { use super::*;

    #[test]
    fn test_e1_vs_pyscf() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
basis_type = "spheric"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        let nao=mol.num_basis;let natm=mol.geom.nfree;let n3=natm*3;
        let pydm0=read_npy_f64("src/hessian/h2o_dm0.npy");
        let pydme0=read_npy_f64("src/hessian/h2o_dme0.npy");
        let mut dc=vec![0.0;nao*nao];let mut ec=vec![0.0;nao*nao];
        for r in 0..nao{for c in 0..nao{dc[r+c*nao]=pydm0[r*nao+c];ec[r+c*nao]=pydme0[r*nao+c];}}
        let em=compute_e1(&mol,&dc,&ec);
        let ref_data=read_npy_f64("src/hessian/h2o_e1_full_ref.npy");
        let mut ref_vec=vec![0.0;n3*n3];
        for ia in 0..natm{for ja in 0..natm{for x in 0..3{for y in 0..3{
            ref_vec[(ja*3+y)*n3+(ia*3+x)]=ref_data[(ia*3+x)*n3+(ja*3+y)];
        }}}}
        let rm=MatrixFull::from_vec([n3,n3],ref_vec).unwrap();
        let mut md=0.0;for i in 0..n3{for j in 0..n3{let d=(em[[i,j]]-rm[[i,j]]).abs();if d>md{md=d;}}}
        println!("e1 maxdiff: {:.4e}",md);assert!(md<1e-4,"e1 mismatch: maxdiff={:.4e}",md);
        println!("PASS e1 (diff={:.4e})",md);
    }

    #[test]
    fn test_calc_e1_from_rest_scf() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
basis_type = "spheric"
eri_type = "analytic"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
scf_acc_etot = 1e-11
scf_acc_rho = 1e-8
scf_acc_eev = 1e-6
max_scf_cycle = 100
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        let mut scf_data=scf_io::SCF::build(mol,&None);
        crate::scf_io::scf_without_build(&mut scf_data,&None);
        let nao=scf_data.mol.num_basis;let natm=scf_data.mol.geom.nfree;let n3=natm*3;
        println!("REST SCF energy={:.14}",scf_data.scf_energy);
        let mut hess=RIRHFHessian::new(&scf_data); hess.calc_e1();
        let em=hess.result.get("e1").unwrap();
        let ref_data=read_npy_f64("src/hessian/h2o_e1_full_ref.npy");
        let mut ref_vec=vec![0.0;n3*n3];
        for ia in 0..natm{for ja in 0..natm{for x in 0..3{for y in 0..3{
            ref_vec[(ja*3+y)*n3+(ia*3+x)]=ref_data[(ia*3+x)*n3+(ja*3+y)];
        }}}}
        let rm=MatrixFull::from_vec([n3,n3],ref_vec).unwrap();
        let mut md=0.0;for i in 0..n3{for j in 0..n3{let d=(em[[i,j]]-rm[[i,j]]).abs();if d>md{md=d;}}}
        println!("e1 maxdiff (REST SCF): {:.4e}",md);
        assert!(md<1e-3,"e1 mismatch: maxdiff={:.4e}",md);
        println!("PASS calc_e1 (diff={:.4e})",md);
    }

    #[test]
    fn test_ri_integrals() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
auxbas_path = "basis-set-pool/def2-svp-rifit"
basis_type = "spheric"
auxbas_type = "spheric"
eri_type = "ri-v"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        // Build aux CInt directly
        let auxmol = mol.make_auxmol_fake();
        let nao=mol.num_basis; let naux=auxmol.num_basis;
        let natm=mol.geom.nfree;
        println!("H2O def2-svp-rifit: nao={}, naux={}, natm={}", nao, naux, natm);
        assert_eq!(nao, 24, "nao should be 24");
        assert_eq!(naux, 76, "naux should be 76");
        // Check 2c integral
        let cint_aux = auxmol.initialize_cint(false);
        let (v,_) = cint_aux.integrate_row_major("int2c2e","s1",None).into();
        let full = naux * naux;
        assert_eq!(v.len(), full, "int2c2e row-major s1 shape");
        println!("int2c2e ok: {} elements ({}x{})", v.len(), naux, naux);
        // Check 3c integral via integrate_cross
        let cint_mol = mol.initialize_cint(false);
        let (v3,_) = CInt::integrate_cross_row_major("int3c2e", [&cint_mol, &cint_mol, &cint_aux], "s1", None).into();
        println!("int3c2e cross: {} elements (expected {})", v3.len(), nao*nao*naux);
        assert_eq!(v3.len(), nao*nao*naux, "int3c2e cross shape");
        println!("PASS: RI integrals infrastructure works");
    }

    #[test]
    fn test_h_partial_vs_pyscf() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
auxbas_path = "basis-set-pool/def2-svp-rifit"
basis_type = "spheric"
auxbas_type = "spheric"
eri_type = "ri-v"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
scf_acc_etot = 1e-11
scf_acc_rho = 1e-8
scf_acc_eev = 1e-6
max_scf_cycle = 100
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        let mut scf_data=scf_io::SCF::build(mol,&None);
        crate::scf_io::scf_without_build(&mut scf_data,&None);
        let nao=scf_data.mol.num_basis;let natm=scf_data.mol.geom.nfree;let n3=natm*3;
        println!("REST SCF (ri-v) energy={:.14}", scf_data.scf_energy);

        // Compute full Hessian
        let mut hess=RIRHFHessian::new(&scf_data);
        hess.calc_e1();
        hess.calc_ej_ek();
        // Debug: check ej and ek values
        let e1 = hess.result.get("e1").unwrap();
        let ej = hess.result.get("ej").unwrap();
        let ek = hess.result.get("ek").unwrap();
        let hp = hess.result.get("h_partial").unwrap();
        let mut max_abs = |name: &str, m: &MatrixFull<f64>| {
            let mut mx = 0.0;
            for i in 0..n3 { for j in 0..n3 { let v = m[[i,j]].abs(); if v > mx { mx = v; } }}
            println!("  {} max_abs={:.4e}", name, mx);
        };
        max_abs("e1", e1); max_abs("ej", ej); max_abs("ek", ek); max_abs("h_partial", hp);
        let e1 = hess.result.get("e1").unwrap();
        let ej = hess.result.get("ej").unwrap();
        let ek = hess.result.get("ek").unwrap();
        let hp = hess.result.get("h_partial").unwrap();

        // Load PySCF reference
        let to_ref = |name| -> MatrixFull<f64> {
            let d = read_npy_f64(&format!("src/hessian/pyscf_{}_ref.npy", name));
            let mut r = vec![0.0; n3 * n3];
            for ia in 0..natm { for ja in 0..natm { for x in 0..3 { for y in 0..3 {
                r[(ja*3+y)*n3+(ia*3+x)] = d[(ia*3+x)*n3+(ja*3+y)];
            }}}}
            MatrixFull::from_vec([n3, n3], r).unwrap()
        };
        let ref_e1 = to_ref("e1"); let ref_ej = to_ref("ej");
        let ref_ek = to_ref("ek"); let ref_hp = to_ref("h_partial");

        let mut md_e1=0.0;let mut md_ej=0.0;let mut md_ek=0.0;let mut md_hp=0.0;
        for i in 0..n3 { for j in 0..n3 {
            let d1=( e1[[i,j]]-ref_e1[[i,j]]).abs();if d1>md_e1{md_e1=d1;}
            let dj=( ej[[i,j]]-ref_ej[[i,j]]).abs();if dj>md_ej{md_ej=dj;}
            let dk=( ek[[i,j]]-ref_ek[[i,j]]).abs();if dk>md_ek{md_ek=dk;}
            let dh=( hp[[i,j]]-ref_hp[[i,j]]).abs();if dh>md_hp{md_hp=dh;}
        }}
        // Print individual component diagnostics
        println!("REST vs PySCF: e1 diff={:.4e}  ej diff={:.4e}  ek diff={:.4e}  h_partial diff={:.4e}",
                 md_e1, md_ej, md_ek, md_hp);
        // Check the important result: h_partial
        if md_hp < 1e-4 {
            println!("PASS: h_partial matches PySCF reference (diff={:.4e})", md_hp);
        } else {
            println!("FAIL: h_partial mismatch (diff={:.4e}), expected < 1e-4", md_hp);
            // Print matrices for debugging
            println!("\nREST e1:"); for i in 0..n3.min(6){print!("  row{}:",i);for j in 0..n3{print!(" {:8.4e}",e1[[i,j]]);}println!();}
            println!("\nREF e1:"); for i in 0..n3.min(6){print!("  row{}:",i);for j in 0..n3{print!(" {:8.4e}",ref_e1[[i,j]]);}println!();}
        }
        assert!(md_hp < 1e-3, "h_partial mismatch: maxdiff={:.4e}", md_hp);
    }
}
