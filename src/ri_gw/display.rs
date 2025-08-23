pub fn full_quasiparticles(quasiparticle_energies:&Vec<f64>,occ_size:usize){
    print!("Quasiparticle Energies from GW Approximation (in Hatree):\n");
    quasiparticle_energies.iter().enumerate().for_each(|(n,e_n)|{
        println!(" QP Energy of Orbital #{}:{}",n,e_n);
        if n==occ_size-1{
            print!("-------Fermi Surface--------\n");
        }
    })
}
pub fn extrapolation_quasiparticles(quasiparticle_energies:&Vec<f64>,occ_size:usize,calc_indices:&Vec<usize>,threshold:f64){
    print!("Quasiparticle Energies from GW Approximation (in Hatree):\n");
    println!("   Note that only energies around the Fermi surface within {} Hatree are explicitly computed by solving the quasiparticle equation,",threshold);
    print!("    while others are extrapolated using average energy shifts of the computed ones compared to corresponding SCF results\n");
    print!("    You can set this threshold to be any other float value using the keyword \"threshold\"\n");
    quasiparticle_energies.iter().enumerate().for_each(|(n,e_n)|{
        let e_type=if calc_indices.contains(&n){"calculated"}else{"extrapolated"};
        println!(" ({}) QP Energy of Orbital #{}:{}",e_type,n,e_n);
        if n==occ_size-1{
            print!("-------Fermi Surface--------\n");
        }
    })
}