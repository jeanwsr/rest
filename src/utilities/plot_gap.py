import matplotlib.pyplot as plt
import numpy as np
import argparse

ELEMENTS_LIST = ['d',
  'H' ,                                                                                                 'He',
  'Li', 'Be',                                                             'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
  'Na', 'Mg',                                                             'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
  'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
  'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
  'Cs', 'Ba', 
              'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                    'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
]
# element symbols to atomic numbers
ELEMENTS_DICT = {e.lower():i for i, e in enumerate(ELEMENTS_LIST)}

def str2bool(arg):
    if isinstance(arg,bool):
        return arg
    else:
        if arg.lower() in ['no','false']:
            return False
        elif arg.lower() in ['yes','true']:
            return True
        else:
            raise ValueError('Expect a bool value but got %s instead'%arg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',type=str,default='rest.out',help='Path to the output file from REST(default rest.out)')
    parser.add_argument('--occ_color',type=str,default='black',help='Color of occupied orbitals(default black)')
    parser.add_argument('--vir_color',type=str,default='blue',help='Color of virtue orbitals(default blue)')
    parser.add_argument('--linewidth',type=float,default=2.0,help='Linewidth of the figure(default 2.0)')
    parser.add_argument('--min',type=float,default=-30.0,help='The minimum of the energy(eV) axis(default -30.0)')
    parser.add_argument('--max',type=float,default=80.0,help='The maximum of the energy(eV) axis(default 80.0)')
    parser.add_argument('-o','--output',type=str,default='rest.png',help='Path to the output figure of band(default rest.png')
    parser.add_argument('--half',type=str2bool,default=False,help='Whether only half of the first Brillouin Zone will be plotted(default False)')
    parser.add_argument('--plot_fermi',type=str2bool,default=False,help='Whether Fermi energy will be shown(default False)')
    parser.add_argument('--fermi_color',type=str,default='gray',help='Color of Fermi energy(default gray, ignored if plot_fermi=False)')
    args = parser.parse_args()
    
    filename = args.input
    occ_num = None
    with open(filename,'r') as f:
        k_points = []
        energy = []
        g = f.readlines()
        for line_idx in range(len(g)):
            if 'Unit Cell' in g[line_idx]:
                elem_list = "".join(g[line_idx].split()[2:])
                occ_num = sum(ELEMENTS_DICT[e.lower()] for e in eval(elem_list))//2
            elif 'k =' in g[line_idx]:
                tmp_k = float(g[line_idx].split()[-1])
                k_points.append(tmp_k)
                tmp_split = g[line_idx+1].split()[3:]
                final_split = np.array(eval("".join(tmp_split)))
                energy.append(final_split)
            elif 'Occupied orbitals' in g[line_idx]:
                occ_num = int(g[line_idx].split()[-1])
            else:
                pass
    if occ_num is None:
        raise ValueError('Can not get the number of occupied orbitals from the input file')
    k_points = np.array(k_points)
    energy = np.stack(energy)
    
    for m in range(energy[0].shape[0]):
        if m <= occ_num-1:
            plt.plot(k_points,energy[:,m]*27.2113863,linewidth=args.linewidth,color=args.occ_color)
        else:
            plt.plot(k_points,energy[:,m]*27.2113863,linewidth=args.linewidth,color=args.vir_color)
    if args.plot_fermi:
        gap = energy[:,occ_num] - energy[:,occ_num-1]
        min_idx = gap.argmin()
        fermi_face = (energy[min_idx,occ_num] + energy[min_idx,occ_num-1])/2*27.2113863
        plt.axhline(y=fermi_face, color=args.fermi_color, linestyle='--', linewidth=1)
        plt.text(-0.02, fermi_face, r'$\epsilon_F$',transform=plt.gca().get_yaxis_transform(),fontsize=12,color='black', va='center', ha='right')
    
    plt.ylabel('Energy/eV')
    plt.xlabel('k points')
    plt.ylim((args.min,args.max))
    if args.half:
        plt.xlim((0,k_points.max().item()))
    else:
        plt.xlim((k_points.min().item(),k_points.max().item()))
    plt.savefig(args.output)
