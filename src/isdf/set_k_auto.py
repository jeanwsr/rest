# requirements: rdkit and obabel
# obabel has been installed in obabel_minimal
# please set BABEL_LIBDIR="/path/to/rest/src/isdf/obabel_minimal/obabel_build/lib/"
import sys
from pathlib import Path
import subprocess
from typing import List
import tomli
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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
ELEMENTS_DICT = {e: i for i, e in enumerate(ELEMENTS_LIST)}

def atomlabel(atom: str):
    z = ELEMENTS_DICT[atom]
    if z == 0:
        return 0
    elif 1 <= z <= 2:
        return 1
    elif 3 <= z <= 10 and (z != 9):
        return 2
    elif z == 9:
        return 2.5
    elif 11 <= z <= 18:
        return 3
    elif 19 <= z <= 36:
        return 4
    elif 37 <= z <= 54:
        return 5
    elif 55 <= z <= 86:
        return 6
    elif 87 <= z <= 118:
        return 7
    else:
        raise ValueError

def ctrl2info(current_dir: Path, ctrl_file: Path):
    obabel_bin = current_dir / 'obabel_minimal' / 'obabel_build' / 'bin' / 'obabel'
    if not ctrl_file.exists():
        raise ValueError('This must be a bug!')
    with open(ctrl_file,'rb') as f:
        data = tomli.load(f)
        charge = data['ctrl']['charge']
        spin = data['ctrl']['spin']
        atoms = data['geom']['position']
        basis = data['ctrl']['basis_path'].split('/')[-1]

    if isinstance(atoms,List):
        atoms = '\n'.join(atoms)

    # normalize
    geom = []
    natm = 0
    for some in atoms.split('\n'):
        if len(some.split()) == 4:
            geom.append(some.lstrip())
            natm += 1
    geom = '\n'.join(geom)

    # build xyz-like string
    geom = f'{natm}\n\n' + geom

    result = subprocess.run(
        [str(obabel_bin), '-ixyz', '-osmi'],
        input=geom,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    smiles = result.stdout.strip()
    assert smiles is not None
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    atom_list = []
    Chem.SanitizeMol(mol,sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES)
    for atom in mol.GetAtoms():
        atom_list.append(atom.GetSymbol())
        num_H = atom.GetNumImplicitHs()
        atom_list.extend(['H'] * num_H)
    info_dict = {}
    # get ring atoms
    ring_atoms = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.IsInRing()]
    info_dict['ring_atoms'] = ring_atoms
    info_dict['num_ring_atoms'] = len(ring_atoms)
    # get formula
    formula = rdMolDescriptors.CalcMolFormula(mol)
    info_dict['formula'] = formula
    # get charge and spin
    info_dict['charge'] = charge
    info_dict['spin'] = spin
    # get label
    tot_label = sum(atomlabel(atom) for atom in atom_list) + len(ring_atoms)
    info_dict['label'] = tot_label / natm
    # get basis
    info_dict['basis'] = basis
    return info_dict

def label2k(label: float, basis: str):
    if label >= 3.0:
        tmp_k = 3
    elif label >= 2.5:
        tmp_k = 4
    elif label >= 2.0:
        tmp_k = 5
    else:
        tmp_k = 6
    if basis.lower() in ['cc-pvdz','def2-sv(p)','def2-svp','6-31gs']:
        return tmp_k
    if basis.lower() in ['cc-pvtz','def2-tzvp']:
        return int(tmp_k * 1.8 + 1)
    raise NotImplementedError

def main():
    if len(sys.argv) != 3:
        raise ValueError('This must be a bug!')

    cdir = Path(sys.argv[1])
    cfile = Path(sys.argv[2])

    info = ctrl2info(cdir, cfile)
    result = label2k(info['label'], info['basis'])
    print(result)

if __name__ == '__main__':
    main()
