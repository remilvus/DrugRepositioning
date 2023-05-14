from rdkit import Chem
from rdkit.Chem import Mol, AllChem
from rdkit.Geometry.rdGeometry import Point3D


class Featurizer:
    def __init__(
        self, use_bonds: bool = True, cutout: bool = False, cutout_radius: float = 5.0
    ):
        self.mol = None
        self.use_bonds = use_bonds
        self.cutout = cutout
        self.cutout_radius = cutout_radius

    def load_from_pdb(self, pdb_file, cutout_center):
        assert not self.use_bonds
        self.mol = Chem.MolFromPDBFile(pdb_file.as_posix())
        if self.cutout:
            cutout_mol = Chem.RWMol()
            atom_positions = []
            for atom in self.mol.GetAtoms():
                pos = self.mol.GetConformer().GetAtomPosition(atom.GetIdx())
                distance = Point3D(pos.x, pos.y, pos.z).Distance(
                    Point3D(*cutout_center)
                )
                if distance <= self.cutout_radius:
                    new_atom = Chem.Atom(atom.GetAtomicNum())
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    cutout_mol.AddAtom(new_atom)
                    atom_positions.append(pos)
            conformer = Chem.Conformer(len(atom_positions))
            for i, pos in enumerate(atom_positions):
                conformer.SetAtomPosition(i, pos)
            cutout_mol.AddConformer(conformer)
            self.mol = cutout_mol.GetMol()
        return self

    def load_from_smiles(self, smiles):
        assert not self.cutout
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except ValueError:
            mol = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(mol)
        self.mol = mol
        return self

    def get_features(self):
        ...

    @staticmethod
    def collate_fn(encodings):
        ...
