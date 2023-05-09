from rdkit import Chem
from rdkit.Chem import Mol, AllChem


class Featurizer:
    def __init__(self, mol: Mol):
        self.mol = mol

    def get_features(self):
        ...

    @staticmethod
    def collate_fn(encodings):
        ...

    @classmethod
    def from_pdb(cls, pdb_file):
        mol = Chem.MolFromPDBFile(pdb_file.as_posix())
        return cls(mol)

    @classmethod
    def from_smiles(cls, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except ValueError:
            mol = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(mol)

        return cls(mol)
