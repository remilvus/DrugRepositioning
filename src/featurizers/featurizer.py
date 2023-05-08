from rdkit import Chem
from rdkit.Chem import Mol


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
        mol = Chem.MolFromPDBFile(pdb_file)
        return cls(mol)

    @classmethod
    def from_smiles(cls, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return cls(mol)
