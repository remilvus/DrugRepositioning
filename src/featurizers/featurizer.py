from rdkit import Chem


class Featurizer:
    def __init__(self, mol):
        self.mol = mol

    def get_features(self):
        ...

    @staticmethod
    def collate_fn(features):
        ...

    @classmethod
    def from_pdb(cls, pdb_file):
        mol = Chem.MolFromPDBFile(pdb_file)
        return cls(mol)

    @classmethod
    def from_smiles(cls, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return cls(mol)
