"""Extracts features from smiles based on different models."""

from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops

"""
fingerprint_features : Sevrier default feature extractor from a smile
"""
def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=size, useChirality=True, useBondTypes=True, useFeatures=False
    )

def make_mole(smile_string):
    return MolFromSmiles(smile_string)
