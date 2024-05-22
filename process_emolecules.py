#!/usr/bin/env python3
# coding: utf-8

'''

Processing the raw SDF from eMolecules search

The eMolecules database was queried on March 5th, 2024
using just plain pyridine (C1=NC=CC=C1). The structure
was drawn into the search window (i.e., we didn't search
for the smiles string).

The initial results were close to 1 million compounds, so
we limited the molecular weight to 300 g/mol. This produced
exactly 100,000 compounds in the search.

'''

import re
from multiprocessing import Pool
from pprint import pprint

from pathlib import Path
from itertools import repeat

# Installed
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, Draw
from rdkit.Chem.Fragments import fr_isothiocyan

from drawing import draw_molecules_to_grid_image

def flatten(xss):
    return [x for xs in xss for x in xs]

SDF_SMILES_PATTERN = re.compile(r'(?<=<SMILES>\n)(.*?)(?=\n)')
PYRIDINE = 'c:1:n:c:c:c:c:1'
H_2_PYR = '[#1]~c~1~n~c~c~c~c~1'
H_3_PYR = '[#1]~c~1~c~c~c~n~c~1'
H_4_PYR = '[#1]~c~1~c~c~n~c~c~1'
SULFONYL_CHLORIDE = 'S([F,Cl,Br,I])(=O)=O'
ACID_HALIDE = 'C([F,Cl,Br,I])=O'
PYRADINONE = 'O=c:1:c:c:c:c:n1'
FOUR_PYRADINONE = 'O=[#6]~1~[#6]~[#6]~[#7]~[#6]~[#6]~1'




def get_smiles_from_sdf(text: str) -> int:
    '''
    Reads the raw text of an sdf file and searches for
    the text between the following two symbols

    Symbol 1:       >  <SMILES>\n
    Symbol 2:       \n

    '''
    return re.findall(SDF_SMILES_PATTERN, text)


def canonicalize_smiles(smiles: str) -> str:
    '''
    Creates an RDKit mol object from a SMILES
    string and converts the mol into a canonical
    SMILES string.

    Parameters
    ----------
    smiles: str
        SMILES string to be canonicalized

    Returns
    ----------
    str
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        print(f'SMILES {smiles} made None mol.')
        return None

def extract_smiles_from_sdf():
    sdf = Path('./EMOLECULES_PYRIDINES_300MWT.sdf')

    # Get the raw text
    print('Opening file...')
    with open(sdf, 'r') as infile:
        text = infile.read()

    smiles = get_smiles_from_sdf(text)
    print(f'There are {len(set(smiles))} unqiue SMILES of {len(smiles)} total smiles.')

    print(f'Working on cleaning smiles')
    with Pool() as pool:
        unique_smiles = pool.map(canonicalize_smiles, list(set(smiles)))

    # Filter out the None molecules
    unique_smiles = [x for x in unique_smiles if x is not None]

    # Write to file
    print(f'Writing {len(unique_smiles)} to file.')
    with open('unique_smiles.txt', 'w') as o:
        for s in unique_smiles:
            o.write(f'{s}\n')

def get_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    return mol


def get_mol_logp(s: str) -> float:
    '''
    Returns the smiles and logp
    '''
    mol = Chem.MolFromSmiles(s)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    return s, Crippen.MolLogP(mol, includeHs=False)

def get_mol_wt(s: str) -> float:
    '''
    Returns the smiles and logp
    '''
    mol = Chem.MolFromSmiles(s)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    return Descriptors.MolWt(mol)

def one_filter_log_p():
    # Then we processed these smiles for their Lipinski's rule's
    # Load in the smiles
    with open('./data/unique_smiles.txt', 'r') as infile:
        smiles = [x.strip() for x in infile.readlines()]

    print(f'Canonicalizing smiles')
    with Pool() as pool:
        smiles = pool.map(canonicalize_smiles, list(set(smiles)))

    print(f'Calculating logP')
    with Pool() as pool:

        # Get a list of smiles,logp tuples
        smiles_logp = pool.map(get_mol_logp, list(set(smiles)))

    df = pd.DataFrame(data=smiles_logp, columns=['SMILES', 'Crippen_logp'])
    print(f'The DataFrame has {df.shape[0]} entries in it.')

    df = df[df['Crippen_logp'].astype(float) <= 5]
    print(f'The DataFrame has {df.shape[0]} entries in it after filtering Crippen_logp <= 5.')

    df.to_csv('./results/1-filter-crippen.csv', index=False)

def _get_lipinski(s: str) -> list[str, int, int, int, float, int, float]:
    '''
    Gets the following properties

    NumHDonors
    NumHAcceptors
    NumRotatableBonds
    TPSA
    natoms
    refractivity
    '''

    #print(s, type(s))
    #exit()
    s = str(s)
    mol = Chem.MolFromSmiles(s)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    NumHDonors = Lipinski.NumHDonors(mol)
    NumHAcceptors = Lipinski.NumHAcceptors(mol)
    NumRotatableBonds = Lipinski.NumRotatableBonds(mol)
    TPSA =  Descriptors.TPSA(mol)
    natoms = int(len(mol.GetAtoms()))
    refractivity = Crippen.MolMR(mol)

    #print(f'X{s}X', type(s), len([NumHDonors, NumHAcceptors, NumRotatableBonds, TPSA, natoms, refractivity]), [NumHDonors, NumHAcceptors, NumRotatableBonds, TPSA, natoms, refractivity])

    return [s, NumHDonors, NumHAcceptors, NumRotatableBonds, TPSA, natoms, refractivity]

def two_filter_other_lipinski() -> pd.DataFrame:
    # Then we processed these smiles for their Lipinski's rule's
    # Load in the smiles
    df = pd.read_csv('./results/1-filter-crippen.csv', header=0)

    #df = df.loc[:10]

    print(f'Calculating lipinski')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_get_lipinski, list(set(df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'TPSA', 'natoms', 'refractivity'])
    data_df.set_index('SMILES', inplace=True)
    df.set_index('SMILES', inplace=True)

    df = pd.concat([df, data_df], axis=1)

    print(f'The DataFrame has {df.shape[0]} entries in it.')
    df = df[df['NumHDonors'] <= 5]
    df = df[df['NumHAcceptors'] <= 10]
    df = df[df['NumRotatableBonds'] <= 10]
    df = df[df['TPSA'] <= 140]
    df = df[df['natoms'] <= 70]
    df = df[df['natoms'] >= 20]
    df = df[df['refractivity'] <= 130]
    df = df[df['refractivity'] >= 40]
    print(f'The DataFrame has {df.shape[0]} entries in it after filtering a melange of Lipinski rules.')

    df.reset_index(inplace=True)

    return df

def _meets_pyridine_specs(smiles: str) -> tuple[str, bool]:
    '''
    Has a hydrogen on pyridine
    '''
    smiles = str(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    for pattern in [H_2_PYR, H_3_PYR, H_4_PYR]:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        if matches != ():
            return smiles, True
    return smiles, False

def _has_multiple_pridines(smiles: str) -> tuple[str, bool]:
    smiles = str(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(PYRIDINE))
    if len(matches) == 1:
        return smiles, False
    else:
        return smiles, True

def filter_pyirdines_without_hydrogen(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Takes a pd.DataFrame with a SMILES column
    and matches it against a smarts pattern to
    determine those that do or do not have
    a pyridine hydrogen

    Returns the filtered_df and the removed_df
    '''
    assert 'SMILES' in _df.columns

    print('Testing whether pyridines meet spec')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_meets_pyridine_specs, list(set(_df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'HAS_PYR_H'])
    data_df.set_index('SMILES', inplace=True)
    _df.set_index('SMILES', inplace=True)

    _df = pd.concat([df, data_df], axis=1)
    return _df[_df['HAS_PYR_H']].copy(deep=True), _df[~_df['HAS_PYR_H']].copy(deep=True)

def filter_multiple_pyridines(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Returns the filtered_df and the removed_df
    '''
    assert 'SMILES' in _df.columns

    print('Testing whether smiles has multiple pyridines')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_has_multiple_pridines, list(set(_df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'HAS_MULTIPLE_PYR'])
    data_df.set_index('SMILES', inplace=True)
    _df.set_index('SMILES', inplace=True)

    _df = pd.concat([df, data_df], axis=1)
    return _df[~_df['HAS_MULTIPLE_PYR']].copy(deep=True), _df[_df['HAS_MULTIPLE_PYR']].copy(deep=True)

def _smiles_has_fragments(s: str) -> bool:
    '''
    Determines if a smiles has a period in it
    '''
    if '.' in s:
        return True
    return False

def _smiles_has_isothiocyanate(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    if fr_isothiocyan(mol):
        return smiles, True
    return smiles, False

def _smiles_has_three_coordinate_nitrogen(smiles: str) -> bool:
    mol = get_mol(smiles)
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts('c:n(c):c'))
    if matches == ():
        return smiles, False
    else:
        return smiles, True

def filter_functional_groups(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''

    Returns the filtered_df and the removed_df
    '''
    assert 'SMILES' in _df.columns

    print('Filtering functional groups')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_smiles_has_isothiocyanate, list(set(_df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'HAS_ISOTHIOCYANATE'])
    data_df.set_index('SMILES', inplace=True)
    _df.set_index('SMILES', inplace=True)
    _df = pd.concat([df, data_df], axis=1)

    _failed = _df[_df['HAS_ISOTHIOCYANATE']].copy(deep=True)
    _df = _df[~_df['HAS_ISOTHIOCYANATE']].copy(deep=True)

    return _df, _failed

def filter_three_coordinate_nitrogen(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''

    Returns the filtered_df and the removed_df
    '''
    assert 'SMILES' in _df.columns

    print('Filtering bad nitrogen atoms')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_smiles_has_three_coordinate_nitrogen, list(set(_df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'HAS_BAD_NITROGEN'])
    data_df.set_index('SMILES', inplace=True)
    _df.set_index('SMILES', inplace=True)
    _df = pd.concat([df, data_df], axis=1)

    _failed = _df[_df['HAS_BAD_NITROGEN']].copy(deep=True)
    _df = _df[~_df['HAS_BAD_NITROGEN']].copy(deep=True)

    return _df, _failed

def _smiles_has_sulfonyl_halide(smiles) -> tuple[str, bool]:
    mol = get_mol(smiles)
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(SULFONYL_CHLORIDE))
    if matches == ():
        return smiles, False
    else:
        return smiles, True

def filter_has_sulfonyl_halide(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Returns the filtered_df and the removed_df
    '''
    assert 'SMILES' in _df.columns

    print('Filtering sulfonyl halides')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_smiles_has_sulfonyl_halide, list(set(_df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'HAS_SULFONYL_HALIDE'])
    data_df.set_index('SMILES', inplace=True)
    _df.set_index('SMILES', inplace=True)
    _df = pd.concat([df, data_df], axis=1)

    _failed = _df[_df['HAS_SULFONYL_HALIDE']].copy(deep=True)
    _df = _df[~_df['HAS_SULFONYL_HALIDE']].copy(deep=True)
    print(f'{_failed.shape[0]} molecules were removed. {_df.shape[0]} remain.')

    return _df, _failed


def _smiles_has_acid_halides(smiles) -> tuple[str, bool]:
    mol = get_mol(smiles)
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ACID_HALIDE))
    if matches == ():
        return smiles, False
    else:
        return smiles, True

def filter_has_acid_halides(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Returns the filtered_df and the removed_df
    '''
    assert 'SMILES' in _df.columns

    print('Filtering acid halides')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_smiles_has_acid_halides, list(set(_df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'HAS_ACID_HALIDES'])
    data_df.set_index('SMILES', inplace=True)
    _df.set_index('SMILES', inplace=True)
    _df = pd.concat([df, data_df], axis=1)

    _failed = _df[_df['HAS_ACID_HALIDES']].copy(deep=True)
    _df = _df[~_df['HAS_ACID_HALIDES']].copy(deep=True)
    print(f'{_failed.shape[0]} molecules were removed. {_df.shape[0]} remain.')

    return _df, _failed




def _smiles_has_pyradinone(smiles) -> tuple[str, bool]:
    mol = get_mol(smiles)
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(PYRADINONE))
    if matches != ():
        return smiles, True

    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(FOUR_PYRADINONE))
    if matches != ():
        return smiles, True

    return smiles, False

def filter_has_pyradinone(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Returns the filtered_df and the removed_df
    '''
    assert 'SMILES' in _df.columns

    print('Filtering PYRADINONE')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_smiles_has_pyradinone, list(set(_df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'HAS_PYRADINONE'])
    data_df.set_index('SMILES', inplace=True)
    _df.set_index('SMILES', inplace=True)
    _df = pd.concat([df, data_df], axis=1)
    print(df)

    _failed = _df[_df['HAS_PYRADINONE']].copy(deep=True)
    _df = _df[~_df['HAS_PYRADINONE']].copy(deep=True)
    print(f'{_failed.shape[0]} molecules were removed. {_df.shape[0]} remain.')

    return _df, _failed

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def _smiles_has_quinoline_or_5_membered_n_heterocycle(smiles) -> tuple[str, bool]:
    mol = get_mol(smiles)

    PYRROLE = '[#6]1~[#6]~[#6]~[#6]~[#7]~1'
    PYRAZOLE = '[#6]1~[#6]~[#6]~[#7]~[#7]~1'
    IMIDAZOLE = 'c1:c:n:c:n:1'
    TRIAZOLE_1 = 'c1:c:n:n:n:1'
    TRIAZOLE_2 = 'c1:n:c:n:n:1'
    TETRAZOLE = '[#6]1~[#7]~[#7]~[#7]~[#7]~1'
    QUINOLINE = 'c:1:2:c:c:c:c:c:1:c:c:c:n:2'
    ISOQUINOLINE = 'c:1:2:c:c:c:c:c:1:c:c:n:c:2'

    THIENOPYRIDINE_1 = 'c:1:c:n:c:2:s:c:c:c:2:c:1'
    THIENOPYRIDINE_2 = 'c:1:c:c:2:c:c:s:c:2:c:n:1'
    THIENOPYRIDINE_3 = 'c:1:c:c:2:s:c:c:c:2:c:n:1'
    THIENOPYRIDINE_4 = 'c:1:c:n:c:2:c:c:s:c:2:c:1'
    THIAZOLE = 'c1:c:s:c:n:1'
    OXADIAZOLE = 'c1:n:c:o:n:1'
    OXAZOLE = 'c1:c:o:c:n:1'


    for filter in [PYRROLE, PYRAZOLE, IMIDAZOLE, TRIAZOLE_1, TRIAZOLE_2, TETRAZOLE, QUINOLINE, ISOQUINOLINE,
                   THIENOPYRIDINE_1, THIENOPYRIDINE_2, THIENOPYRIDINE_3, THIENOPYRIDINE_4, THIAZOLE,
                   OXADIAZOLE, OXAZOLE]:

        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(filter))

        #if matches == ():
        #    print(f'No match with {namestr(filter, locals())}')
        #else:
        #    print(f'Match with {namestr(filter, locals())}')
        #    Draw.ShowMol(mol, highlightAtoms=matches[0])

        if matches != ():
            return smiles, True

    return smiles, False

def filter_bad_heterocycles(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Filters quinolines and 5 membered nitrogen heterocycles
    '''
    assert 'SMILES' in _df.columns

    print('Filtering HAS_5MEMBERED_HETEROCYCLE_OR_QUINOLINE')
    with Pool() as pool:
        # Get a list of smiles,logp tuples
        list_of_lists = pool.map(_smiles_has_quinoline_or_5_membered_n_heterocycle, list(set(_df['SMILES'])))
    data_df = pd.DataFrame(data=list_of_lists, columns=['SMILES', 'HAS_5MEMBERED_HETEROCYCLE_OR_QUINOLINE'])
    data_df.set_index('SMILES', inplace=True)
    _df.set_index('SMILES', inplace=True)
    _df = pd.concat([df, data_df], axis=1)

    _failed = _df[_df['HAS_5MEMBERED_HETEROCYCLE_OR_QUINOLINE']].copy(deep=True)
    _df = _df[~_df['HAS_5MEMBERED_HETEROCYCLE_OR_QUINOLINE']].copy(deep=True)
    print(f'{_failed.shape[0]} molecules were removed based on HAS_5MEMBERED_HETEROCYCLE_OR_QUINOLINE. {_df.shape[0]} remain.')

    return _df, _failed

# canonicalize some filters
#smiles = [
#    'C12=C(C=CS2)C=CC=N1',
#    'C34=C(C=CS4)C=CN=C3',
#    'C56=C(C=CS6)C=NC=C5',
#    'C78=C(C=CS8)N=CC=C7',
#    'C9=CSC=N9',
#    'N%10=CN=CO%10',
#    'C%11=CN=CO%11'
#]

#for s in smiles:
#    print(canonicalize_smiles(s))


#smiles = 'C12=C(C=CN2)C=CC=C1'
#_smiles_has_quinoline_or_5_membered_n_heterocycle(smiles)
#
#mol = get_mol(smiles)
#matches = flatten(mol.GetSubstructMatches(Chem.MolFromSmarts(PYRADINONE)))
#Draw.ShowMol(mol, highlightAtoms=matches)
#exit()

if __name__ == "__main__":
    # First we extracted the smiles from the sdf
    #extract_smiles_from_sdf()

    # Then we filter pased on logp
    #one_filter_log_p()

    # Then we filter based on other lipinski rules
    #df = two_filter_other_lipinski()
    #df.to_csv('./results/2-filter-lipinski.csv', index=False)

    #df = pd.read_csv('./results/2-filter-lipinski.csv', header=0)
    #df, failed = filter_pyirdines_without_hydrogen(df)
    #print(f'There are {df.shape[0]} pyridines left.')
    #print(f'{failed.shape[0]} pyridines were removed.')
    #df.to_csv('./results/3-filtered-pyrh.csv')
    #failed.to_csv('./results/3.1-failed-filtered-pyrh.csv')

    #df = pd.read_csv('./results/3-filtered-pyrh.csv', header=0)
    #df, failed = filter_multiple_pyridines(df)
    #print(f'There are {df.shape[0]} pyridines left.')
    #print(f'{failed.shape[0]} pyridines were removed.')
    #df.to_csv('./results/4-filtered-multiple-pyr.csv')
    #failed.to_csv('./results/4.1-failed-multiple-pyr.csv')

    #df = pd.read_csv('./results/4-filtered-multiple-pyr.csv')
    #failed = pd.read_csv('./results/4.1-failed-multiple-pyr.csv')
    #draw_molecules_to_grid_image(failed, Path('./figures/failed-multiple-pyridines.png'))

    # Get the df where the smiles does not have fragments
    #df['HAS_FRAGMENTS'] = df['SMILES'].apply(_smiles_has_fragments)
    #failed = df[df['HAS_FRAGMENTS']]
    #df = df[~df['HAS_FRAGMENTS']]
    #df.to_csv('./results/5-filtered-fragments.csv', index=False)
    #failed.to_csv('./results/5.1-failed-has-fragments.csv', index=False)

    #df = pd.read_csv('./results/5-filtered-fragments.csv', header=0)

    #df, failed = filter_functional_groups(df)

    #df, failed = filter_three_coordinate_nitrogen(df)
    #df.to_csv('./6-filtered-bad-nitrogen-atoms.csv')
    #failed.to_csv('./6.1-failed-bad-nitrogen-atoms.csv')
    #df = pd.read_csv('./results/6-filtered-bad-nitrogen-atoms.csv', header=0)
    #failed = pd.read_csv('./results/6.1-failed-bad-nitrogen-atoms.csv', header=0)
    ##draw_molecules_to_grid_image(failed, './figures/filtered_bad_nitrogen.png')
    #df['MOL_WT'] = df['SMILES'].apply(get_mol_wt)
    #df.set_index('SMILES', inplace=True)
    #df.to_csv('./7-added-mol-wt.csv')
    #df = pd.read_csv('./results/7-added-mol-wt.csv')

    #df, failed = filter_has_sulfonyl_halide(df)
    #failed.to_csv('./results/8.1-failed-sulfonyl-halide.csv')
    #df.to_csv('./results/8-filtered-sulfonyl-halide.csv')

    #df = pd.read_csv('./results/8-filtered-sulfonyl-halide.csv')
    #failed = pd.read_csv('./results/8.1-failed-sulfonyl-halide.csv')
    #draw_molecules_to_grid_image(failed, './figures/has_sulfonyl_halides.png')

    #df, failed = filter_has_acid_halides(df)
    #failed.to_csv('./results/9.1-failed-acid-halide.csv')
    #df.to_csv('./results/9-filtered-acid-halide.csv')

    #df = pd.read_csv('./results/9-filtered-acid-halide.csv')
    #failed = pd.read_csv('./results/9.1-failed-acid-halide.csv')
    ###draw_molecules_to_grid_image(failed, './figures/has_acid_halides.png')

    #df, failed = filter_has_pyradinone(df)
    #failed.to_csv('./results/10.1-failed-pyradinone.csv')
    #df.to_csv('./results/10-filtered-pyradinone.csv')

    #failed = pd.read_csv('./results/10.1-failed-pyradinone.csv')
    #df = pd.read_csv('./results/10-filtered-pyradinone.csv')

    #df = pd.read_csv('./results/11-added-vendors.csv')
    #print(df.shape)
    #print(df[df['MOL_WT'] <= 250].shape)
    #df = df[df['MOL_WT'] <= 200]
    #df.to_csv('./results/200mw-library.csv', index=False)

    #draw_molecules_to_grid_image(df, './figures/zz-200mw-library.png')

    #df = pd.read_csv('./results/200mw-library.csv', header=0)
    #xtb = pd.read_csv('./results/pyridine_props_xtb.csv', header=0)
    #df = df.merge(xtb, left_on='INCHI_KEY', right_on='Name')
    #for i, row in df.iterrows():
    #    assert row['Name'] == row['INCHI_KEY']
    #df.to_csv('./results/12-added-xtb-props.csv', index=False)

    #df = pd.read_csv('./results/12-added-xtb-props.csv', header=0)
    #df, failed = filter_bad_heterocycles(df)
    #failed.to_csv('./results/13-failed-bad-heterocycles.csv')
    #df.to_csv('./results/13-filtered-bad-heterocycles.csv')

    # Read in the previous dataframe
    #df = pd.read_csv('./results/13-filtered-bad-heterocycles.csv', header=0)
    #df.drop(columns=['HAS_PYRADINONE'], inplace=True)
    #df, failed = filter_has_pyradinone(df)
    #failed.to_csv('./results/14.1-failed-four-pyradinone.csv')
    #df.to_csv('./results/14-filtered-four-pyradinone.csv')


    # Add MORFEUS steric properties
    df = pd.read_csv('./results/14-filtered-four-pyradinone.csv')
    prop_df = pd.read_csv('/Users/jameshoward/Documents/Programming/XTBPropertyCalculator/data/pyridines_morfeus_parameters.csv')
    print(df.shape)
    print(prop_df.shape)
    df = df.merge(prop_df, left_on='INCHI_KEY', right_on='INCHI_KEY')
    df.to_csv('./results/15-added-morfeus-props.csv', index=False)
    print(df.shape)

    #draw_molecules_to_grid_image(failed, './figures/filtered-quinolines-and-5-membered-heterocycles.png')




