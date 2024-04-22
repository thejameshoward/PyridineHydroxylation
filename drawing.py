from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw

def draw_molecules_to_grid_image(dataframe: pd.DataFrame,
                                 save_path: Path):
    '''
    Draws the molecules in dataframe['SMILES'] to a file
    in filtered_molecules
    '''
    if isinstance(save_path, str):
        save_path = Path(save_path)

    # Get just the smiles as a list
    smiles = dataframe['SMILES'].to_list()

    # Specify grid shape
    grid_height = 10
    grid_width = 10

    # Define mols per image
    mols_per_image = grid_width * grid_height

    # Specify resolution of the molecule images
    image_size = (800, 800)

    # Make a list of grid_width *
    list_of_lists = [smiles[i:i+mols_per_image] for i in range(0,len(smiles),mols_per_image)]

    for i, list in enumerate(list_of_lists):

        mols = [Chem.MolFromSmiles(s) for s in list]

        png = Draw.MolsToGridImage(mols,
                                molsPerRow=grid_width,
                                subImgSize=image_size,
                                highlightAtomLists=None,
                                highlightBondLists=None,
                                returnPNG=False,
                                legends=[f"SMILES: {s}" for s in list])

        png.save(save_path.parent / f'{save_path.name}_{i}.png')