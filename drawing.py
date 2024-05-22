from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw

def draw_molecules_to_grid_image(dataframe: pd.DataFrame,
                                 save_path: Path,
                                 legend_columns: list[str] = None):
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

        legend = [s for s in list]
        if legend_columns is not None:
            for column in legend_columns:
                for i, string in enumerate(legend):
                    legend[i] = legend[i] + f'\n{dataframe.loc[dataframe["SMILES"] == string, column].value}'

        png = Draw.MolsToGridImage(mols,
                                molsPerRow=grid_width,
                                subImgSize=image_size,
                                highlightAtomLists=None,
                                highlightBondLists=None,
                                returnPNG=False,
                                legends=legend)

        png.save(save_path.parent / f'{save_path.name}_{i}.png')