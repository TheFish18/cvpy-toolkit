import os
from pathlib import Path
from ast import literal_eval
from typing import (List, Tuple, Dict, Iterator)

import cv2
import pandas as pd

from cvpy.imseg.immasks import CompositeMask


class CompositeUtils:
    @staticmethod
    def generate_from_directory(
            mask_directory: str | Path,
            comp_data: Dict[str, Dict[str, List[Tuple[int, int, int]]]],
            error='raise'
    ) -> Iterator[CompositeMask]:
        """
        Yields a generator of Composite Masks by reading masks in from directory
        Args:
            mask_directory:
            comp_data:
                {image_name:
                    {defect_type: [
                            (r, g, b), ...
                        ], ...
                    }, ...
                }
            error: {raise, continue}

        Yields:
            CompositeMasks
        """

        if isinstance(mask_directory, str):
            mask_directory = Path(mask_directory)

        mask_files = mask_directory.iterdir()

        for mask_file in mask_files:
            if not mask_file.is_file():
                continue

            mask = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)

            filename = mask_file.name
            mask_name = os.path.splitext(filename)[0]

            if filename in comp_data:
                comp_datum = comp_data[filename]
            elif mask_name in comp_data:
                comp_datum = comp_data[mask_name]
            else:
                if error == "raise":
                    raise KeyError(f"{filename} not in comp_data")
                elif error == "continue":
                    continue
                else:
                    raise ValueError("got unexpected value for 'error'")

            yield CompositeMask(mask, comp_datum)

    @staticmethod
    def comp_data_from_dfs(
            image_df: pd.DataFrame | str | Path,
            annot_df: pd.DataFrame | str | Path,
            image_id_col="image_id",
            image_name_col='image_name',
            defect_name_col="defect_name",
            composite_rgb_col="composite_rgb"
    ) -> Dict[str, Dict[str, List[Tuple[int, int, int]]]]:
        """
        labelbox2.pull_project creates two dataframes: image_df and annot_df. This method combines those dataframes
        and coerces
        Args:
            image_df: image dataframe or path to image dataframe: must contain image_id and image_name cols
            annot_df: annotation dataframe or path to annotation dataframe: must contain image_id, defect_name, and composite_rgb
            image_id_col: name of the image id column
            image_name_col: name of the image name column
            defect_name_col: name of the defect name column
            composite_rgb_col: name of the composite rgb column

        Returns:

        """
        if isinstance(image_df, str) or isinstance(image_df, Path):
            image_df = pd.read_csv(image_df)
        if isinstance(annot_df, str) or isinstance(annot_df, Path):
            annot_df = pd.read_csv(annot_df)

        df = image_df.merge(annot_df, how='outer', on=image_id_col)
        out_d = {}

        for image_name, group in df.groupby(image_name_col):
            img_cls_d = out_d.get(image_name, {})

            for cls_name, cls_group in group.groupby(defect_name_col):
                img_cls_d[cls_name] = [literal_eval(color) for color in cls_group[composite_rgb_col].tolist()]
            out_d[image_name] = img_cls_d

        return out_d
