import os
import numpy as np
import matlab
from typing import Any, List

# Get the directory of this file
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)


def brain_heatmap(
    eng: Any,
    subject: str,
    correlation: np.array,
    hemi: str,
    output_path: str,
    range: List[float] = [-0.15, 0.15],
) -> dict:
    """
    This function generates a brain heatmap using the provided parameters.

    Parameters:
    eng (Any): An instance of the eng module. This is the module used to generate the heatmap.
    subject (str): The subject for which the heatmap is to be generated. This is typically a string identifier for the subject.
    correlation (np.array): The correlation data to be used for the heatmap. This is a 2d numpy array containing the correlation values. The second dimension should be 1.
    hemi (str): The hemisphere ('left' or 'right') for which the heatmap is to be generated. This is a string indicating the hemisphere.
    output_path (str): The path where the output heatmap image will be saved. This is a string representing the file path.
    range (List[float], optional): The range of values for the heatmap. Defaults to [-0.15,0.15]. This is a list containing two float values representing the minimum and maximum values for the heatmap.

    Returns:
    result (dict): The result of the eng function execution. This is typically an object containing information about the generated heatmap.
    """
    eng.addpath(dir_path)
    result = eng.plot_circle_map2(
        subject,
        matlab.double(correlation.reshape(-1, 1)),
        hemi,
        output_path,
        "range",
        matlab.double(range),
    )
    return result
