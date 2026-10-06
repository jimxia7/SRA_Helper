from typing import Literal, overload

import numpy as np
import numpy.typing as npt

@overload
def stair_step_Vs_profile(Thickness: npt.ArrayLike,
                          Vs: npt.ArrayLike,
                          starting_depth: float = 0,
                          Depth_columns: Literal[1] = 1) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def stair_step_Vs_profile(Thickness: npt.ArrayLike,
                          Vs: npt.ArrayLike,
                          starting_depth: float = 0,
                          *,
                          Depth_columns: Literal[2]) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def stair_step_Vs_profile(Thickness: npt.ArrayLike,
                          Vs: npt.ArrayLike,
                          starting_depth: float,
                          Depth_columns: Literal[2]) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def stair_step_Vs_profile(Thickness: npt.ArrayLike,
                          Vs: npt.ArrayLike,
                          starting_depth: float = 0,
                          Depth_columns: int = 1):

    """
    Converts a layered soil profile into a stair-step profile for plotting.

    Each layer is represented as a horizontal step, with the same Vs value
    repeated at the top and bottom of that layer.

    Parameters
    ----------
    Thickness : array_like
        Thickness of each soil layer (m). Shape: (n,)
    Vs : array_like
        Shear wave velocity of each soil layer (m/s). Shape: (n,)
    starting_depth : float, optional
        Depth of the top of the first layer (m). Default: 0
    Depth_columns : int, optional
        Layout of the returned depths. Default: 1

        1 : flat stair-step array, top and bottom interleaved, for plotting.
        2 : separate arrays of layer tops and bottoms.

    Returns
    -------
    If Depth_columns == 1 (two arrays):
        Depth : np.ndarray
            Stair-step depths (m), top and bottom interleaved. Shape: (2n,)
        New_Vs : np.ndarray
            Vs paired with each depth (m/s). Shape: (2n,)
    If Depth_columns == 2 (three arrays):
        tops : np.ndarray
            Depth to the top of each layer (m). Shape: (n,)
        bottoms : np.ndarray
            Depth to the bottom of each layer (m). Shape: (n,)
        Vs : np.ndarray
            Vs of each layer (m/s). Shape: (n,)
    """
    
    Thickness = np.asarray(Thickness, dtype=float)
    Vs = np.asarray(Vs)

    if len(Thickness) != len(Vs):
        raise ValueError("Thickness and Vs arrays must have the same length.")
    if Depth_columns not in (1, 2):
        raise ValueError("Depth_columns must be 1 or 2.")

    bottoms = starting_depth + np.cumsum(Thickness)
    tops = np.concatenate([[starting_depth], bottoms[:-1]])

    if Depth_columns == 2:
        return np.asarray(tops), np.asarray(bottoms), np.asarray(Vs)

    Depth = np.stack([tops, bottoms], axis=1).ravel()
    New_Vs = np.repeat(Vs, 2)

    return Depth, New_Vs

def point_vs_to_thickness(depth,vs,starting_depth = 0,ending_depth = False):
    new_thickness = []
    new_vs = []
    max_depth_index = len(depth)-1

    for i,d in enumerate(depth):
        if d == starting_depth and i == 0:
            new_thickness.append((depth[i+1]-d)/2)
            new_vs.append(vs[i])
        elif i == 0:
            new_thickness.append(d-starting_depth)
            new_vs.append(vs[i])
        elif i == max_depth_index:
            if ending_depth:
                new_thickness.append(
                    (d-depth[i-1])/2
                    +
                    (ending_depth-d))
                new_vs.append(vs[i])
            else:
                new_thickness.append((d-depth[i-1])/2)
                new_vs.append(vs[i])
        else:
            new_thickness.append(
                (depth[i+1]-d)/2
                +
                (d-depth[i-1])/2
                )
            new_vs.append(vs[i])
        

    return np.array(new_thickness), np.array(new_vs)

def calculate_vs30(depth,vs):

    mask = (depth <= 30)
    depth_masked = depth[mask]
    vs_masked = vs[mask]

    thickness,new_vs = point_vs_to_thickness(depth_masked,vs_masked,0,30)
    
    time = thickness/new_vs

    time_sum = time.sum()

    return 30/time_sum


    