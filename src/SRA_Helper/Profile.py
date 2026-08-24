import numpy as np

def stair_step_Vs_profile(Thickness: np.ndarray, 
                          Vs: np.ndarray,
                          starting_depth: float = 0,
                          Depth_columns: int = 1):
    
    """
    Converts a layered soil profile into a stair-step profile for plotting.

    Each layer is represented as a horizontal step, with the same Vs value
    repeated at the top and bottom of that layer.

    Parameters
    ----------
    Thickness : np.ndarray
        Thickness of each soil layer (m). Shape: (n,)
    Vs : np.ndarray
        Shear wave velocity of each soil layer (m/s). Shape: (n,)
    starting_depth : float, optional
        Depth of the top of the first layer (m). Default: 0
    Depth_columns : int, optional
        Layout of the returned depths. Default: 1

        1 : flat stair-step array, top and bottom interleaved, for plotting.
        2 : one row per layer with columns [top, bottom].

    Returns
    -------
    Depth : np.ndarray
        Depth values (m). Shape: (2n,) if Depth_columns == 1,
        (n, 2) with columns [top, bottom] if Depth_columns == 2.
    New_Vs : np.ndarray
        Vs values paired with each depth row (m/s). Shape: (2n,) if
        Depth_columns == 1, (n,) if Depth_columns == 2.
    """
    
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


    