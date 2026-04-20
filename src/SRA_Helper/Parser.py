import re
import numpy as np

def read_V2_accel(filepath, return_dt=False):

    with open(filepath, "r") as f:
        lines = f.readlines()

    header = lines[45]

    # Number of points
    npts = int(header.split()[0])

    # Sampling interval
    dt_match = re.search(r'at\s+([-+]?\d*\.?\d+)\s+sec', header)
    dt = float(dt_match.group(1)) if dt_match else None

    # Unit
    unit_match = re.search(r'in\s+([^\s,]+)', header)
    unit = unit_match.group(1).rstrip('.') if unit_match else None

    # Extract format (e.g., 8f10.6)
    fmt_match = re.search(r'\((\d+)f(\d+)\.(\d+)\)', header)
    if not fmt_match:
        raise ValueError("Cannot detect fixed-width format from header.")

    n_per_line = int(fmt_match.group(1))
    width = int(fmt_match.group(2))

    # Calculate number of lines containing data
    n_lines = int(np.ceil(npts / n_per_line))

    selected_lines = lines[46:46 + n_lines]

    # Read fixed-width values
    data = []
    for line in selected_lines:
        for i in range(0, width * n_per_line, width):
            value_str = line[i:i+width].strip()
            if value_str:
                data.append(float(value_str))

    data = np.array(data[:npts])  # ensure exact length

    # Unit conversion
    if unit == "cm/sec2":
        data = data * 0.0010197162129779282
    elif unit == "g":
        pass

    if return_dt:
        return dt
    
    return data