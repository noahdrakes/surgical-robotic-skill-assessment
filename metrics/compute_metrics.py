import numpy as np
import math

def compute_duration(data_frame, axis):
    df = np.array(data_frame)
    time = df[:,axis]

    return time[-1]


def compute_average_speed_magnitude(data_frame, axis_array):
    df = np.array(data_frame)
    
    spd_sqrd = 0

    for axis in axis_array:
        speed_sum = df[:, axis].sum(0)
        spd_sqrd += pow(speed_sum, 2)

    return math.sqrt(spd_sqrd)

dummy = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (1, 1, 1, 1, 1)]
# dummy = [(0,3)]

dt = np.array(dummy)
print(dt.sum(0))

print(compute_duration(dummy, 0))
print(compute_average_speed_magnitude(dummy,[0,1]))
