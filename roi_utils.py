import numpy as np 
from matplotlib import pyplot as plt 

def fit_oval(points):
    center_x, center_y = np.mean(points, axis=0)
    height = np.max(points[:, 0]) - np.min(points[:, 0])    
    width = np.max(points[:, 1]) - np.min(points[:, 1])    

    return center_x, center_y, width, height


def create_oval_mask(array_shape, center, semi_major_axis, semi_minor_axis):
    y, x = np.ogrid[:array_shape[0], :array_shape[1]]
    distance_from_center = ((x - center[0]) / semi_major_axis)**2 + ((y - center[1]) / semi_minor_axis)**2
    mask = distance_from_center <= 1
    return mask


def extract_roi_values(video, tracks):
    values = []
    for frame in range(video.shape[0]):
        points = tracks[frame]

        x_center, y_center, height, width = fit_oval(points)
        mask = create_oval_mask(video.shape[1:], (x_center, y_center), width / 2, height / 2)
        roi = video[frame] * mask
        roi_average = np.sum(roi) / (roi != 0).sum()
        values.append(roi_average)
    return np.array(values)


def draw_roi_plot(frames, values, visibles, out):
    fig, ax = plt.subplots()
    for visible in [0, 1]:
        ax.scatter(frames[visibles == visible], values[visibles == visible], 
                   s=1, c='blue' if visible else 'red', 
                   label='Visible' if visible else 'Not Visible')
    ax.set_xlabel('Frames')
    ax.set_ylabel('RoI Avg')
    ax.set_title(f'Average RoI throughout the video')
    ax.legend()
    plt.savefig(out)
    plt.close()