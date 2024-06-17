import torch
import torch.nn.functional as F
from tapnet.torch import tapir_model
import numpy as np
from tapnet.utils import transforms

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

checkpoint_path = 'tapnet/checkpoints/bootstapir_checkpoint_v2.pt'
ckpt = torch.load(checkpoint_path)
model = tapir_model.TAPIR(pyramid_level=1)
model.load_state_dict(ckpt)
model = model.to(device)


def convert_points_to_query_points(frame, points, scale_actor, height, width, resize_height, resize_width):
    """
    Returns a numpy array that has exact shape as points but frames are also included
    It also resizes it to given height and width
    """
    points = np.stack(points) / scale_actor
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
    query_points[:, 0] = frame
    query_points[:, 1:] = points

    query_points = transforms.convert_grid_coordinates(
                                query_points, (1, height, width),
                                (1, resize_height, resize_width),
                                coordinate_format='tyx')
    return query_points


def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.float()
  frames = frames / 255 * 2 - 1
  return frames


def postprocess_occlusions(occlusions, expected_dist):
  visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
  return visibles


def inference(frames, query_points):
  # Preprocess video to match model inputs format
  frames = preprocess_frames(frames)
  num_frames, height, width = frames.shape[0:3]
  query_points = query_points.float()
  frames, query_points = frames[None], query_points[None]

  # Model inference
  with torch.no_grad():
    outputs = model(frames, query_points)
  tracks, occlusions, expected_dist = outputs['tracks'][0], outputs['occlusion'][0], outputs['expected_dist'][0]

  # Binarize occlusions
  visibles = postprocess_occlusions(occlusions, expected_dist)
  return tracks, visibles
