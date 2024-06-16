from tapnet.utils import model_utils
import jax
import numpy as np
from tapnet.utils import transforms
from tapnet import tapir_model

checkpoint_path = 'tapnet/checkpoints/causal_bootstapir_checkpoint.npy'
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']
kwargs = dict(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
kwargs.update(dict(
    pyramid_level=1,
    extra_convs=True,
    softmax_temperature=10.0
))
tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)


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

def online_model_init(frames, query_points):
    """Initialize query features for the query points."""
    frames = model_utils.preprocess_frames(frames)
    feature_grids = tapir.get_feature_grids(frames, is_training=False)
    query_features = tapir.get_query_features(
        frames,
        is_training=False,
        query_points=query_points,
        feature_grids=feature_grids,
    )
    return query_features

online_model_init=jax.jit(online_model_init)

def online_model_predict(frames, query_features, causal_context):
    """Compute point tracks and occlusions given frames and query points."""
    frames = model_utils.preprocess_frames(frames)
    feature_grids = tapir.get_feature_grids(frames, is_training=False)
    trajectories = tapir.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=query_features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True,
    )
    causal_context = trajectories['causal_context']
    del trajectories['causal_context']
    # Take only the predictions for the final resolution.
    # For running on higher resolution, it's typically better to average across
    # resolutions.
    tracks = trajectories['tracks'][-1]
    occlusions = trajectories['occlusion'][-1]
    uncertainty = trajectories['expected_dist'][-1]
    visibles = model_utils.postprocess_occlusions(occlusions, uncertainty)
    return tracks, visibles, causal_context

online_model_predict=jax.jit(online_model_predict)