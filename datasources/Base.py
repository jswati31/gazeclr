import logging
import os
import pickle
from typing import List
import cv2 as cv
import h5py
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from datasources.common import stimulus_type_from_folder_name

logger = logging.getLogger(__name__)

source_to_fps = {
    'screen': 30,
    'basler': 60,
    'webcam_l': 30,
    'webcam_c': 30,
    'webcam_r': 30,
}

source_to_interval_ms = dict([
    (source, 1e3 / fps) for source, fps in source_to_fps.items()
])

sequence_segmentations = None
cache_pkl_path = './eve_segmentation_cache.pkl'


class BaseDataSource(Dataset):

    def __init__(self,
                 dataset_path: str,
                 config,
                 participants_to_use: List[str] = None,
                 cameras_to_use: List[str] = None,
                 types_of_stimuli: List[str] = None,
                 stimulus_name_includes: str = ''):

        if types_of_stimuli is None:
            types_of_stimuli = ['image', 'video', 'wikipedia']
        if cameras_to_use is None:
            cameras_to_use = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
        assert ('points' not in types_of_stimuli)  # NOTE: deal with this in another way

        self.config = config

        self.path = dataset_path
        self.types_of_stimuli = types_of_stimuli
        self.stimulus_name_includes = stimulus_name_includes
        self.participants_to_use = participants_to_use
        self.cameras_to_use = cameras_to_use
        self.max_seq_len = config.max_sequence_len

        self.participant_to_id = {participants_to_use[i]: i for i in range(len(participants_to_use))}

        # Some sanity checks
        assert (len(self.participants_to_use) > 0)
        assert (30 > config.assumed_frame_rate)
        assert (30 % config.assumed_frame_rate == 0)

        # Load or calculate sequence segmentations (start/end indices)
        global cache_pkl_path, sequence_segmentations
        cache_pkl_path = (
                './segmentation_cache/%dHz_seqlen%d.pkl' % (
            config.assumed_frame_rate, config.max_sequence_len,
        )
        )
        if sequence_segmentations is None:
            if not os.path.isfile(cache_pkl_path):
                self.build_segmentation_cache()
                assert (os.path.isfile(cache_pkl_path))

            with open(cache_pkl_path, 'rb') as f:
                sequence_segmentations = pickle.load(f)

        # Register entries
        self.select_sequences()
        logger.info('Initialized dataset class for: %s' % self.path)

    def build_segmentation_cache(self):
        """Create support data structure for knowing how to segment (cut up) time sequences."""
        all_folders = sorted([
            d for d in os.listdir(self.path) if os.path.isdir(self.path + '/' + d)
        ])

        output_to_cache = {}
        for folder_name in all_folders:
            participant_path = '%s/%s' % (self.path, folder_name)
            assert (os.path.isdir(participant_path))
            output_to_cache[folder_name] = {}

            subfolders = sorted([
                p for p in os.listdir(participant_path)
                if os.path.isdir(os.path.join(participant_path, p))
                   and p.split('/')[-1].startswith('step')
                   and 'eye_tracker_calibration' not in p
            ])
            for subfolder in subfolders:
                subfolder_path = '%s/%s' % (participant_path, subfolder)
                output_to_cache[folder_name][subfolder] = {}

                # NOTE: We assume that the videos are synchronized and have the same length in time.
                #       This should be the case for the publicly released EVE dataset.
                for source in ('screen', 'basler', 'webcam_l', 'webcam_c', 'webcam_r'):
                    current_outputs = []
                    source_path_pre = '%s/%s' % (subfolder_path, source)
                    available_indices = np.loadtxt('%s.timestamps.txt' % source_path_pre)
                    num_available_indices = len(available_indices)

                    # Determine desired length and skips
                    fps = source_to_fps[source]
                    target_len_in_s = self.config.max_sequence_len / self.config.assumed_frame_rate
                    num_original_indices_in_sequence = fps * target_len_in_s
                    assert (num_original_indices_in_sequence.is_integer())
                    num_original_indices_in_sequence = int(
                        num_original_indices_in_sequence
                    )
                    index_interval = int(fps / self.config.assumed_frame_rate)
                    start_index = 0
                    while start_index < num_available_indices:
                        end_index = min(
                            start_index + num_original_indices_in_sequence,
                            num_available_indices
                        )
                        picked_indices = list(range(start_index, end_index, index_interval))
                        current_outputs.append(picked_indices)

                        # Move along sequence
                        start_index += num_original_indices_in_sequence

                    # Store back indices
                    if len(current_outputs) > 0:
                        output_to_cache[folder_name][subfolder][source] = current_outputs
                        # print('%s: %d' % (source_path_pre, len(current_outputs)))

        # Do the caching
        with open(cache_pkl_path, 'wb') as f:
            pickle.dump(output_to_cache, f)

        logger.info('> Stored indices of sequences to: %s' % cache_pkl_path)

    def select_sequences(self):
        """Select sequences (start/end indices) for the selected participants/cameras/stimuli."""
        self.all_subfolders = {}
        self.meta_data = []
        self.index_to_id = defaultdict(list)
        count = 0
        for participant_name, participant_data in sequence_segmentations.items():
            if participant_name not in self.participants_to_use:
                continue

            self.all_subfolders[participant_name] = {}

            for stimulus_name, stimulus_segments in participant_data.items():
                current_stimulus_type = stimulus_type_from_folder_name(stimulus_name)
                if current_stimulus_type not in self.types_of_stimuli:
                    continue
                if len(self.stimulus_name_includes) > 0:
                    if self.stimulus_name_includes not in stimulus_name:
                        continue

                self.all_subfolders[participant_name][stimulus_name] = {}

                skip_index = []
                # removing non-valid samples
                for camera, all_indices in stimulus_segments.items():
                    if camera not in self.cameras_to_use:
                        continue

                    hdf = h5py.File('%s/%s/%s/%s.h5' % (self.path, participant_name, stimulus_name, camera), 'r')
                    selected_validities = np.array(hdf['face_g_tobii']['validity'][()])

                    for i, indices in enumerate(all_indices):
                        if i not in skip_index and not all(selected_validities[indices]):
                            skip_index += [i]

                for camera, all_indices in stimulus_segments.items():
                    if camera not in self.cameras_to_use:
                        continue
                    self.all_subfolders[participant_name][stimulus_name][camera] = {}
                    for i, indices in enumerate(all_indices):
                        if i not in skip_index:
                            self.all_subfolders[participant_name][stimulus_name][camera][i] = indices

                            self.meta_data.append({
                                'camera_name': camera,
                                'participant': participant_name,
                                'subfolder': stimulus_name,
                                'partial_path': '%s/%s' % (participant_name, stimulus_name),
                                'index': i
                            })
                            self.index_to_id[participant_name] += [count]
                            count += 1

    def __len__(self):
        return len(self.meta_data)

    def preprocess_frames(self, frames):
        frames = np.array(frames)
        if len(frames.shape) == 3:
            # Expected input:  H x W x C
            # Expected output: C x H x W
            frames = np.transpose(frames, [2, 0, 1])
        elif len(frames.shape) == 4:
            # Expected input:  N x H x W x C
            # Expected output: N x C x H x W
            frames = np.transpose(frames, [0, 3, 1, 2])

        frames = frames.astype(np.float32)
        frames *= 1.0 / 255.0
        return frames

    def load_image(self, path):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.asarray(img).astype(np.uint8)
        return img

    def get_path(self, participant, subolder, camera, timestamp):
        """Return path of image."""
        raise NotImplementedError('BaseDataSource::get_path not implemented.')

    def get_gaze_data(self, index, partial_path, cameras, key):
        gaze_info = {}
        required_keys = ['{}_g_tobii'.format(key), 'camera_transformation', '{}_R'.format(key)]
        # required_keys = ['camera_transformation', '{}_g_tobii'.format(key),  'camera_matrix',
        #                  'inv_camera_transformation', 'face_R', 'face_o', 'millimeters_per_pixel',
        #                  'pixels_per_millimeter']

        for k in required_keys:
            temp_data = []
            for cam in cameras:
                hdf = h5py.File('%s/%s/%s.h5' % (self.path, partial_path, cam), 'r')
                if isinstance(hdf[k], h5py.Group):
                    _data = np.array(hdf[k]['data'][index])
                else:
                    _data = np.array(hdf[k])
                temp_data.append(_data)
            temp_data = np.array(temp_data, dtype=np.float32)
            gaze_info[k] = temp_data
        return gaze_info

    def __getitem__(self, idx):
        """Return a item which reads an entry from disk or memory."""
        raise NotImplementedError('BaseDataSource::getitem not implemented.')
