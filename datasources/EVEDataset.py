
import cv2 as cv
import numpy as np
import torch
from datasources.Base import BaseDataSource
from datasources.common import predefined_splits


source_to_label = {
    'basler': 0,
    'webcam_l': 1,
    'webcam_c': 2,
    'webcam_r': 3,
}


class EVEDataset(BaseDataSource):

    def __init__(self,
                 dataset_path,
                 config,
                 transforms=None,
                 is_load_label=False,
                 num_positives: int = 0,
                 test: bool = False,
                 **kwargs,
                 ):
        
        super(EVEDataset, self).__init__(dataset_path, config, **kwargs)

        self.transforms = transforms
        self.is_load_label = is_load_label
        self.num_positives = num_positives
        self.test = test

    def preprocess_image(self, img):
        if self.config.camera_frame_type == "eyes":
            ew, eh = self.config.eyes_size
            img = cv.resize(img, (2 * ew, eh))
            img = img[:, ew:, :]
        else:
            ew, eh = self.config.face_size
            img = cv.resize(img, (ew, eh))

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = self.preprocess_frames(img)
        return img

    def get_path(self, participant, subolder, camera, timestamp):

        if self.config.camera_frame_type == 'eyes':
            return '%s/images_eyes/%s/%s/%s/%s.png' % (self.path, participant, subolder, camera, timestamp)
        else:
            return '%s/images_face/%s/%s/%s/%s.png' % (self.path, participant, subolder, camera, timestamp)

    def __getitem__(self, idx):

        entry = {}

        key_label = 'face' if self.config.camera_frame_type == 'face' else 'left'

        spec = self.meta_data[idx]
        participant = spec['participant']
        stimuli = spec['subfolder']
        camera = spec['camera_name']
        index = spec['index']
        timestamp = self.all_subfolders[participant][stimuli][camera][index][0]

        # Add meta data
        # entry['participant'] = self.participant_to_id[participant]
        # entry['subfolder'] = stimuli

        entry['img_a'] = self.preprocess_image(self.load_image(self.get_path(participant, stimuli, camera, timestamp)))
        # second augmented single-view learning sample
        entry['inv_a'] = self.preprocess_image(self.load_image(self.get_path(participant, stimuli, camera, timestamp)))

        view_labels = [source_to_label[camera]]

        if self.is_load_label:
            gaze_information_entry = self.get_gaze_data(timestamp, spec['partial_path'], [camera], key_label)
            for k, v in gaze_information_entry.items():
                entry[k] = v

        if self.num_positives > 0:
            positive_images = [entry['img_a']]
            invariant_positive_images = [entry['inv_a']]
            cameras_to_consider = np.random.choice([k for k in self.cameras_to_use if k != camera],
                                                   self.num_positives, replace=False)
            for cam in cameras_to_consider:
                view_labels += [source_to_label[cam]]
                pos_timestamp = self.all_subfolders[participant][stimuli][cam][index][0]
                positive_images.append(self.preprocess_image(self.load_image(
                    self.get_path(participant, stimuli, cam, pos_timestamp))))
                # second augmented single-view learning sample
                invariant_positive_images.append(self.preprocess_image(self.load_image(
                    self.get_path(participant, stimuli, cam, pos_timestamp))))

                if self.is_load_label:
                    gaze_information_entry = self.get_gaze_data(pos_timestamp, spec['partial_path'], [cam], key_label)
                    for k, v in gaze_information_entry.items():
                        entry[k] = np.concatenate((entry[k], v), axis=0)

            entry['img_a'] = np.stack(positive_images, axis=0)
            entry['inv_a'] = np.stack(invariant_positive_images, axis=0)

        if self.num_positives > 0:
            view_labels = np.array(view_labels, dtype=np.int)
            view_labels_sorted_index = np.argsort(view_labels)

            entry['view_labels'] = view_labels[view_labels_sorted_index]
            entry['img_a'] = entry['img_a'][view_labels_sorted_index]
            entry['inv_a'] = entry['inv_a'][view_labels_sorted_index]

        torch_entry = dict([
            (k, torch.from_numpy(a)) if isinstance(a, np.ndarray) else (k, a)
            for k, a in entry.items()
        ])

        return torch_entry


class EVEDatasetTrain(EVEDataset):
    def __init__(self, dataset_path: str, config, **kwargs):
        super(EVEDatasetTrain, self).__init__(
            dataset_path,
            config,
            participants_to_use=predefined_splits['train'],
            **kwargs,
        )


class EVEDatasetVal(EVEDataset):
    def __init__(self, dataset_path: str, config, **kwargs):
        super(EVEDatasetVal, self).__init__(
            dataset_path,
            config,
            participants_to_use=predefined_splits['train'][-1:],
            **kwargs,
        )


class EVEDatasetTest(EVEDataset):
    def __init__(self, dataset_path: str, config, **kwargs):
        super(EVEDatasetTest, self).__init__(
            dataset_path,
            config,
            participants_to_use=predefined_splits['val'],
            **kwargs,
        )













