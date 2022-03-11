
import os
import cv2
from decord import VideoReader
from decord import cpu
import pickle
from tqdm import tqdm


with open('segmentation_cache/10Hz_seqlen1.pkl', 'rb') as f:
    sequence_segmentations = pickle.load(f)

path = "/path/to/eve/dataset"

output_save_path = os.path.join(path, 'images_face')

predefined_splits = {
    'train': ['train%02d' % i for i in range(1, 40)],
    'val': ['val%02d' % i for i in range(1, 6)],
    'test': ['test%02d' % i for i in range(1, 11)],
    'etc': ['etc%02d' % i for i in range(1, 3)],
}

participants_to_use = predefined_splits['train'] + predefined_splits['val']
cameras_to_use = ["basler", "webcam_l", "webcam_c", "webcam_r"]
types_of_stimuli = ["image", "video", "wikipedia"]
stimulus_name_includes = ''


def stimulus_type_from_folder_name(folder_name):
    parts = folder_name.split('_')
    if parts[1] in ('image', 'video', 'wikipedia'):
        return parts[1]
    elif parts[1] == 'eye':
        return 'points'
    raise ValueError('Given folder name unexpected: %s' % folder_name)


def select_sequences():
    """Select sequences (start/end indices) for the selected participants/cameras/stimuli."""
    all_subfolders = []
    for participant_name, participant_data in sequence_segmentations.items():
        if participant_name not in participants_to_use:
            continue
        for stimulus_name, stimulus_segments in participant_data.items():
            current_stimulus_type = stimulus_type_from_folder_name(stimulus_name)
            if current_stimulus_type not in types_of_stimuli:
                continue
            if len(stimulus_name_includes) > 0:
                if stimulus_name_includes not in stimulus_name:
                    continue
            for camera, all_indices in stimulus_segments.items():
                if camera not in cameras_to_use:
                    continue

                for i, indices in enumerate(all_indices):
                    all_subfolders.append({
                        'camera_name': camera,
                        'participant': participant_name,
                        'subfolder': stimulus_name,
                        'partial_path': '%s/%s' % (participant_name, stimulus_name),
                        'full_path': '%s/%s/%s' % (path, participant_name, stimulus_name),
                        'indices': indices,
                        'screen_indices': stimulus_segments['screen'][i],
                    })
    return all_subfolders


def load_all_from_source(path, source, selected_indices, o_path):
    assert (source in ('basler', 'webcam_l', 'webcam_c', 'webcam_r', 'screen'))

    # Get frames
    video_path = '%s/%s_face' % (path, source)

    vr = VideoReader(video_path + '.mp4', ctx=cpu(0))
    frames = vr.get_batch(selected_indices)
    frames = frames.asnumpy()

    for k in range(len(selected_indices)):
        cv2.imwrite(os.path.join(o_path, str(selected_indices[k])+'.png'), frames[k][:,:,::-1])


all_subfolders = select_sequences()

total_num = len(all_subfolders)

for ind in tqdm(range(len(all_subfolders))):
    spec = all_subfolders[ind]
    path = spec['full_path']
    source = spec['camera_name']
    indices = spec['indices']
    screen_indices = spec['screen_indices']

    out_path = os.path.join(output_save_path, spec['partial_path'], source)

    os.makedirs(out_path, exist_ok=True)

    # Grab all data
    load_all_from_source(path, source, indices, out_path)

