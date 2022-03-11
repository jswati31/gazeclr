
# cite: original EVE repository

predefined_splits = {
    'train': ['train%02d' % i for i in range(1, 40)],
    'val': ['val%02d' % i for i in range(1, 6)],
    'test': ['test%02d' % i for i in range(1, 11)],
    'etc': ['etc%02d' % i for i in range(1, 3)],
}


def stimulus_type_from_folder_name(folder_name):
    parts = folder_name.split('_')
    if parts[1] in ('image', 'video', 'wikipedia'):
        return parts[1]
    elif parts[1] == 'eye':
        return 'points'
    raise ValueError('Given folder name unexpected: %s' % folder_name)

