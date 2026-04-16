import glob
import os
from os.path import join

import torch.utils.data as data
from PIL import Image

from .data_loader import DatasetParams, register_data_params, register_dataset_obj


class CycleGANDataset(data.Dataset):

    def __init__(self, root, regexp, transform=None, target_transform=None, 
            download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.image_paths, self.labels = self.find_images(regexp)

    @staticmethod
    def _parse_filename_label(path):
        """Parse label from filename when encoded as numeric prefix.

        Supported formats:
        - <label>_xxx.png
        - frame_<label>_xxx.png
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split('_')
        if parts and parts[0].isdigit():
            return int(parts[0])
        if len(parts) > 1 and parts[0] == 'frame' and parts[1].isdigit():
            return int(parts[1])
        return None

    def find_images(self, regexp='*.png'):
        basenames = sorted(glob.glob(join(self.root, '**', regexp), recursive=True))
        image_paths = []
        labels = []
        unresolved = []

        # First pass: parse labels from filenames when available.
        for path in basenames:
            label = self._parse_filename_label(path)
            if label is not None:
                image_paths.append(path)
                labels.append(label)
            else:
                unresolved.append(path)

        if not unresolved:
            return image_paths, labels

        # Second pass: parse labels from parent folder names.
        class_name_to_id = {}
        class_names = sorted({os.path.basename(os.path.dirname(path)) for path in unresolved})
        next_class_id = 0
        for class_name in class_names:
            if class_name.startswith('class_') and class_name[len('class_'):].isdigit():
                class_name_to_id[class_name] = int(class_name[len('class_'):])
            elif class_name.isdigit():
                class_name_to_id[class_name] = int(class_name)
            else:
                class_name_to_id[class_name] = next_class_id
                next_class_id += 1

        for path in unresolved:
            class_name = os.path.basename(os.path.dirname(path))
            image_paths.append(path)
            labels.append(class_name_to_id[class_name])

        return image_paths, labels

    def __getitem__(self, index):
        im = Image.open(self.image_paths[index]) #.convert('L')
        target = self.labels[index]

        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        return len(self.image_paths)


class Office31DomainDataset(data.Dataset):
    """Load Office-31 style domains stored as class subfolders."""

    IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')

    @staticmethod
    def _parse_dir_label(class_name):
        if class_name.startswith('class_') and class_name[len('class_'):].isdigit():
            return int(class_name[len('class_'):])
        if class_name.isdigit():
            return int(class_name)
        return None

    def __init__(self, root, train=True, transform=None, target_transform=None,
            download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        split_root = root
        if train and os.path.isdir(join(root, 'train')):
            split_root = join(root, 'train')
        elif (not train) and os.path.isdir(join(root, 'test')):
            split_root = join(root, 'test')

        self.image_paths, self.labels = self.find_images(split_root)

    def find_images(self, root_dir):
        image_paths = []
        labels = []

        class_dirs = [d for d in sorted(os.listdir(root_dir))
                if os.path.isdir(join(root_dir, d))] if os.path.isdir(root_dir) else []

        # Preferred layout: root/class_x/*.jpg
        if class_dirs:
            raw_class_labels = []
            for class_name in class_dirs:
                parsed = self._parse_dir_label(class_name)
                raw_class_labels.append(parsed if parsed is not None else class_name)

            # Always map to contiguous [0, C-1] to match CE loss expectations.
            class_map = {
                raw_label: idx for idx, raw_label in enumerate(sorted(set(raw_class_labels), key=lambda x: str(x)))
            }

            for class_name, raw_label in zip(class_dirs, raw_class_labels):
                class_idx = class_map[raw_label]
                class_root = join(root_dir, class_name)
                for ext in self.IMG_EXTS:
                    for path in sorted(glob.glob(join(class_root, ext))):
                        image_paths.append(path)
                        labels.append(class_idx)
            return image_paths, labels

        # Fallback layout: flat folder with label-prefixed filenames.
        for ext in self.IMG_EXTS:
            for path in sorted(glob.glob(join(root_dir, ext))):
                name = os.path.basename(path)
                try:
                    label = int(name.split('_')[0])
                except Exception:
                    continue
                image_paths.append(path)
                labels.append(label)
        return image_paths, labels

    def __getitem__(self, index):
        im = Image.open(self.image_paths[index]).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return im, target

    def __len__(self):
        return len(self.image_paths)


@register_dataset_obj('svhn2mnist')
class Svhn2MNIST(CycleGANDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
            download=False):
        if not train:
            print('No test set for svhn2mnist.')
            self.image_paths = []
        else:
            super(Svhn2MNIST, self).__init__(root, '*_fake_B.png',
                    transform=transform, target_transform=target_transform, 
                    download=download)

@register_data_params('svhn2mnist')
class Svhn2MNISTParams(DatasetParams):
    num_channels = 3
    image_size = 32
    mean = 0.5
    std = 0.5
    #mean = 0.1307
    #std = 0.3081
    
    # mean and std (when scaled between [0,1])
    #mean = 0.127 # ep50
    #mean = 0.21 # ep100 -- more white pixels...
    #std = 0.29

    #mean = 0.21
    #std = 0.2
    
    num_cls = 10
    target_transform = None

@register_dataset_obj('usps2mnist')
class Usps2Mnist(CycleGANDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
            download=False):
        if not train:
            print('No test set for usps2mnist.')
            self.image_paths = []
        else:
            super(Usps2Mnist, self).__init__(root, '*_fake_A.png',
                    transform=transform, target_transform=target_transform, 
                    download=download)

@register_data_params('usps2mnist')
class Usps2MnistParams(DatasetParams):
    num_channels = 3
    image_size = 16
    #mean = 0.1307
    #std = 0.3081
    mean = 0.5
    std = 0.5
    num_cls = 10
    target_transform = None


@register_dataset_obj('mnist2usps')
class Mnist2Usps(CycleGANDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
            download=False):
        if not train:
            print('No test set for mnist2usps.')
            self.image_paths = []
        else:
            super(Mnist2Usps, self).__init__(root, '*_fake_B.png',
                    transform=transform, target_transform=target_transform, 
                    download=download)

@register_data_params('mnist2usps')
class Mnist2UspsParams(DatasetParams):
    num_channels = 3
    image_size = 16 # this seems wrong...
    #mean = 0.25
    #std = 0.37
    
    #mean = 0.1307
    #std = 0.3081
    mean = 0.5
    std = 0.5
    num_cls = 10
    target_transform = None


@register_dataset_obj('amazon2webcam')
class Amazon2Webcam(CycleGANDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
            download=False):
        if not train:
            print('No test set for amazon2webcam.')
            self.image_paths = []
            self.labels = []
        else:
            super(Amazon2Webcam, self).__init__(root, '*_fake_B.png',
                    transform=transform, target_transform=target_transform,
                    download=download)


@register_data_params('amazon2webcam')
class Amazon2WebcamParams(DatasetParams):
    num_channels = 3
    image_size = 32
    mean = 0.5
    std = 0.5
    num_cls = 31
    target_transform = None


@register_dataset_obj('webcam')
class Webcam(Office31DomainDataset):
    pass


@register_data_params('webcam')
class WebcamParams(DatasetParams):
    num_channels = 3
    image_size = 32
    mean = 0.5
    std = 0.5
    num_cls = 31
    target_transform = None


@register_dataset_obj('amazon')
class Amazon(Office31DomainDataset):
    pass


@register_data_params('amazon')
class AmazonParams(DatasetParams):
    num_channels = 3
    image_size = 32
    mean = 0.5
    std = 0.5
    num_cls = 31
    target_transform = None
