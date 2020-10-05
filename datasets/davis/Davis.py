import glob
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms

from datasets.BaseDataset import VideoDataset, INFO, IMAGES_, TARGETS
from utils.Resize import ResizeMode
from .transform import AddSeams, AddSuperpixel, RemoveBoundary, GetFgBg


class Davis(VideoDataset):
    def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2,
                 imset=None, transform=transforms.Compose([GetFgBg(), AddSuperpixel(), RemoveBoundary(), AddSeams()])):
        self.imset = imset
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.raw_samples = []
        self.transform = transform
        super(Davis, self).__init__(root, mode, resize_mode,
                                    resize_shape, tw, max_temporal_gap, num_classes)

    def __getitem__(self, idx):
        """return warped fg(with noise), warped bg(with noise) and original image

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        sample = self.samples[idx]
        tensors_resized = self.read_sample(sample)

        if self.transform:   # 3 augmentations
            tensors_resized = self.transform(tensors_resized)

        padded_tensors = self.pad_tensors(tensors_resized)
        padded_tensors = self.normalise(padded_tensors)

        return {"fg": np.transpose(padded_tensors['fg'].squeeze(), (2, 0, 1)).astype(np.float32),  # 8*480*864*3 in video. 8 frames in a batch
                "bg": np.transpose(padded_tensors['bg'].squeeze(), (2, 0, 1)).astype(np.float32),
                "target": {"images": np.transpose(padded_tensors['images'].squeeze(), (2, 0, 1)).astype(np.float32)},
                'info': padded_tensors['info']}

    def filter_samples(self, video):
        filtered_samples = [
            s for s in self.raw_samples if s[INFO]['video'] == video]
        self.samples = filtered_samples

    def set_video_id(self, video):
        self.current_video = video
        self.start_index = self.get_start_index(video)
        self.filter_samples(video)

    def get_video_ids(self):
        # shuffle the list for training
        return random.sample(self.videos, len(self.videos)) if self.is_train() else self.videos

    def get_support_indices(self, index, sequence):
        # index should be start index of the clip
        if self.is_train():
            index_range = np.arange(index, min(self.num_frames[sequence],
                                               (index + max(self.max_temporal_gap, self.tw))))
        else:
            index_range = np.arange(index,
                                    min(self.num_frames[sequence], (index + self.tw)))

        support_indices = np.random.choice(index_range, min(
            self.tw, len(index_range)), replace=False)
        support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                       self.tw - len(support_indices))))

        # print(support_indices)
        return support_indices

    def create_sample_list(self):
        image_dir = os.path.join(self.root, 'JPEGImages', '480p')
        mask_dir = os.path.join(self.root, 'Annotations', '480p')
        if self.is_train():
            _imset_f = '2017/train.txt'
        elif self.imset:
            _imset_f = self.imset
        else:
            _imset_f = '2017/val.txt'

        with open(os.path.join(self.root, "ImageSets", _imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos += [_video]
                img_list = list(
                    glob.glob(os.path.join(image_dir, _video, '*.jpg')))
                img_list.sort()
                # self.videos.append(_video)

                num_frames = len(
                    glob.glob(os.path.join(image_dir, _video, '*.jpg')))
                self.num_frames[_video] = num_frames

                _mask_file = os.path.join(mask_dir, _video, '00000.png')
                _mask = np.array(Image.open(os.path.join(
                    mask_dir, _video, '00000.png')).convert("P"))
                num_objects = np.max(_mask)
                self.num_objects[_video] = num_objects
                self.shape[_video] = np.shape(_mask)

                for i, img in enumerate(img_list):
                    sample = {INFO: {}, IMAGES_: [], TARGETS: []}
                    support_indices = self.get_support_indices(i, _video)
                    sample[INFO]['support_indices'] = support_indices
                    images = [os.path.join(image_dir, _video, '{:05d}.jpg'.format(
                        s)) for s in np.sort(support_indices)]
                    targets = [os.path.join(mask_dir, _video, '{:05d}.png'.format(
                        s)) for s in np.sort(support_indices)]
                    sample[IMAGES_] = images
                    sample[TARGETS] = targets

                    sample[INFO]['video'] = _video
                    sample[INFO]['num_frames'] = num_frames
                    sample[INFO]['num_objects'] = num_objects
                    sample[INFO]['shape'] = np.shape(_mask)

                    self.samples += [sample]
        self.raw_samples = self.samples


if __name__ == '__main__':
    davis = Davis(root="/globalwork/data/DAVIS-Unsupervised/DAVIS/",
                  resize_shape=(480, 854), resize_mode=ResizeMode.FIXED_SIZE, mode="train", max_temporal_gap=32)

    # davis.set_video_id('cat-girl')
    print("Dataset size: {}".format(davis.__len__()))

    for i, _input in enumerate(davis):
        print(_input['info'])
        print("Image Max {}, Image Min {}".format(_input['images'].max(), _input['images'].min()),
              "Target max {}, Target Min {}".format(_input['target']['mask'].max(), _input['target']['mask'].min()))
