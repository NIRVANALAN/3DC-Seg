from typing import List
from skimage import segmentation, color
import torch
from skimage.io import imread
from skimage.future import graph
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import pickle


def get_fg_bg(img, mask):
    import PIL
    import numpy as np
    if type(img) is not np.ndarray:
        img = np.asarray(img)
    if type(mask) is not np.ndarray:
        mask = np.asarray(mask)
    expand_mask = mask[:, :, np.newaxis]
    expand_mask[expand_mask > 0] = 1  # only care foreground
    fg = img * expand_mask
    bg = img - fg
    return fg, bg, expand_mask


class GetFgBg(object):
    """return fg and bg

    Args:
        object ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __call__(self, sample) -> List:
        images, masks = sample['images'], sample['targets']
        foregrounds = images.copy()
        backgrounds = images.copy()
        # if images.ndim == 4:
        #     for i in range(images.shape[0]):
        #         # implicit cast from Image to ndarray
        #         foregrounds[i], backgrounds[i], _ = get_fg_bg(
        #             images[i], masks[i])
        # else:
        foregrounds, backgrounds, _ = get_fg_bg(images, masks)
        return {'images': images, 'targets': masks, 'fg': foregrounds, 'bg': backgrounds, 'info': sample['info']}


class AddSeams(object):  # TODO
    """Add Seams noise into image
    """

    def __call__(self, sample):
        masks, foregrounds = sample['targets'], sample['fg']
        # if foregrounds.ndim == 4:
        #     for i in range(masks.shape[0]):
        #         # foregrounds[i] = add_seams(foregrounds[i], masks[i])
        #         sample['fg'][i] = add_seams(sample['fg'][i], masks[i])
        # else:
        sample['fg'] = add_seams(sample['fg'], masks)
        return sample


class GetMaskedParts(object):  # TODO
    def __call__(self, sample) -> dict:
        inpainted_area = sample['images'] - (sample['fg'] + sample['bg'])
        inpaint_mask = np.zeros_like(inpainted_area)
        inpaint_mask[inpainted_area > 0] = 1.
        sample['inpaint_mask'] = inpaint_mask
        # save_sample(sample)
        return sample


class AddSuperpixel(object):  # TODO
    """Add Holes 
    """

    def __call__(self, sample):
        # aug_images, masks, foregrounds = sample['aug_images'], sample['targets'], sample['fg']
        # for i in range(images.shape[0]):
        # implicit cast from Image to ndarray
        # if images.ndim == 4:
        #     for i in range(images.shape[0]):
        #         # implicit cast from Image to ndarray
        #         images[i] = add_holes(images[i], masks[i])
        # else:
        sample['fg'] = add_holes(sample['images'].copy(), sample['targets'])
        return sample


class RemoveBoundary(object):  # TODO
    """Randomly Remove Boundary
    """

    def __call__(self, sample):
        # images, masks, foregrounds, backgrounds, info = sample['images'], sample[
        #     'targets'], sample['fg'], sample['bg'], sample['info']

        # if sample['bg'].ndim == 4:  # video sequence
        #     for i in range(sample['bg'].shape[0]):
        #         # implicit cast from Image to ndarray
        #         sample['bg'] = remove_boundary(sample['bg'][i])
        # else:
        #     import pdb
        sample['bg'] = remove_boundary(sample['bg'])
        return sample


def add_seams(fg, mask, return_Image=False):
    # fg, bg, _ = get_fg_bg(img, mask)
    THICKNESS = 5
    try:
        x_indexes, y_indexes = np.where(mask)
        x_boundary = (x_indexes[0], x_indexes[-1])
        y_boundary = (y_indexes[0], y_indexes[-1])
        x_start = np.random.randint(
            x_indexes[0], max(x_indexes[-1]-THICKNESS, x_indexes[0]+1), 1)[0]
        y_start = np.random.randint(
            y_indexes[0], max(y_indexes[-1]-THICKNESS, y_indexes[0]+1), 1)[0]
        seam_mask = np.ones_like(mask)
        seam_mask[x_start:x_start+THICKNESS, y_boundary[0]:y_boundary[1]] = 0
        seam_mask[x_boundary[0]:x_boundary[1], y_start:y_start+THICKNESS] = 0
        seam_mask *= mask
        fg *= seam_mask[:, :, np.newaxis]  # TODO
    except:
        pass
    return fg


def add_holes(fg, mask, remove_number=10, return_Image=False):  # TODO, unstable
    # fg, bg, expand_mask = get_fg_bg(fg, bg)
    img_segments = segmentation.slic(fg, compactness=20, n_segments=500)
    img_segments *= mask
    segments_left = np.unique(img_segments[img_segments > 0])
    # print(segments_left)
    # seg_range = [np.min(img_segments[img_segments>0]), np.max(img_segments)]
    try:
        removed_sp = np.random.choice(segments_left, remove_number)
        for sp in removed_sp:
            fg[img_segments == sp] = 0
    except:
        pass
    # superpixels = color.label2rgb(img_segments, fg, kind='avg')
    return fg


def remove_boundary(img, return_Image=False):
    if type(img) is not np.ndarray:
        img = np.asarray(img)
    img = img.copy()
    THICKNESS = 30
    # ?
    side = np.random.randint(low=0, high=4, size=1)
    if side == 0:
        img[:THICKNESS, :] = 0
    elif side == 1:
        img[-THICKNESS:, :] = 0
    elif side == 2:
        img[:, -THICKNESS:] = 0
    else:
        img[:, :THICKNESS] = 0
    if return_Image:
        return Image.fromarray(img)
    return img


def save_sample(sample):
    def save_obj(obj, name):
        with open('results/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        with open('results/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def save_img():
        # exp = Image.fromarray(inpaint_mask.squeeze()*255)
        # exp.save('{}.png'.format(inpaint_mask.sum()))
        pass

    import pdb
    pdb.set_trace()
    save_obj(sample, 'sample')
