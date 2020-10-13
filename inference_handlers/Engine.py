import logging
import os
import pdb
import pickle

import numpy as np
from abc import abstractmethod
from torch.utils.data import dataset
from tqdm import trange, tqdm
from pathlib import Path

import torch
from PIL import Image
from scipy.misc import imresize
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.nn import functional as F
from imageio import imread, imwrite

from util import color_map
from utils.AverageMeter import AverageMeter
from utils.Constants import PRED_LOGITS, PRED_SEM_SEG
from utils.util import iou_fixed_torch


class BaseInferenceEngine():
    def __init__(self, cfg):
        self.cfg = cfg
        self.results_dir = os.path.join('results', cfg.NAME)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        log_file = os.path.join(self.results_dir, 'output.log')
        logging.basicConfig(filename=log_file, level=logging.INFO)

    def infer(self, dataset, model):
        pass


class DeepBlenderInferenceEngine(BaseInferenceEngine):
    def __init__(self, cfg) -> None:
        super(DeepBlenderInferenceEngine, self).__init__(cfg)  # ?

    def infer(self, dataset, model):  # ?
        fs = AverageMeter()
        # switch to evaluate mode
        model.eval()
        pred_for_eval = []
        gt_for_eval = []

        with torch.no_grad():
            for video in tqdm(dataset.get_video_ids()):
                # logging.info('processing video_id: {}. Total sequence number: {}'.format(seq, dataset.get_video_ids()))
                ious_per_video = AverageMeter()
                dataset.set_video_id(video)
                # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
                test_sampler = None
                dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, sampler=test_sampler,
                                        pin_memory=True)
                for iter, input_dict in enumerate((dataloader)):
                    if not self.cfg.INFERENCE.EXHAUSTIVE and (iter % (self.cfg.INPUT.TW - self.cfg.INFERENCE.CLIP_OVERLAP)) != 0:
                        continue

                    fg = input_dict["fg"].float().cuda()
                    bg = input_dict["bg"].float().cuda()
                    mask = input_dict['inpaint_mask'].cuda()
                    # get_dict = dict([(k, t.float().cuda())

                    target_dict = dict([(k, t.float().cuda())
                                        for k, t in input_dict['target'].items()])

                    model_pred = model(fg, bg)
                    model_pred, fg, bg = map(
                        convert_tensor_img, (model_pred, fg, bg))
                    # reconstruction = model_pred + fg + bg  # G(A)
                    reconstruction = model_pred * mask + fg+bg
                    # reconstruction = reconstruction.squeeze(0)
                    save_dir = Path(self.results_dir)
                    # import pdb
                    # pdb.set_trace()
                    import pdb
                    pdb.set_trace()
                    if iter > 100:
                        exit(0)


def save_samples(input_dict, model_pred, iter, save_dir):
    save_dir = Path(save_dir)
    fg, bg, image, mask = (input_dict[key].cuda() for key in (
        'fg', 'bg', 'images', 'inpaint_mask'))
    mask = mask[:, None, :, :]
    model_pred, fg, bg = map(convert_tensor_img, (model_pred, fg, bg))
    save_image(fg, save_dir /
               '{}_fg.png'.format(iter), normalize=True)
    save_image(bg,  save_dir / '{}_bg.png'.format(iter), normalize=True)
    save_image(image, save_dir / '{}_gt.png'.format(iter), normalize=True)
    save_image(model_pred * mask + fg + bg, save_dir /
               '{}.png'.format(iter), normalize=True)


def convert_tensor_img(tensor_img):
    return (tensor_img*0.5)+0.5


class SaliencyInferenceEngine(BaseInferenceEngine):
    def __init__(self, cfg):
        super(SaliencyInferenceEngine, self).__init__(cfg)

    def infer(self, dataset, model):
        fs = AverageMeter()
        maes = AverageMeter()
        ious = AverageMeter()
        # switch to evaluate mode
        model.eval()
        pred_for_eval = []
        gt_for_eval = []

        with torch.no_grad():
            for seq in tqdm(dataset.get_video_ids()):
                # logging.info('processing video_id: {}. Total sequence number: {}'.format(seq, dataset.get_video_ids()))
                ious_per_video = AverageMeter()
                dataset.set_video_id(seq)
                # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
                test_sampler = None
                dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, sampler=test_sampler,
                                        pin_memory=True)

                all_semantic_pred = {}
                all_targets = {}
                for iter, input_dict in enumerate((dataloader)):
                    if not self.cfg.INFERENCE.EXHAUSTIVE and (iter % (self.cfg.INPUT.TW - self.cfg.INFERENCE.CLIP_OVERLAP)) != 0:
                        continue

                    info = input_dict['info'][0]
                    input = input_dict["images"]
                    batch_size = input.shape[0]
                    target_dict = dict([(k, t.float().cuda())
                                        for k, t in input_dict['target'].items()])
                    input_var = input.float().cuda()

                    # compute output
                    pred = model(input_var)
                    # pred = format_pred(pred)

                    pred_mask = F.softmax(pred[0], dim=1)
                    clip_frames = info['support_indices'][0].data.cpu().numpy()

                    assert batch_size == 1
                    for i, f in enumerate(clip_frames):
                        if f in all_semantic_pred:
                            # all_semantic_pred[clip_frames] += [torch.argmax(pred_mask, dim=1).data.cpu().int()[0]]
                            all_semantic_pred[f] += [pred_mask[0,
                                                               :, i].data.cpu().float()]
                        else:
                            all_semantic_pred[f] = [
                                pred_mask[0, :, i].data.cpu().float()]
                            # Use binary masks
                            if 'gt_frames' not in info or f in info['gt_frames']:
                                all_targets[f] = (target_dict['mask'] != 0)[
                                    0, 0, i].data.cpu().float()

                masks = [torch.stack(pred).mean(
                    dim=0) for key, pred in all_semantic_pred.items() if key in all_targets]
                iou = iou_fixed_torch(torch.stack(masks).cuda(), torch.stack(
                    list(all_targets.values())).cuda())
                ious_per_video.update(iou, 1)
                ious.update(iou, 1)
                f, mae, pred_flattened, gt_flattened = self.save_results(
                    all_semantic_pred, all_targets, info)
                fs.update(f)
                maes.update(mae)
                pred_for_eval += [pred_flattened]
                gt_for_eval += [gt_flattened]
                logging.info(
                    'Sequence {}: F_max {}  MAE {} IOU {}'.format(input_dict['info'][0]['video'], f, mae, ious_per_video.avg))

        print("IOU: {}".format(iou))
        gt = np.hstack(gt_for_eval).flatten()
        p = np.hstack(pred_for_eval).flatten()
        precision, recall, _ = precision_recall_curve(gt, p)
        Fmax = 2 * (precision * recall) / (precision + recall)
        mae = np.mean(np.abs(p - gt))
        logging.info('Finished Inference F measure: {:.5f} MAE: {: 5f} IOU: {:5f}'
                     .format(np.max(Fmax), mae, ious.avg))

    def save_results(self, pred, targets, info):
        results_path = os.path.join(self.results_dir, info['video'][0])
        pred_for_eval = []
        # pred = pred.data.cpu().numpy().astype(np.uint8)
        (lh, uh), (lw, uw) = info['pad']
        for f in pred.keys():
            M = torch.argmax(torch.stack(pred[f]).mean(dim=0), dim=0)
            h, w = M.shape[-2:]
            M = M[lh[0]:h - uh[0], lw[0]:w - uw[0]]

            if f in targets:
                pred_for_eval += [torch.stack(pred[f]).mean(dim=0)
                                  [:, lh[0]:h - uh[0], lw[0]:w - uw[0]]]

            shape = info['shape']
            img_M = Image.fromarray(
                imresize(M.byte(), shape, interp='nearest'))
            img_M.putpalette(color_map().flatten().tolist())
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))
            if self.cfg.INFERENCE.SAVE_LOGITS:
                prob = torch.stack(pred[f]).mean(dim=0)[-1]
                pickle.dump(prob, open(os.path.join(
                    results_path, '{:05d}.pkl'.format(f)), 'wb'))

        assert len(targets.values()) == len(pred_for_eval)
        pred_for_F = torch.argmax(torch.stack(pred_for_eval), dim=1)
        pred_for_mae = torch.stack(pred_for_eval)[:, -1]
        gt = torch.stack(list(targets.values()))[
            :, lh[0]:h - uh[0], lw[0]:w - uw[0]]
        precision, recall, _ = precision_recall_curve(gt.data.cpu().numpy().flatten(),
                                                      pred_for_F.data.cpu().numpy().flatten())
        Fmax = 2 * (precision * recall) / (precision + recall)
        mae = (pred_for_mae - gt).abs().mean()

        return Fmax.max(), mae, pred_for_mae.data.cpu().numpy().flatten(), gt.data.cpu().numpy().flatten()
