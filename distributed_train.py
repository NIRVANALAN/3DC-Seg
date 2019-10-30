import os
import time

import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torchsummary import summary

from Forward import forward
from inference_handlers.inference import infer
from network.RGMP import Encoder
from utils.Argparser import parse_args
# Constants
from utils.AverageMeter import AverageMeter
from utils.Constants import DAVIS_ROOT, network_models
from utils.Saver import load_weights, save_checkpoint
from utils.dataset import get_dataset
from utils.util import get_lr_schedulers, show_image_summary, get_model, init_torch_distributed

NUM_EPOCHS = 400
TRAIN_KITTI = False
MASK_CHANGE_THRESHOLD = 1000

BBOX_CROP = True
BEST_IOU=0

palette = Image.open(DAVIS_ROOT + '/Annotations/480p/bear/00000.png').getpalette()


def train(train_loader, model, criterion, optimizer, epoch, foo):
  global count
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()
  losses_extra = AverageMeter()
  ious_extra = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, input_dict in enumerate(train_loader):
    input, input_var, iou, loss, loss_image, masks_guidance, output, target, loss_extra = \
      forward(args, criterion, input_dict, ious, model, ious_extra=ious_extra)
    losses.update(loss.item(), input.size(0))
    losses_extra.update(loss_extra.item(), 1)
    foo.add_scalar("data/loss", loss, count)
    foo.add_scalar("data/iou", iou, count)
    if args.show_image_summary:
      if "proposals" in input_dict:
        foo.add_images("data/proposals", input_dict['proposals'][:, :, -1].repeat(1, 3, 1, 1), count)
      show_image_summary(count, foo, input_var[0:1], masks_guidance, target[0:1], output[0:1])

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    count = count + 1

    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Loss Extra {loss_extra.val:.4f}({loss_extra.avg:.4f})\t'
          'IOU {iou.val:.4f} ({iou.avg:.4f})\t'
          'IOU Extra {iou_extra.val:.4f} ({iou_extra.avg:.4f})\t'.format(
      epoch, i * args.bs, len(train_loader)*args.bs, batch_time=batch_time,
      data_time=data_time, loss=losses, iou=ious, loss_extra=losses_extra, iou_extra=ious_extra), flush=True)

  print('Finished Train Epoch {} Loss {losses.avg:.5f} Loss Extra {losses_extra.avg: .5f} IOU {iou.avg: .5f}'.format(epoch, losses=losses, losses_extra=losses_extra, iou=ious), flush=True)
  return losses.avg, ious.avg


def validate(dataset, model, criterion, epoch, foo):
  batch_time = AverageMeter()
  losses = AverageMeter()
  losses_extra = AverageMeter()
  ious = AverageMeter()
  ious_extra = AverageMeter()

  # switch to evaluate mode
  model.eval()

  end = time.time()
  print("Starting validation for epoch {}".format(epoch), flush=True)
  for seq in dataset.get_video_ids():
    dataset.set_video_id(seq)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.cuda.device_count(),
                                                                   rank=local_rank, shuffle=False)
    testloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler)
    ious_video = AverageMeter()
    for i, input_dict in enumerate(testloader):
      with torch.no_grad():
        # compute output
        input, input_var, iou, loss, loss_image, masks_guidance, output, target, loss_extra = forward(args, criterion, input_dict, ious,
                                                                                                      model, ious_extra=ious_extra)
        ious_video.update(np.mean(iou))
        losses.update(loss.item(), input.size(0))
        losses_extra.update(loss_extra.item(), 1)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.show_image_summary:
          show_image_summary(count, foo, input_var, masks_guidance, target, output)

        print('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.5f})\t'
              'Loss Extra {loss_extra.val:.4f} ({loss_extra.avg:.5f})\t'
              'IOU {iou.val:.4f} ({iou.avg:.5f})\t'
              'IOU Extra {iou_extra.val:.4f} ({iou_extra.avg:.5f})\t'.format(
          input_dict['info']['name'], i, len(testloader), batch_time=batch_time, loss=losses, iou=ious_video,
          loss_extra=losses_extra, iou_extra=ious_extra),
          flush=True)
    print('Sequence {0}\t IOU {iou.avg}'.format(input_dict['info']['name'], iou=ious_video), flush=True)
    ious.update(ious_video.avg)

    foo.add_scalar("data/losses-test", losses.avg, epoch)

  print('Finished Eval Epoch {} Loss {losses.avg:.5f} Losses Extra {losses_extra.avg} IOU {iou.avg: 5f}'
        .format(epoch, losses=losses, losses_extra=losses_extra, iou=ious), flush=True)

  return losses.avg, ious.avg


if __name__ == '__main__':
    args = parse_args()
    count = 0
    local_rank = 0
    device = None
    MODEL_DIR = os.path.join('saved_models', args.network_name)
    print("Arguments used: {}".format(args), flush=True)

    trainset, testset = get_dataset(args)
    #train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.data_sample) \
    #  if args.data_sample is not None else \
    #  torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    #shuffle = True if args.data_sample is None else False
    model = get_model(args, network_models)

    # model = FeatureAgg3dMergeTemporal()
    print("Using model: {}".format(model.__class__), flush=True)
    print(args)

    if torch.cuda.is_available():
      init_torch_distributed()
      # model = torch.nn.DataParallel(model)
      model.cuda()

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
    train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.data_sample) \
      if args.data_sample is not None else \
      torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    shuffle = True if args.data_sample is None else False

    trainloader = DataLoader(trainset, batch_size=args.bs, num_workers=args.num_workers,
                             sampler=train_sampler)

    print(summary(model, tuple((256,256)), batch_size=1))
    writer = SummaryWriter(log_dir="runs/" + args.network_name)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr)
    model, optimizer, start_epoch, best_iou_train, best_iou_eval, best_loss_train, best_loss_eval = \
      load_weights(model, optimizer, args, MODEL_DIR, scheduler=None)# params
    lr_schedulers = get_lr_schedulers(optimizer, args, start_epoch)

    params = []
    for key, value in dict(model.named_parameters()).items():
      if value.requires_grad:
        params += [{'params':[value],'lr':args.lr, 'weight_decay': 4e-5}]

    criterion = torch.nn.BCEWithLogitsLoss(reduce=False) if args.n_classes == 2 else \
      torch.nn.CrossEntropyLoss(reduce=False)
    print("Using {} criterion", criterion)
    # iters_per_epoch = len(Trainloader)
    # model.eval()

    if args.task == "train":
      # best_loss = best_loss_train
      # best_iou = best_iou_train
      if args.freeze_bn:
        encoders = [module for module in model.modules() if isinstance(module, Encoder)]
        for encoder in encoders:
          encoder.freeze_batchnorm()
      for epoch in range(start_epoch, args.num_epochs):
        loss_mean, iou_mean  = train(trainloader, model, criterion, optimizer, epoch, writer)
        for lr_scheduler in lr_schedulers:
          lr_scheduler.step(epoch)
        if iou_mean > best_iou_train or loss_mean < best_loss_train:
          if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
          best_iou_train = iou_mean if iou_mean > best_iou_train else best_iou_train
          best_loss_train = loss_mean if loss_mean < best_loss_train else best_loss_train
          save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best_train")
          save_checkpoint(epoch, iou_mean, loss_mean, model, optimizer, save_name, is_train=True,
                          scheduler=None)

        if (epoch + 1) % args.eval_epoch == 0:
          loss_mean, iou_mean = validate(testset, model, criterion, epoch, writer)
          if iou_mean > best_iou_eval:
            best_iou_eval = iou_mean
            save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best_eval")
            save_checkpoint(epoch, iou_mean, loss_mean, model, optimizer, save_name, is_train=False,
                            scheduler=lr_scheduler)
    elif args.task == 'eval':
      validate(testset, model, criterion, 1, writer)
    elif 'infer' in args.task:
      infer(args, testset, model, criterion, writer)
    else:
      raise ValueError("Unknown task {}".format(args.task))

