import math
import os
import signal
import time

import apex
from numpy.lib.utils import deprecate
# from inference_handlers.inference import infer
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config import get_cfg
from inference_handlers.infer_utils.util import get_inference_engine
from loss.loss_utils import compute_loss
# Constants
from utils.Argparser import parse_argsV2
from utils.AverageMeter import AverageMeter, AverageMeterDict
from utils.Saver import save_checkpointV2, load_weightsV2, save_checkpointV3, load_weightsV3
from utils.util import get_lr_schedulers, show_image_summary, get_model, cleanup_env, \
    reduce_tensor, is_main_process, synchronize, get_datasets, get_optimiser, init_torch_distributed, _find_free_port, \
    format_pred, iou_fixed_torch
from utils.Constants import PRED_LOGITS
from loss.loss_utils import *
from network import networks

NUM_EPOCHS = 400
TRAIN_KITTI = False
MASK_CHANGE_THRESHOLD = 1000

BBOX_CROP = True
BEST_IOU = 0
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, args, port):
        cfg = get_cfg()
        cfg.merge_from_file(args.config)
        self.device = torch.device('cuda') if torch.cuda.is_available(
        ) else torch.device('cpu')  # get device name: CPU or GPU
        self.cfg = cfg
        self.port = port
        assert os.path.exists(
            'saved_models'), "Create a path to save the trained models: <default: ./saved_models> "
        self.model_dir = os.path.join('saved_models', cfg.NAME)
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.model_dir, "summary"))
        self.iteration = 0
        self.start_epoch = 0
        # * dataset
        self.trainset, self.testset = get_datasets(cfg)
        # * model
        self.net_G = get_model(cfg).to(self.device)
        print("Using model: {}".format(self.net_G.__class__), flush=True)

        if args.task == 'train':
            self.loss_dict = {}
            self.net_D = networks.define_D(3, 64, 'basic',
                                           3, gpu_ids=[0]).to(self.device)
            self.criterionGAN = networks.GANLoss('vanilla').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                self.net_G.parameters(), lr=cfg.TRAINING.BASE_LR, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.net_D.parameters(), lr=cfg.TRAINING.BASE_LR, betas=(0.5, 0.999))
            self.net_G, self.net_D, self.optimizer_G, self.optimizer_D, self.start_epoch, self.iteration = load_weightsV3(  # TODO
                self.net_G, self.net_D, self.optimizer_G, self.optimizer_D, args.wts, self.model_dir)
            self.lr_sched_netG = get_lr_schedulers(
                self.optimizer_G, cfg, self.start_epoch)
            self.lr_sched_netD = get_lr_schedulers(
                self.optimizer_D, cfg, self.start_epoch)
            self.batch_size = self.cfg.TRAINING.BATCH_SIZE
        # self.model, self.optimiser, self.start_epoch, start_iter = \
        #   load_weightsV2(self.model, self.optimiser, args.wts, self.model_dir)

        self.world_size = 1  # single card training
        self.args = args
        self.epoch = 0
        self.best_loss_train = math.inf
        self.losses = AverageMeterDict()
        self.ious = AverageMeterDict()

        num_samples = None if cfg.DATALOADER.NUM_SAMPLES == - \
            1 else cfg.DATALOADER.NUM_SAMPLES
        # if torch.cuda.device_count() > 1:
        #     # shuffle parameter does not seem to shuffle the data for distributed sampler
        #     self.train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         torch.utils.data.RandomSampler(
        #             self.trainset, replacement=True, num_samples=num_samples),
        #         shuffle=True)
        # else:
        self.train_sampler = torch.utils.data.RandomSampler(self.trainset, replacement=True, num_samples=num_samples) \
            if num_samples is not None else None
        shuffle = True if self.train_sampler is None else False
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                      shuffle=shuffle, sampler=self.train_sampler)

        # print(summary(self.model, tuple((3, cfg.INPUT.TW, 256, 256)), batch_size=1))
        INPUT_SHAPE = (512, 512)
        INPUT_NF = 3
        print(summary(self.net_G, [
              (INPUT_NF, *INPUT_SHAPE), (INPUT_NF, *INPUT_SHAPE)], batch_size=1))
        print("Arguments used: {}".format(cfg), flush=True)
        # params = []
        # for key, value in dict(self.model.named_parameters()).items():
        #   if value.requires_grad:
        #     params += [{'params': [value], 'lr': args.lr, 'weight_decay': 4e-5}]

    def init_distributed(self, cfg):
        torch.cuda.set_device(args.local_rank)
        init_torch_distributed(self.port)
        model = apex.parallel.convert_syncbn_model(self.net_G)
        model.cuda()
        # model, optimizer, start_epoch, best_iou_train, best_iou_eval, best_loss_train, best_loss_eval, amp_weights = \
        #   load_weights(model, self.optimiser, args, self.model_dir, scheduler=None, amp=amp)  # params
        # lr_schedulers = get_lr_schedulers(optimizer, args, start_epoch)
        # opt_levels = {'fp32': 'O0', 'fp16': 'O2', 'mixed': 'O1'}
        # if cfg.TRAINING.PRECISION in opt_levels:
        #     opt_level = opt_levels[cfg.TRAINING.PRECISION]
        # else:
        #     opt_level = opt_levels['fp32']
        #     print('WARN: Precision string is not understood. Falling back to fp32')
        # model, optimiser = amp.initialize(
        #     model, optimiser, opt_level=opt_level)
        # amp.load_state_dict(amp_weights)
        if torch.cuda.device_count() > 1:
            model = apex.parallel.DistributedDataParallel(
                model, delay_allreduce=True)
        self.world_size = torch.distributed.get_world_size()
        print("Intitialised distributed with world size {} and rank {}".format(
            self.world_size, args.local_rank))
        # return model, optimiser # TODO

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.raw_pred = self.net_G(self.fg, self.bg)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        self.reconstruction = self.fg+self.bg + self.raw_pred
        pred_fake = self.net_D(self.reconstruction.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.net_D(self.target_dict['images'])
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * 0.001
        self.loss_dict['loss_D'] = self.loss_D.item()
        self.loss_D.backward()

    def backward_G(self):
        target = self.target_dict['images']  # TODO
        synth_img = self.fg+self.bg + self.raw_pred
        self.loss_G_L1 = self.criterionL1(
            synth_img, target.float().cuda()) * 0.999
        pred_fake = self.net_D(synth_img)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.001
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_dict['loss_G_GAN'] = self.loss_G_GAN.item()
        self.loss_dict['loss_G_L1'] = self.loss_G_L1.item()
        # combine loss and calculated gradients
        # * final loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.net_D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(self.net_D, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.loss_dict['total_loss'] = self.loss_dict['loss_G_GAN'] + \
            self.loss_dict['loss_D'] + self.loss_dict['loss_G_L1']

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @deprecate
    def compute_loss(self):
        """
        :param cfg: configuration file
        :param target_dict: dict of targets({"flow":<>, "mask": <>})
        :param pred_dict: dictionary of predictions
        """
        result = {'total_loss': torch.tensor(0).float().cuda()}
        if 'l1' in self.cfg.TRAINING.LOSSES.NAME:  # * l1_rec + GAN_Loss
            pass
        return result

    def train(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # switch to train mode
        self.net_G.train()
        self.net_D.train()
        # self.ious.reset()
        self.losses.reset()

        end = time.time()
        for i, input_dict in enumerate(self.trainloader):
            self.fg = input_dict["fg"].float().cuda()
            self.bg = input_dict["bg"].float().cuda()
            self.target_dict = dict([(k, t.float().cuda())
                                     for k, t in input_dict['target'].items()])
            data_time.update(time.time() - end)

            # * forward
            self.optimize_parameters()
            pred = format_pred(self.raw_pred)
            # * loss

            self.iteration += 1
            # Average loss and accuracy across processes for logging
            if torch.cuda.device_count() > 1:
                reduced_loss = dict(
                    [(key, reduce_tensor(val, self.world_size).data.item()) for key, val in self.loss_dict.items()])
            else:
                reduced_loss = dict([(key, val)
                                     for key, val in self.loss_dict.items()])

            self.losses.update(reduced_loss)  # tw frames in a batch

            for k, v in self.losses.val.items():
                self.writer.add_scalar("loss_{}".format(k), v, self.iteration)
            if args.show_image_summary:
                show_image_summary(self.iteration, self.writer, self.in_dict, self.target_dict,
                                   pred)

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            loss_str = ' '.join(["{}:{:4f}({:4f})".format(k, self.losses.val[k], self.losses.avg[k])
                                 for k, v in self.losses.val.items()])

            if args.local_rank == 0:
                print('[Iter: {0}]Epoch: [{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'LOSSES - {loss})\t'.format(
                          self.iteration, self.epoch, i * self.world_size * self.batch_size,
                          len(self.trainloader) *
                          self.batch_size * self.world_size,
                          self.world_size * self.batch_size / batch_time.val,
                          self.world_size * self.batch_size / batch_time.avg,
                          batch_time=batch_time, data_time=data_time, loss=loss_str), flush=True)

                if self.iteration % 10000 == 0:
                    if not os.path.exists(self.model_dir):
                        os.makedirs(self.model_dir)
                    save_name = '{}/{}.pth'.format(self.model_dir,
                                                   self.iteration)
                    save_checkpointV3(
                        self.epoch, self.iteration, self.net_G, self.net_D, self.optimizer_G, self.optimizer_D, save_name)

        if args.local_rank == 0:
            print('Finished Train Epoch {} Loss {losses.avg}'.
                  format(self.epoch, losses=self.losses), flush=True)

        return self.losses.avg

    def eval(self):
        batch_time = AverageMeter()
        losses = AverageMeterDict()
        count = 0
        # switch to evaluate mode
        self.net_G.eval()

        end = time.time()
        print("Starting validation for epoch {}".format(self.epoch), flush=True)
        for seq in self.testset.get_video_ids():
            self.testset.set_video_id(seq)
            if torch.cuda.device_count() > 1:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.testset, shuffle=False)
            else:
                test_sampler = None
            # test_sampler.set_epoch(epoch)
            testloader = DataLoader(self.testset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler,
                                    pin_memory=True)
            losses_video = AverageMeterDict()
            for i, input_dict in enumerate(testloader):
                with torch.no_grad():
                    input = input_dict["images"]
                    target_dict = dict([(k, t.float().cuda())
                                        for k, t in input_dict['target'].items()])
                    if 'masks_guidance' in input_dict:
                        masks_guidance = input_dict["masks_guidance"]
                        masks_guidance = masks_guidance.float().cuda()
                    else:
                        masks_guidance = None
                    info = input_dict["info"]
                    input_var = input.float().cuda()
                    # compute output
                    self.pred = self.net_G(input_var, masks_guidance)
                    pred = format_pred(pred)
                    in_dict = {"input": input_var, "guidance": masks_guidance}
                    loss_dict = {}
                    # loss_dict = compute_loss( # TODO
                    #     in_dict, pred, target_dict, self.cfg)
                    total_loss = loss_dict['total_loss']

                    self.iteration += 1

                    # Average loss and accuracy across processes for logging
                    if torch.cuda.device_count() > 1:
                        reduced_loss = dict(
                            [(key, reduce_tensor(val, self.world_size).data.item()) for key, val in loss_dict.items()])
                    else:
                        reduced_loss = dict([(key, val.data.item())
                                             for key, val in loss_dict.items()])

                    count = count + 1

                    losses_video.update(reduced_loss, args.world_size)
                    losses.update(reduced_loss, args.world_size)
                    for k, v in losses.val.items():
                        self.writer.add_scalar(
                            "loss_{}".format(k), v, self.iteration)

                    # if args.show_image_summary:
                    #   masks_guidance = input_dict['masks_guidance'] if 'masks_guidance' in input_dict else None
                    #   show_image_summary(count, self.writer, input_dict['images'], masks_guidance, input_dict['target'],
                    #                      pred_mask)

                    torch.cuda.synchronize()
                    batch_time.update((time.time() - end) / args.print_freq)
                    end = time.time()

                    if args.local_rank == 0:
                        loss_str = ' '.join(["{}:{:4f}({:4f})".format(k, losses_video.val[k], losses_video.avg[k])
                                             for k, v in losses_video.val.items()])
                        print('{0}: [{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'LOSSES - {loss})\t'.format(
                                  info[0]['video'], i *
                                  args.world_size, len(
                                      testloader) * args.world_size,
                                  batch_time=batch_time, loss=loss_str),
                              flush=True)
        if args.local_rank == 0:
            loss_str = ' '.join(["{}:{:4f}({:4f})".format(k, losses.val[k], losses.avg[k])
                                 for k, v in losses.val.items()])
            print('Finished Test: Loss --> loss {}'.format(loss_str), flush=True)

        return losses.avg

    def start(self):
        if args.task == "train":
            # best_loss = best_loss_train
            # best_iou = best_iou_train
            # if args.freeze_bn:
            #   encoders = [module for module in self.model.modules() if isinstance(module, Encoder)]
            #   for encoder in encoders:
            #     encoder.freeze_batchnorm()

            start_epoch = self.epoch
            for epoch in range(start_epoch, self.cfg.TRAINING.NUM_EPOCHS):
                self.epoch = epoch
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                loss_mean = self.train()

                for lr_scheduler in self.lr_sched_netD:
                    lr_scheduler.step()
                for lr_scheduler in self.lr_sched_netG:
                    lr_scheduler.step()

                if args.local_rank == 0:
                    print("Total Loss {}".format(loss_mean))
                    if loss_mean['total_loss'] < self.best_loss_train:
                        if not os.path.exists(self.model_dir):
                            os.makedirs(self.model_dir)
                        self.best_loss_train = loss_mean['total_loss'] if loss_mean[
                            'total_loss'] < self.best_loss_train else self.best_loss_train
                        save_name = '{}/{}.pth'.format(
                            self.model_dir, "model_best_train")
                        save_checkpointV3(
                            epoch, self.iteration, self.net_G, self.net_D, self.optimizer_G, self.optimizer_D, save_name)

                # val_loss = self.eval()

        elif args.task == 'eval':
            self.eval()
        elif args.task == 'infer':
            inference_engine = get_inference_engine(self.cfg)
            inference_engine.infer(self.testset, self.net_G)
        else:
            raise ValueError("Unknown task {}".format(args.task))

    def backup_session(self, signalNumber, _):
        if is_main_process() and self.args.task == 'train':
            save_name = '{}/{}_{}.pth'.format(self.model_dir,
                                              "checkpoint", self.iteration)
            print("Received signal {}. \nSaving model to {}".format(
                signalNumber, save_name))
            save_checkpointV2(self.epoch, self.iteration,
                              self.net_G, self.optimizer_G, save_name)
        synchronize()
        cleanup_env()
        exit(1)


def register_interrupt_signals(trainer):
    signal.signal(signal.SIGHUP, trainer.backup_session)
    signal.signal(signal.SIGINT, trainer.backup_session)
    signal.signal(signal.SIGQUIT, trainer.backup_session)
    signal.signal(signal.SIGILL, trainer.backup_session)
    signal.signal(signal.SIGTRAP, trainer.backup_session)
    signal.signal(signal.SIGABRT, trainer.backup_session)
    signal.signal(signal.SIGBUS, trainer.backup_session)
    signal.signal(signal.SIGALRM, trainer.backup_session)
    signal.signal(signal.SIGTERM, trainer.backup_session)


if __name__ == '__main__':
    args = parse_argsV2()
    port = _find_free_port()
    trainer = Trainer(args, port)
    register_interrupt_signals(trainer)
    trainer.start()
    if args.local_rank == 0:
        trainer.backup_session(signal.SIGQUIT, None)
    synchronize()
    cleanup_env()
