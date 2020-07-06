# coding:utf-8
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss


from model.utils.parser_func import parse_args, set_dataset_args


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.log_ckpt_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.vgg16_HTCN import vgg16
    from model.faster_rcnn.resnet_HTCN import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, lc=args.lc,
                           gc=args.gc, la_attention = args.LA_ATT, mid_attention = args.MID_ATT)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            lc=args.lc, gc=args.gc, la_attention = args.LA_ATT, mid_attention = args.MID_ATT)
    # elif args.net == 'res50':
    #     fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, context=args.context)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(10000 / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")
    count_iter = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()

        count_step = 0
        loss_temp_last = 1  
        loss_temp = 0
        loss_rpn_cls_temp = 0
        loss_rpn_box_temp = 0
        loss_rcnn_cls_temp = 0
        loss_rcnn_box_temp = 0

        start = time.time()
        # if epoch % (args.lr_decay_step + 1) == 0:
        if epoch - 1 in  args.lr_decay_step:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)

        for step in range(1, iters_per_epoch + 1):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            #eta = 1.0

            #put source data into variable
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            out_d_ins_softmax = F.softmax(out_d_ins, 1) #[256,2]

            count_step += 1

            loss_temp += loss.item()
            loss_rpn_cls_temp += rpn_loss_cls.mean().item()
            loss_rpn_box_temp += rpn_loss_box.mean().item()
            loss_rcnn_cls_temp += RCNN_loss_cls.mean().item()
            loss_rcnn_box_temp += RCNN_loss_bbox.mean().item()

            ######################### da loss 1 #####################################
            # domain label
            domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda())
            # global alignment loss
            dloss_s = 0.5 * FL(out_d, domain_s)
            # dloss_s = 0.5 * F.cross_entropy(out_d, domain_s)

            ######################### da loss 2 #####################################
            # domain label
            domain_s_mid = Variable(torch.zeros(out_d_mid.size(0)).long().cuda())
            ##### mid alignment loss
            # dloss_s_mid = 0.5 * FL(out_d_mid, domain_s_mid)
            dloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            # dloss_s_ins = 0.5 * torch.mean(out_d_ins_softmax[:,1] ** 2) # out_d_ins[:,1] = 1 - out_d_ins[:,0]
            domain_gt_ins = Variable(torch.zeros(out_d_ins.size(0)).long().cuda())
            dloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################

            #put target data into variable
            im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
            gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()

            out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
            out_d_ins_softmax = F.softmax(out_d_ins, 1)

            ######################### da loss 1 #####################################
            # domain label
            domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
            dloss_t = 0.5 * FL(out_d, domain_t)
            # dloss_t = 0.5 * F.cross_entropy(out_d, domain_t)

            ######################### da loss 2 #####################################
            # domain label
            domain_t_mid = Variable(torch.ones(out_d_mid.size(0)).long().cuda())
            ##### mid alignment loss
            # dloss_t_mid = 0.5 * FL(out_d_mid, domain_t_mid)
            dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            # dloss_t_ins = 0.5 * torch.mean(out_d_ins_softmax[:, 0] ** 2) # out_d_ins[:,0] = 1 - out_d_ins[:,1]
            domain_gt_ins = Variable(torch.ones(out_d_ins.size(0)).long().cuda())
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################

            if args.dataset == 'sim':
                loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + dloss_s_mid * 0.15 + dloss_t_mid * 0.15 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5) * args.eta
            else:
                loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + dloss_s_mid * 0.15 + dloss_t_mid * 0.15 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()

                loss_temp /= count_step
                loss_rpn_cls_temp /= count_step
                loss_rpn_box_temp /= count_step
                loss_rcnn_cls_temp /= count_step
                loss_rcnn_box_temp /= count_step


                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    dloss_s = dloss_s.item()
                    dloss_t = dloss_t.item()
                    dloss_s_p = dloss_s_p.item()
                    dloss_t_p = dloss_t_p.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, step: %3d, count: %3d" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr, count_step, count_iter))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f dloss s: %.4f dloss t: %.4f dloss s pixel: %.4f dloss t pixel: %.4f eta: %.4f" \
                    % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp, dloss_s, dloss_t, dloss_s_p, dloss_t_p,
                       args.eta))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls_temp,
                        'loss_rpn_box': loss_rpn_box_temp,
                        'loss_rcnn_cls': loss_rcnn_cls_temp,
                        'loss_rcnn_box': loss_rcnn_box_temp
                    }
                    # logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                    #                    (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalars(args.log_ckpt_name, info,
                                       (epoch - 1) * iters_per_epoch + step)

                count_step = 0
                loss_temp_last = loss_temp
                loss_temp = 0
                loss_rpn_cls_temp = 0
                loss_rpn_box_temp = 0
                loss_rcnn_cls_temp = 0
                loss_rcnn_box_temp = 0

                start = time.time()

            if epoch > 6 and step in [2000, 4000, 6000, 8000]:
                save_name = os.path.join(output_dir,
                                         'globallocal_target_{}_eta_{}_local_context_{}_global_context_{}_gamma_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                             args.dataset_t, args.eta,
                                             args.lc, args.gc, args.gamma,
                                             args.session, epoch,
                                             step))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save model: {}'.format(save_name))

        save_name = os.path.join(output_dir,
                                 'target_{}_eta_{}_local_{}_global_{}_gamma_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                     args.dataset_t,args.eta,
                                     args.lc, args.gc, args.gamma,
                                     args.session, epoch,
                                     step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()

