import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config
from util.ctw3d import CTW3DParts
from util.common_util import (
    AverageMeter,
    create_labeled_result_dirs,
    intersectionAndUnionGPU,
)
from util.data_util import collate_fn
from util.weak_loss import WeakSupervisionLoss
from util import transform as t


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default=os.path.join(PROJECT_ROOT, 'config.yaml'), help='config file')
    parser.add_argument('opts', help='see config.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


@torch.no_grad()
def update_ema_model(teacher, student, decay):
    """Update teacher weights with the exponential moving average of student weights."""
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters()
    ):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)

def main():
    args = get_parser()
    args.data_root = project_path(args.data_root)
    args.save_path, _ = create_labeled_result_dirs(
        project_path(args.result_root),
        args.labeled_point,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    if args.data_name != 'ctw3d_parts':
        raise ValueError("CTWSeg supports the CTW3D-Parts dataset only.")

    run_training(args)


def run_training(config_args):
    global args, best_iou
    args, best_iou = config_args, 0

    from model.ctwseg import ctwseg as Model
    train_transform = t.Compose(
        [
            t.RandomScale([0.9, 1.1]),
            t.ChromaticAutoContrast(),
            t.ChromaticTranslation(),
            t.ChromaticJitter(),
            t.HueSaturationTranslation(),
        ]
    )
    train_data = CTW3DParts(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop, labeled_point=args.labeled_point)
    val_data = CTW3DParts(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=None)
    
    model = Model(
        c=args.fea_dim,
        k=args.classes,
        encoder_channels=args.encoder_channels,
        encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks,
        encoder_strides=args.encoder_strides,
        knn_k=args.knn_k,
        dropout_rate=args.dropout_rate,
        interpolation_k=args.interpolation_k,
        use_position_encoding=args.use_position_encoding,
        use_center_attention=args.use_center_attention,
    ).cuda()
    criterion = WeakSupervisionLoss(
        classes=args.classes,
        ignore_label=args.ignore_label,
        dice_weight=args.dice_weight,
        pseudo_weight=args.pseudo_weight,
        entropy_weight=args.entropy_weight,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_temperature=args.pseudo_temperature,
        pseudo_rampup_epochs=args.pseudo_rampup_epochs,
    )
    validation_criterion = nn.CrossEntropyLoss(
        ignore_index=args.ignore_label
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60,80], gamma=0.1)

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_iou = checkpoint['best_iou']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    teacher = deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if args.resume and os.path.isfile(args.resume):
        teacher_state = checkpoint.get('teacher_state_dict')
        if teacher_state is not None:
            teacher.load_state_dict(teacher_state, strict=True)

    logger.info("train_data samples: '{}'".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    val_loader = None
    if args.evaluate:
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    for epoch in range(args.start_epoch, args.epochs):
        last_checkpoint = os.path.join(args.save_path, 'model_last.pth')
        print('======================================')
        print(last_checkpoint)
        print('======================================')
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader,
            model,
            teacher,
            criterion,
            optimizer,
            epoch,
        )
        scheduler.step()
        epoch_log = epoch + 1
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader,
                teacher,
                validation_criterion,
            )

            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            is_best = mIoU_val > best_iou
            best_iou = max(best_iou, mIoU_val)
            print('-'*100)
            print(best_iou)
            print('-'*100)
        if epoch_log % args.save_freq == 0:
            filename = last_checkpoint
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(),
                        'teacher_state_dict': teacher.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou,
                        'is_best': is_best}, filename)
            if is_best:
                logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
                shutil.copyfile(
                    filename,
                    os.path.join(args.save_path, 'model_best.pth'),
                )

    writer.close()
    logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, teacher, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    supervised_ce_meter = AverageMeter()
    supervised_dice_meter = AverageMeter()
    pseudo_loss_meter = AverageMeter()
    entropy_loss_meter = AverageMeter()
    pseudo_coverage_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    teacher.eval()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(train_loader):
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        
        data_time.update(time.time() - end)
        if target.shape[-1] == 1:
            target = target[:, 0]

        with torch.no_grad():
            teacher_output = teacher([coord, feat, offset])
        output = model([coord, feat, offset])
        loss, loss_details = criterion(
            output,
            target,
            teacher_output,
            epoch,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_model(teacher, model, args.ema_decay)

        output = output.max(1)[1]
        n = coord.size(0)

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        supervised_ce_meter.update(loss_details['supervised_ce'])
        supervised_dice_meter.update(loss_details['supervised_dice'])
        pseudo_loss_meter.update(loss_details['pseudo_loss'])
        entropy_loss_meter.update(loss_details['entropy_loss'])
        pseudo_coverage_meter.update(loss_details['pseudo_coverage'])
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'SupCE {supervised_ce.val:.4f} '
                        'Dice {supervised_dice.val:.4f} '
                        'Pseudo {pseudo_loss.val:.4f} '
                        'Coverage {pseudo_coverage.val:.3f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          supervised_ce=supervised_ce_meter,
                                                          supervised_dice=supervised_dice_meter,
                                                          pseudo_loss=pseudo_loss_meter,
                                                          pseudo_coverage=pseudo_coverage_meter,
                                                          accuracy=accuracy))
        writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
        writer.add_scalar('loss_supervised_ce', supervised_ce_meter.val, current_iter)
        writer.add_scalar('loss_supervised_dice', supervised_dice_meter.val, current_iter)
        writer.add_scalar('loss_pseudo', pseudo_loss_meter.val, current_iter)
        writer.add_scalar('loss_entropy', entropy_loss_meter.val, current_iter)
        writer.add_scalar('pseudo_coverage', pseudo_coverage_meter.val, current_iter)
        writer.add_scalar('pseudo_rampup', loss_details['pseudo_rampup'], current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]
        with torch.no_grad():
            output = model([coord, feat, offset])
        
        loss = criterion(output, target)

        output = output.max(1)[1]
        n = coord.size(0)

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mIoU, mAcc, allAcc

if __name__ == '__main__':
    import gc
    gc.collect()
    main()
