import glob
import os

import torch
from torch import autograd
import tqdm
from torch.nn.utils import clip_grad_norm_
from pillartrack.ops.iou3d_nms import iou3d_nms_utils
from pillartrack.utils import box_utils
from eval_utils import eval_track_utils

from pillartrack.models import build_network, load_data_to_gpu
import numpy as np
import time

def train_one_epoch(model, optimizer, train_loader, model_func, epoch, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    epoch_loss = 0
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)
        epoch_loss += loss.item()
        loss.backward()
        
        if optim_cfg.GRAD_NORM_CLIP > 0:
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
                    
    epoch_loss /= total_it_each_epoch         
    if tb_log is not None:
        tb_log.add_scalar('train/epoch_loss', epoch_loss, epoch)

    if rank == 0:
        pbar.close()
    return accumulated_iter


def _list_saved_ckpts(ckpt_save_dir):
    ckpt_list = glob.glob(str(ckpt_save_dir / '*.pth'))
    ckpt_list = [x for x in ckpt_list if not x.endswith('_optim.pth')]
    ckpt_list.sort(key=os.path.getmtime)
    return ckpt_list


def _format_metric_for_filename(metric):
    if metric is None:
        return 'nan'
    try:
        metric = float(metric)
    except (TypeError, ValueError):
        return 'nan'
    if not np.isfinite(metric):
        return 'nan'
    return f'{metric:.4f}'

def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, test_loader=None, dataset_cls=None,
                logger=None, eval_output_dir=None, save_eval_to_file=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                epoch=cur_epoch,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']
            if tb_log is not None:
                tb_log.add_scalar('meta_data/epoch_lr', cur_lr, cur_epoch)
            lr_scheduler.step()

            eval_success = float('nan')
            eval_precision = float('nan')
            if test_loader is not None and dataset_cls is not None and logger is not None:
                if rank == 0:
                    cur_result_dir = None
                    if eval_output_dir is not None:
                        cur_result_dir = eval_output_dir / ('epoch_%02d' % (cur_epoch + 1))
                        cur_result_dir.mkdir(parents=True, exist_ok=True)

                    with torch.no_grad():
                        tb_dict = eval_track_utils.eval_track_one_epoch(
                            model, test_loader, cur_epoch + 1, logger, dataset_cls,
                            dist_test=torch.distributed.is_available() and torch.distributed.is_initialized(),
                            result_dir=cur_result_dir, save_to_file=save_eval_to_file
                        )

                    eval_success = tb_dict.get('test/Success', float('nan'))
                    eval_precision = tb_dict.get('test/Precision', float('nan'))
                    if tb_log is not None:
                        tb_log.add_scalar('test/Success', eval_success, cur_epoch + 1)
                        tb_log.add_scalar('test/Precision', eval_precision, cur_epoch + 1)

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()

            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = _list_saved_ckpts(ckpt_save_dir)
                if len(ckpt_list) >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / (
                    'epoch=%02d_success=%s_precision=%s'
                    % (
                        trained_epoch,
                        _format_metric_for_filename(eval_success),
                        _format_metric_for_filename(eval_precision)
                    )
                )
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pillartrack
        version = 'pillartrack+' + pillartrack.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
