# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.distributed as dist
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import os

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.opt = opt

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        #import ipdb;ipdb.set_trace()
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_g_opt = train_opt['optim_g']
        optim_type = optim_g_opt.pop('type')

        # PyTorch's foreach Adam/AdamW implementation can create large temporary
        # tensors during optimizer.step(). Disabling it reduces peak VRAM.
        if optim_type in ['Adam', 'AdamW']:
            optim_g_opt.setdefault('foreach', False)

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **optim_g_opt)
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **optim_g_opt)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **optim_g_opt)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        # Keep the reconstructed validation image on CPU. The corrected test()
        # stores crop outputs on CPU, so moving the final image back to GPU would
        # waste VRAM just before metrics/image conversion.
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq

    @torch.enable_grad()
    def generate_adv(self):
        orig_image = self.lq[:len(self.lq)//2].detach().clone()
        gt = self.gt[:len(self.lq)//2]
        iterations = self.opt['attack']['iterations']
        method = self.opt['attack']['method']
        epsilon = self.opt['attack']['epsilon']
        alpha = self.opt['attack']['alpha']

        if 'pgd' in method:
            noise = torch.empty_like(orig_image).uniform_(-epsilon, epsilon)
            sample_images = torch.clamp(orig_image + noise, min=0, max=1).detach()
        else:
            sample_images = orig_image.detach().clone()

        for _ in range(iterations):
            sample_images = sample_images.detach().requires_grad_(True)

            preds = self.net_g(sample_images)
            if not isinstance(preds, list):
                preds = [preds]

            output = preds[-1]

            l_total = 0
            if self.cri_pix:
                l_pix = 0.
                for pred in preds:
                    l_pix += self.cri_pix(pred, gt)
                l_total += l_pix

            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(output, gt)
                if l_percep is not None:
                    l_total += l_percep
                if l_style is not None:
                    l_total += l_style

            # For adversarial image generation, only d(loss)/d(image) is needed.
            # Using autograd.grad avoids accumulating gradients on model weights.
            data_grad = torch.autograd.grad(
                l_total,
                sample_images,
                retain_graph=False,
                create_graph=False
            )[0]

            sample_images = self.fgsm_attack(
                perturbed_image=sample_images,
                epsilon=epsilon,
                alpha=alpha,
                data_grad=data_grad,
                orig_image=orig_image
            )

            self.net_g.zero_grad(set_to_none=True)

            del preds, output, l_total, data_grad
            if 'l_pix' in locals():
                del l_pix
            if 'l_percep' in locals():
                del l_percep
            if 'l_style' in locals():
                del l_style

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.lq[:len(self.lq)//2] = sample_images.detach()

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad(set_to_none=True)

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        ## Add code for adversarial training ##
        if self.opt.get('adv_train', False):
            self.generate_adv()
            # generate_adv computes gradients w.r.t. the input image only. Clear any
            # accidental parameter gradients before the real training backward.
            self.optimizer_g.zero_grad(set_to_none=True)

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)

            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    # FGSM attack code
    def fgsm_attack(self, perturbed_image, epsilon, alpha, data_grad, orig_image, targeted=False):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        if targeted:
            alpha *= -1
        perturbed_image = perturbed_image.detach() + alpha*sign_data_grad
        # Adding clipping to maintain [0,1] range
        delta = torch.clamp(perturbed_image - orig_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(orig_image + delta, 0, 1).detach()
        return perturbed_image

    #@autocast()
    def test(self):
        self.net_g.eval()
        self.data_grad = None

        attack_opt = self.opt.get('attack', None)
        attack_exists = bool(self.opt.get('attacking', False) and attack_opt is not None)

        n = len(self.lq)
        m = self.opt['val'].get('max_minibatch', 1)

        # Normal validation: no attack, no gradients, output immediately moved to CPU.
        if not attack_exists:
            outs = []
            with torch.no_grad():
                i = 0
                while i < n:
                    j = min(i + m, n)
                    pred = self.net_g(self.lq[i:j])
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    del pred
                    i = j

            self.output = torch.cat(outs, dim=0)
            self.outs = self.output

            del outs
            self.net_g.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self.data_grad

        # Attack validation: gradients are used only to build the attacked image.
        orig_image = self.lq.detach().clone()
        targeted = attack_opt['targeted']
        iterations = attack_opt['iterations']

        if ('cospgd' in attack_opt['method'] or
                attack_opt['method'] == 'segpgd' or
                attack_opt['method'] == 'pgd'):
            self.lq = self.lq + torch.empty_like(self.lq).uniform_(
                -attack_opt['epsilon'], attack_opt['epsilon'])
            self.lq = torch.clamp(self.lq, 0, 1).detach()

        train_opt = self.opt['train']
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
            train_opt['pixel_opt']['type'] = pixel_type

        def forward_for_attack(lq_tensor):
            attack_outs = []
            i = 0
            while i < n:
                j = min(i + m, n)
                pred = self.net_g(lq_tensor[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                attack_outs.append(pred)
                i = j
            return attack_outs

        with torch.enable_grad():
            for t in range(iterations):
                self.lq = self.lq.detach().requires_grad_(True)
                outs = forward_for_attack(self.lq)

                if targeted:
                    self.gt = torch.ones_like(self.gt)

                l_pix = 0.
                for pred in outs:
                    l_pix += self.cri_pix(pred, self.gt)

                if 'cospgd' in attack_opt['method']:
                    if attack_opt['method'] == 'cospgd_softmax':
                        cossim = F.cosine_similarity(
                            F.softmax(self.lq, dim=1),
                            F.softmax(self.gt, dim=1),
                            dim=1,
                            eps=10**-20
                        )
                    elif attack_opt['method'] == 'cospgd_sigmoid':
                        cossim = F.sigmoid(
                            F.cosine_similarity(self.lq, self.gt, dim=1, eps=10**-20)
                        )
                    else:
                        cossim = F.cosine_similarity(
                            F.softmax(self.lq, dim=1),
                            F.softmax(self.gt, dim=1),
                            dim=1,
                            eps=10**-20
                        )
                    if targeted:
                        cossim = 1 - cossim
                    l_pix = cossim.detach() * l_pix
                    l_pix = torch.sum(l_pix)
                    l_pix /= cossim.shape[-1] * cossim.shape[-2]
                elif attack_opt['method'] == 'segpgd':
                    lambda_t = t / (2 * iterations)
                    l_pix = torch.sum(
                        torch.where(
                            pred == self.gt,
                            (1 - lambda_t) * l_pix,
                            lambda_t * l_pix
                        )
                    ) / (pred.shape[-2] * pred.shape[-1])
                else:
                    l_pix = l_pix.mean()

                data_grad = torch.autograd.grad(
                    l_pix,
                    self.lq,
                    retain_graph=False,
                    create_graph=False
                )[0]

                self.data_grad = str(data_grad.max().item())
                self.lq = self.fgsm_attack(
                    self.lq,
                    attack_opt['epsilon'],
                    attack_opt['alpha'],
                    data_grad,
                    orig_image,
                    targeted=targeted
                )

                self.net_g.zero_grad(set_to_none=True)
                del outs, l_pix, data_grad
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        final_outs = []
        with torch.no_grad():
            i = 0
            while i < n:
                j = min(i + m, n)
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                final_outs.append(pred.detach().cpu())
                del pred
                i = j

        self.output = torch.cat(final_outs, dim=0)
        self.outs = self.output

        del final_outs
        self.net_g.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.data_grad

    def _slice_val_data(self, val_data, sample_idx):
        """Return one validation image from a possibly batched dataloader item."""
        one_sample = {}
        for key, value in val_data.items():
            if torch.is_tensor(value):
                one_sample[key] = value[sample_idx:sample_idx + 1]
            elif isinstance(value, (list, tuple)):
                one_sample[key] = [value[sample_idx]]
            else:
                one_sample[key] = value
        return one_sample

    def _release_validation_tensors(self):
        """Delete validation tensors held by the model before the next image."""
        for attr in ('lq', 'gt', 'output', 'outs', 'origin_lq', 'idxes', 'original_size'):
            if hasattr(self, attr):
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        self.data_grad = None
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
        except Exception:
            pass

        total = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else len(dataloader)
        if rank == 0:
            pbar = tqdm(total=total, unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            batch_size = val_data['lq'].size(0) if torch.is_tensor(val_data.get('lq', None)) else 1

            for sample_idx in range(batch_size):
                one_val_data = self._slice_val_data(val_data, sample_idx)
                img_name = osp.splitext(osp.basename(one_val_data['lq_path'][0]))[0]

                try:
                    self.feed_data(one_val_data, is_val=True)
                    if self.opt['val'].get('grids', False):
                        self.grids()

                    self.data_grad = self.test()
                    if rank == 0:
                        pbar.set_postfix({"Sanity ": self.data_grad})

                    if self.opt['val'].get('grids', False):
                        self.grids_inverse()

                    visuals = self.get_current_visuals()
                    sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)

                    gt_img = None
                    if 'gt' in visuals:
                        gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)

                    if save_img:
                        if sr_img.shape[2] == 6:
                            L_img = sr_img[:, :, :3]
                            R_img = sr_img[:, :, 3:]
                            visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                            imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                            imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                        else:
                            if self.opt['is_train']:
                                save_img_path = osp.join(self.opt['path']['visualization'],
                                                         img_name,
                                                         f'{img_name}_{current_iter}.png')

                                save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                            img_name,
                                                            f'{img_name}_{current_iter}_gt.png')
                            else:
                                save_img_path = osp.join(
                                    self.opt['path']['visualization'], dataset_name,
                                    f'{img_name}.png')
                                save_gt_img_path = osp.join(
                                    self.opt['path']['visualization'], dataset_name,
                                    f'{img_name}_gt.png')

                            imwrite(sr_img, save_img_path)
                            if gt_img is not None:
                                imwrite(gt_img, save_gt_img_path)

                    if with_metrics:
                        opt_metric = deepcopy(self.opt['val']['metrics'])
                        if use_image:
                            for name, opt_ in opt_metric.items():
                                metric_type = opt_.pop('type')
                                self.metric_results[name] += getattr(
                                    metric_module, metric_type)(sr_img, gt_img, **opt_)
                        else:
                            for name, opt_ in opt_metric.items():
                                metric_type = opt_.pop('type')
                                self.metric_results[name] += getattr(
                                    metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

                    cnt += 1
                    if rank == 0:
                        pbar.update(1)
                        pbar.set_description(f'Test {img_name}')

                finally:
                    self._release_validation_tensors()

        if rank == 0:
            pbar.close()

        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)
            self.collected_metrics = collected_metrics
        else:
            self.collected_metrics = OrderedDict(cnt=torch.tensor(cnt).float().to(self.device))

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        if self.opt.get('dist', False):
            torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0 and with_metrics:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        return self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)