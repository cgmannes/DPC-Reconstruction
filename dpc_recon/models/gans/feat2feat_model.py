# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import GAN_MODELS
from mmgen.models.common import set_requires_grad
from mmgen.models.translation_models import Pix2Pix
from torch.nn.parallel.distributed import _find_tensors
from torch.nn.utils import clip_grad


@GAN_MODELS.register_module()
class Feat2FeatModel(Pix2Pix):
    """Feat2Feat model for paired feature_map-to-feature_map translation.

    Ref:
      Image-to-Image Translation with Conditional Adversarial Networks
    """

    def __init__(self, grad_max_norm, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.grad_max_norm = grad_max_norm

    def _get_disc_loss(self, outputs):
        '''
        GAN loss for the discriminator.

        Modified discriminator loss from Pix2Pix by detaching both
        fake_ab and real_ab.
        '''
        losses = dict()

        discriminators = self.get_module(self.discriminators)
        target_domain = self._default_domain
        source_domain = self.get_other_domains(target_domain)[0]
        fake_ab = torch.cat((outputs[f'real_{source_domain}'],
                             outputs[f'fake_{target_domain}']), 1)
        fake_pred = discriminators[target_domain](fake_ab.detach())
        losses['loss_gan_d_fake'] = self.gan_loss(
            fake_pred, target_is_real=False, is_disc=True)
        real_ab = torch.cat((outputs[f'real_{source_domain}'],
                             outputs[f'real_{target_domain}']), 1)
        real_pred = discriminators[target_domain](real_ab.detach())
        losses['loss_gan_d_real'] = self.gan_loss(
            real_pred, target_is_real=True, is_disc=True)

        loss_d, log_vars_d = self._parse_losses(losses)
        loss_d *= 0.5

        return loss_d, log_vars_d

    def _get_gen_loss(self, outputs):
        target_domain = self._default_domain
        source_domain = self.get_other_domains(target_domain)[0]
        losses = dict()

        discriminators = self.get_module(self.discriminators)
        # GAN loss for the generator
        fake_ab = torch.cat((outputs[f'real_{source_domain}'],
                             outputs[f'fake_{target_domain}']), 1)
        fake_pred = discriminators[target_domain](fake_ab)
        losses['loss_gan_g'] = self.gan_loss(
            fake_pred, target_is_real=True, is_disc=False)

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs)
                if loss_ is None:
                    continue
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses:
                    losses[loss_module.loss_name(
                    )] = losses[loss_module.loss_name()] + loss_
                else:
                    losses[loss_module.loss_name()] = loss_

        loss_g, log_vars_g = self._parse_losses(losses)
        return loss_g, log_vars_g

    def optimize_discriminator(self, outputs, optimizers, ddp=None, loss_weight=1.0):
        """Backward function for the discriminators.

        Args:
            outputs (dict): Dict of forward results.
            optimizers (dict): Dict of model optimizers.
            ddp (DDP): Distributed data parallel object.
            loss_weight (float): Weighting factor for GANCenterPoint.

        Returns:
            dict: Discriminators' loss and loss dict.
        """
        discriminators = self.get_module(self.discriminators)
        set_requires_grad(discriminators, True)
        optimizers['discriminators'].zero_grad()

        loss_d, log_vars_d = self._get_disc_loss(outputs)
        loss_d = loss_weight * loss_d

        if ddp is not None:
            ddp.prepare_for_backward(_find_tensors(loss_d))
        loss_d.backward(retain_graph=True)

        log = self.grad_norm(discriminators, 'disc')
        optimizers['discriminators'].step()
        set_requires_grad(discriminators, False)
        log_vars_d['disc_loss'] = log_vars_d.pop('loss')
        log_vars_d.update(log)

        return log_vars_d

    def optimize_generator(self, outputs, optimizers, ddp=None, loss_weight=1.0):
        """Backward function for the generators.

        Args:
            outputs (dict): Dict of forward results.
            optimizers (dict): Dict of model optimizers.
            ddp (DDP): Distributed data parallel object.
            loss_weight (float): Weighting factor for GANCenterPoint.

        Returns:
            dict: Generators' loss and loss dict.
        """
        generators = self.get_module(self.generators)
        optimizers['generators'].zero_grad()

        loss_g, log_vars_g = self._get_gen_loss(outputs)
        loss_g = loss_weight * loss_g

        if ddp is not None:
            ddp.prepare_for_backward(_find_tensors(loss_g))
        loss_g.backward(retain_graph=True)

        log = self.grad_norm(generators, 'gen')
        optimizers['generators'].step()
        log_vars_g['gen_loss'] = log_vars_g.pop('loss')
        log_vars_g.update(log)

        return log_vars_g

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad or p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, self.grad_max_norm)

    def grad_norm(self, model, module):
        grad_norm = self.clip_grads(model.parameters())
        log = dict()
        if grad_norm is not None:
            if grad_norm !=0:
                # Add grad norm to the logger.
                log.update({f'model_grad_norm-feat2feat_gan-{module}': float(grad_norm)})
            elif grad_norm == 0:
                pass
                # raise ValueError(f'Returned value of grad_norm is zero.')
        else:
            raise ValueError(f'Returned value of grad_norm is {type(grad_norm)}.')
        
        return log