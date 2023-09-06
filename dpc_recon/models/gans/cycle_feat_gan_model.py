# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import GAN_MODELS
from mmgen.models.common import set_requires_grad
from mmgen.models.translation_models import CycleGAN
from torch.nn.parallel.distributed import _find_tensors
from torch.nn.utils import clip_grad


@GAN_MODELS.register_module()
class CycleFeatGANModel(CycleGAN):
    """CycleGAN model for unpaired feature_map-to-feature_map translation.

    Ref:
      Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
      Networks
    """

    def __init__(self, grad_max_norm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_max_norm = grad_max_norm

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
                log.update({f'model_grad_norm-cycle_feat_gan-{module}': float(grad_norm)})
            elif grad_norm == 0:
                raise ValueError(f'Returned value of grad_norm is zero.')
        else:
            raise ValueError(f'Returned value of grad_norm is {type(grad_norm)}.')
        
        return log