# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import RUNNERS
from mmgen.core.runners.dynamic_iterbased_runner import *


@RUNNERS.register_module()
class MMDet3dMMGenDynamicIterBasedRunner(DynamicIterBasedRunner):
    """Dynamic Iterbased Runner for MMDet3d-MMGen framework.

    In this Dynamic Iterbased Runner, we will pass the ``reducer`` to the
    ``train_step`` so that the models can be trained with dynamic architecture.
    More details and clarification can be found in this [tutorial](docs/en/tutorials/ddp_train_gans.md).  # noqa

    Args:
        is_dynamic_ddp (bool, optional): Whether to adopt the dynamic ddp.
            Defaults to False.
        pass_training_status (bool, optional): Whether to pass the training
            status. Defaults to False.
        fp16_loss_scaler (dict | None, optional): Config for fp16 GradScaler
            from ``torch.cuda.amp``. Defaults to None.
        use_apex_amp (bool, optional): Whether to use apex.amp to start mixed
            precision training. Defaults to False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, data_loader, **kwargs):
        if is_module_wrapper(self.model):
            _model = self.model.module
        else:
            _model = self.model
        self.model.train()
        self.mode = 'train'
        # check if self.optimizer from model and track it
        if self.optimizer_from_model:
            self.optimizer = _model.optimizer

        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        self.call_hook('before_fetch_train_data')
        data_batch = next(self.data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')

        # prepare input args for train_step
        # running status
        if self.pass_training_status:
            running_status = dict(iteration=self.iter, epoch=self.epoch)
            kwargs['running_status'] = running_status
        # ddp reducer for tracking dynamic computational graph
        if self.is_dynamic_ddp:
            kwargs.update(dict(ddp_reducer=self.model.reducer))

        if self.with_fp16_grad_scaler:
            kwargs.update(dict(loss_scaler=self.loss_scaler))

        if self.use_apex_amp:
            kwargs.update(dict(use_apex_amp=True))

        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)

        # the loss scaler should be updated after ``train_step``
        if self.with_fp16_grad_scaler:
            self.loss_scaler.update()

        # further check for the cases where the optimizer is built in
        # `train_step`.
        if self.optimizer is None:
            if hasattr(_model, 'optimizer'):
                self.optimizer_from_model = True
                self.optimizer = _model.optimizer

        # check if self.optimizer from model and track it
        if self.optimizer_from_model:
            self.optimizer = _model.optimizer
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1