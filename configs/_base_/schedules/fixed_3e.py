optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = None
momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=3)
