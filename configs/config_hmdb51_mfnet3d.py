# optimizer and learning rate
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=2)

# runtime settings
work_dir = './demo'
gpus = range(3)
dist_params = dict(backend='nccl')
data_workers = 2  # data workers per gpu
checkpoint_config = dict(interval=1)  # save checkpoint at every epoch
workflow = [('val', 1)]
total_epochs = 3
resume_from = None
load_from = None

# logging settings
log_level = 'INFO'
log_config = dict(
    interval=50,  # log at every 50 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
