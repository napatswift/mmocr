_base_ = [
    '_base_dbnet_resnet18_thaidoc.py',
    '../_base_/datasets/thvote.py',
    '../_base_/datasets/thvl.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_thaivotelog.py',
]

# dataset settings
thvotelog_textdet_train = dict(
       type='ConcatDataset', datasets=[_base_.thvl_textdet_train,
                                       _base_.thvote_textdet_train],
       pipeline=_base_.train_pipeline,
)
thvotelog_textdet_test = dict(
       type='ConcatDataset', datasets=[_base_.thvote_textdet_test],
       pipeline=_base_.test_pipeline,
)
thvotelog_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=thvotelog_textdet_train)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=thvotelog_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)
