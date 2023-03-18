# training schedule for 1x
_base_ = [
    '../_base_/datasets/thvl.py',
    '../_base_/datasets/synTH.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]
# dataset settings
train_list = [
    _base_.thvl_textrecog_train,
    _base_.synTH_textrecog_train
]
test_list = [
    _base_.thvl_textrecog_test,
    _base_.synTH_textrecog_test
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50),)

dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/thai_digits_symbols.txt',
    with_padding=True)

train_dataloader = dict(
    batch_size=384,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

test_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

val_dataloader = test_dataloader

# val_evaluator = dict(
#     dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
# test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64 * 4)
