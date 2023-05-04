thvc_textdet_data_root = 'data/det/vl+vc-textdet'

thvc_textdet_train = dict(
    type='OCRDataset',
    data_root=thvc_textdet_data_root,
    ann_file='textdet_train.json',
    data_prefix=dict(img_path='imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

thvc_textdet_test = dict(
    type='OCRDataset',
    data_root=thvc_textdet_data_root,
    ann_file='textdet_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=None)
