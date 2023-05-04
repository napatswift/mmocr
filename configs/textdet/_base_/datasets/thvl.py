# synthetic dataset
thvl_textdet_data_root = 'data/det/thvl'

thvl_textdet_train = dict(
    type='OCRDataset',
    data_root=thvl_textdet_data_root,
    ann_file='textdet_train.json',
    data_prefix=dict(img_path='imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

thvl_textdet_test = dict(
    type='OCRDataset',
    data_root=thvl_textdet_data_root,
    ann_file='textdet_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=None)
