thvl_textrecog_data_root = 'data/recog/thvl_recog'

thvl_textrecog_train = dict(
    type='OCRDataset',
    data_root=thvl_textrecog_data_root,
    data_prefix=dict(img_path='imgs'),
    ann_file='textrecog_train.json',
    pipeline=None)

thvl_textrecog_test = dict(
    type='OCRDataset',
    data_root=thvl_textrecog_data_root,
    ann_file='textrecog_test.json',
    data_prefix=dict(img_path='imgs'),
    test_mode=True,
    pipeline=None)