synTH_textrecog_data_root = 'data/recog/synTH'

synTH_textrecog_train = dict(
    type='OCRDataset',
    data_root=synTH_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

synTH_textrecog_test = dict(
    type='OCRDataset',
    data_root=synTH_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)