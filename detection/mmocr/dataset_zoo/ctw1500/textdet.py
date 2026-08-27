data_root = 'data/ctw1500'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://universityofadelaide.box.com/shared/static/'
                'py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip',
                save_name='ctw1500_train_images.zip',
                md5='e689f94482bde47e52241ff0da16cbb1',
                split=['train'],
                content=['image'],
                mapping=[[
                    'ctw1500_train_images/ctw1500_train_images/train_images',
                    'textdet_imgs/train'
                ]]),
            dict(
                url='https://universityofadelaide.box.com/shared/static/'
                'jikuazluzyj4lq6umzei7m2ppmt3afyw.zip',
                save_name='ctw1500_train_labels.zip',
                md5='0f78190864c1c40fe93ac1dd85fe84a2',
                split=['train'],
                content=['annotation'],
                mapping=[[
                    'ctw1500_train_labels/ctw1500_train_labels/ctw1500_train_labels/',
                    'annotations/train'
                ]])
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'(\d{4}).jpg', r'\1.xml']),
    parser=dict(type='CTW1500AnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://universityofadelaide.box.com/shared/static/'
                't4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip',
                save_name='ctw1500_test_images.zip',
                md5='f0a456dd9dec7ce9d8c0c2fea1d99035',
                split=['test'],
                content=['image'],
                mapping=[[
                    'ctw1500_test_images/ctw1500_test_images/test_images',
                    'textdet_imgs/test'
                ]]),
            dict(
                url='https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/'
                'download',
                save_name='ctw1500_test_labels.zip',
                md5='73dca6f6f1366409681858cf45d66828',
                split=['test'],
                content=['annotation'],
                mapping=[['ctw1500_test_labels/ctw1500_test_labels',
                          'annotations/test']])
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'(\d{4}).jpg', r'000\1.txt']),
    parser=dict(type='CTW1500AnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)
delete = [
    'ctw1500_train_images', 'ctw1500_test_images', 'annotations',
    'ctw1500_train_labels', 'ctw1500_test_labels'
]
config_generator = dict(type='TextDetConfigGenerator')
