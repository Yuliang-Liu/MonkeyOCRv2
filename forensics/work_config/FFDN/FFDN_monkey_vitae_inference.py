_base_ = ['./FFDN_monkey_vitae.py']

name = 'FFDN_monkey_vitae_inference'

vis_backends = []
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
