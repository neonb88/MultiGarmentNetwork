#NUM, IMG_SIZE, FACE = 8, 720, False
#NUM, IMG_SIZE, FACE = 8, 240, False           # nxb, `Sun Feb 23 07:05:24 EST 2020
#NUM, IMG_SIZE, FACE = 8, 481, False           # nxb, `Sun May 5 21:56:24 EST 2020
NUM, IMG_SIZE, FACE = 8, 720, False           # nxb, `Sun May 5 23:52:24 EST 2020

config = lambda: None
config.expName = None
config.checkpoint_dir = None
config.train = lambda: None
config.train.batch_size = 4
config.train.lr = 0.001
config.train.decay = 0.001
config.train.epochs = 10
config.latent_code_garms_sz = 1024

config.PCA_=35
config.garmentKeys = ['Pants', 'ShortPants', 'ShirtNoCoat', 'TShirtNoCoat', 'LongCoat']
config.NVERTS =  27554
