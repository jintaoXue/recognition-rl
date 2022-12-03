import os
import imageio
from tqdm import tqdm


def parse_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='Nothing', help='video name.')
    argparser.add_argument('--format', default='mp4', type=str, help='gif, mp4')
    argparser.add_argument('--image-dir', default='./results/tmp_c', type=str, help='')
    argparser.add_argument('--fps', default=30, type=int, help='')

    args = argparser.parse_args()
    return args

args = parse_args()

png_names = os.listdir(args.image_dir)
png_names = sorted(png_names, key=lambda x: eval(x.split('-')[-1][:-4]))
png_names = sorted(png_names, key=lambda x: eval(x.split('-')[0]))

'''
pip install imageio
pip install imageio-ffmpeg
'''

writer = imageio.get_writer('./results/{}.{}'.format(args.description, args.format), fps=args.fps)

for pn in tqdm(png_names):
    writer.append_data(imageio.imread(os.path.join(args.image_dir, pn)))
writer.close()


