
import imageio
import os, sys
import tqdm



def convert(input_path, format):
    """Reference: http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie"""
    outputpath = os.path.splitext(input_path)[0] + '.' + format

    print('reading input')
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    for i,im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    print("\r\nFinalizing...")
    writer.close()
    print("Done.")

convert(os.path.join('./results/tmp', 'adaptive_wo_case3.mp4'), 'gif')

