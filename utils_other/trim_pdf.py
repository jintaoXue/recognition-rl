from turtle import up
import PyPDF2
import os


def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-f', '--file', default=None, type=str, help='')
    argparser.add_argument('-t', '--trim', default=[0.0, 0.0, 0.0, 0.0], type=float, nargs='+', help='trim={<left> <right> <up> <down>}')

    args = argparser.parse_args()
    return args

args = generate_args()

print(args)


input_path = args.file
input_path = os.path.expanduser(input_path)



input_dir, input_name = os.path.split(input_path)

output_path = os.path.join(input_dir, input_name.split('.')[0] + '-trim.pdf')



output_file = PyPDF2.PdfFileWriter()
input_file = PyPDF2.PdfFileReader(open(input_path, 'rb'))

page_info = input_file.getPage(0)  # 这里假设每一页PDF都一样大
width = float(page_info.mediaBox.getWidth())  # 宽度
height = float(page_info.mediaBox.getHeight())  # 高度


left = width *args.trim[0]
right = width *args.trim[1]
up = height *args.trim[2]
down = height *args.trim[3]

page_count = input_file.getNumPages()  # 页数

this_page = input_file.getPage(0)  # 获取第1页
this_page.mediaBox.lowerLeft = (0+left, 0+down)
this_page.mediaBox.lowerRight = (width-right, 0+down)
this_page.mediaBox.upperLeft = (0+left, height-up)
this_page.mediaBox.upperRight = (width-right, height-up)

output_file.addPage(this_page)
output_file.write(open(output_path, 'wb'))
