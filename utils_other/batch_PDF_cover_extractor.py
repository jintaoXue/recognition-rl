from pdf2image import convert_from_bytes
from PyPDF2 import PdfWriter, PdfReader
import io
import argparse
import os
# Set up command-line argument parser
# parser = argparse.ArgumentParser(
#     description='Extract cover photo from PDF file.')
# parser.add_argument('input_files',
#                     type=str,
#                     nargs='+',
#                     help='PDF file names to extract cover photos from')
# args = parser.parse_args()

# Loop over input files
def cover_one_pdf(file, save_path):
    print("Extracting cover photo from {}...".format(file))

    # Open input PDF file and extract first page
    pdf_reader = PdfReader(file)
    first_page = pdf_reader.pages[0]

    # Create output PDF writer and add first page
    pdf_writer = PdfWriter()
    pdf_writer.add_page(first_page)

    # Write output PDF data to memory buffer
    buffer = io.BytesIO()
    pdf_writer.write(buffer)

    # Convert output PDF data to image and save as PNG file
    images = convert_from_bytes(buffer.getvalue(),dpi=500)
    output_file = file[:-4] + ".png"
    # breakpoint()
    images[0].save(output_file)

    # Print confirmation message
    print("Done, your cover photo has been saved as {}".format(output_file))

    # Close memory buffer
    buffer.close()

if __name__ == '__main__':
    dir = os.getcwd()
    # dir = os.path.dirname(__file__)

    file_list =  [subdir for subdir in os.listdir(dir) if not os.path.isdir(dir+'/'+subdir)]
    # fir_dir = os.path.abspath(__file__)

    # if not os.path.exists(save_path): os.mkdir(save_path)
    for file in file_list:
        if file.split('.')[-1] == 'pdf' : cover_one_pdf(dir+'/'+file, dir)