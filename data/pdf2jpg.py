import fitz
import os
import argparse

def get_pdf_files(folder='./'):
    pdf_filenames = []
    for entry in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, entry)) and entry.lower().endswith('.pdf'):
            pdf_filenames.append(os.path.join(folder, entry))
    return pdf_filenames

def pdf_to_jpg(pdf_path: str, output_folder: str, fixed_height: int, page_index: int):

    pdf_document = fitz.open(pdf_path)

    page_num = len(pdf_document)
    for page_number in range(page_num):
        page = pdf_document.load_page(page_number)

        original_width = page.rect.width
        original_height = page.rect.height
        new_width = int((fixed_height / original_height) * original_width)
        zoom_x = new_width / original_width
        zoom_y = fixed_height / original_height
        mat = fitz.Matrix(zoom_x, zoom_y)

        pix = page.get_pixmap(matrix=mat)

        output_path = f"{output_folder}/page_{page_number + page_index}.jpg"
        pix.save(output_path)

    pdf_document.close()
    return page_num + page_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pdf files to images.")
    parser.add_argument('-f', '--folder', type=str, default='./pdf_files', help='The folder path where the PDF document is saved.')
    parser.add_argument('-o', '--output', type=str, default='./image', help='The folder path where the converted images are saved.')
    parser.add_argument('--height', type=int, default=1000, help='height of the converted images.')

    args = parser.parse_args()


    pdf_list = get_pdf_files(args.folder)
    print(pdf_list)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    
    page_index = 1
    for pdf_path in pdf_list:
        page_index = pdf_to_jpg(pdf_path, args.output, args.height, page_index)