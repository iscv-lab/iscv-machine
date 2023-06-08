from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.shared import Pt
import os, sys
from docx.shared import Cm
from docx2pdf import convert
import shutil

sys.path.append(os.path.abspath(os.path.join("..", "..")))


def create_docx_if_not_exist(file_path):
    if not os.path.exists(file_path):
        doc = Document()
        doc.save(file_path)
        print(f"Created a new Word document: {file_path}")
    else:
        print(f"The Word document already exists: {file_path}")


def convert_docx_to_pdf(docx_file, pdf_file):
    try:
        convert(docx_file, pdf_file)
        print("Conversion successful!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def handle_report(employeeid: int, employeename: str, interview_id: str):
    current_path = os.path.dirname(__file__) + "/"
    folder_path = f"./public/interview/{interview_id}/"
    docx_path = folder_path + "report.docx"
    shutil.copyfile(current_path + "template.docx", docx_path)
    # Đường dẫn tới file .txt
    qa_path = folder_path + "qa.txt"
    result_path = folder_path + "result.txt"

    # Đọc dữ liệu từ file .txt
    with open(qa_path, "r") as file:
        qa_data = file.read()

    with open(result_path, "r") as file:
        result_data = file.read()
    # create_docx_if_not_exist(docx_path)
    # Load file .docx
    document = Document(docx_path)

    paragraph_index = 3
    if paragraph_index < len(document.paragraphs):
        # Truy cập vào đoạn văn cần xem
        paragraph = document.paragraphs[paragraph_index]
        text = paragraph.text
        # Cái này phải là ID chứ ví dụ video số 5 là 005 chứ sao em set cứng
        # dạ e mới làm trên 1 người e chưa cho chạy vòng lặp á a
        new_text = text[:9] + employeeid + text[9:]
        paragraph.text = new_text
        name_author = employeename
        # In ra nội dung của đoạn văn
        text = paragraph.text

        for run in paragraph.runs:
            run.font.name = "Cambria"
        for run in paragraph.runs:
            run.font.size = Pt(11)
            run.font.bold = True
            words = text.split()
            for word in words:
                # Kiểm tra nếu từ là "ID001"
                if word == "-":
                    text = paragraph.add_run(
                        "Người thực hiện phỏng vấn: " + name_author + " "
                    )
                    # Đặt chữ in đậm
                    text.font.italic = True

            # nó in nghiên cái ID luôn rồi a
    print(paragraph.text)

    # =================================================================
    # Load dữ liệu vào bảng

    table_index = 1
    table = document.tables[table_index]
    for i in range(len(table.rows) - 1):
        row_index = i + 1
        column_index1 = 2
        cell = table.cell(row_index, column_index1)
        # Thêm dữ liệu vào ô
        paragraph1 = cell.add_paragraph()
        run = paragraph1.add_run(qa_data[i * 2])
        if cell.text.startswith(""):
            cell.text = cell.text.lstrip()

    for i in range(len(table.rows) - 1):
        row_index = i + 1
        column_index1 = 6
        cell = table.cell(row_index, column_index1)
        k = (i + 25) * 2
        # Thêm dữ liệu vào ô
        paragraph1 = cell.add_paragraph()
        run = paragraph1.add_run(qa_data[k])
        if cell.text.startswith(""):
            cell.text = cell.text.lstrip()
    # Định dạng dữ liệu
    column_index = 2
    for row in table.rows:
        cell = row.cells[column_index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.paragraphs[0].alignment = WD_ALIGN_VERTICAL.CENTER

        # In đậm dữ liệu và căn giữa trong ô
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                paragraph.alignment = WD_ALIGN_VERTICAL.CENTER
                run.font.name = "Cambria"

    column_index = 6
    for row in table.rows:
        cell = row.cells[column_index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.paragraphs[0].alignment = WD_ALIGN_VERTICAL.CENTER

        # In đậm dữ liệu và căn giữa trong ô
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                paragraph.alignment = WD_ALIGN_VERTICAL.CENTER
                run.font.name = "Cambria"

    # =================================================================
    # Thêm hình ảnh

    image_path = folder_path + "chart.png"
    tables = document.tables
    table_index = 2
    row_index = 0
    column_index = 0
    table = tables[table_index]
    cell = table.cell(row_index, column_index)
    paragraph2 = cell.paragraphs[0]
    run = paragraph2.add_run()
    run.add_picture(image_path, width=Cm(5), height=Cm(5))
    paragraph2.alignment = 1
    # =================================================================
    # Thêm chỉ số
    with open(result_path, "r") as file:
        lines = file.readlines()
    table_index = 2
    column_index = 2
    # Openness
    row_index = 1
    table = tables[table_index]
    cell = table.cell(row_index, column_index)
    content = lines[8]
    text = content[30:-1]
    paragraph3 = cell.add_paragraph()
    run = paragraph3.add_run(text)
    if cell.text.startswith(""):
        cell.text = cell.text.lstrip()
    # Conscientiousness
    row_index = 2
    table = tables[table_index]
    cell = table.cell(row_index, column_index)
    content = lines[4]
    text = content[25:-1]
    paragraph3 = cell.add_paragraph()
    run = paragraph3.add_run(text)
    if cell.text.startswith(""):
        cell.text = cell.text.lstrip()
    # Extroversion
    row_index = 3
    table = tables[table_index]
    cell = table.cell(row_index, column_index)
    content = lines[0]
    text = content[20:-1]
    paragraph3 = cell.add_paragraph()
    run = paragraph3.add_run(text)
    if cell.text.startswith(""):
        cell.text = cell.text.lstrip()
    # Agreeableness
    row_index = 4
    table = tables[table_index]
    cell = table.cell(row_index, column_index)
    content = lines[2]
    text = content[21:-1]
    paragraph3 = cell.add_paragraph()
    run = paragraph3.add_run(text)
    if cell.text.startswith(""):
        cell.text = cell.text.lstrip()
    # Neuroticism
    row_index = 5
    table = tables[table_index]
    cell = table.cell(row_index, column_index)
    content = lines[6]
    text = content[19:-1]
    paragraph3 = cell.add_paragraph()
    run = paragraph3.add_run(text)
    if cell.text.startswith(""):
        cell.text = cell.text.lstrip()

    column_index = 2
    for row in table.rows:
        cell = row.cells[column_index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.paragraphs[0].alignment = WD_ALIGN_VERTICAL.CENTER

        # In đậm dữ liệu và căn giữa trong ô
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                paragraph.alignment = WD_ALIGN_VERTICAL.CENTER
                run.font.name = "Cambria"
    # =================================================================
    # Thêm kết luận

    # Lấy đoạn Extroversion trong docx
    paragraph_index = 8
    paragraph = document.paragraphs[paragraph_index]
    # Lấy Extroversion Comment trong txt
    content = lines[1]
    text = paragraph.add_run(content[22:-1])
    for run in paragraph.runs:
        run.font.name = "Cambria"

    # Lấy đoạn Conscientiousness trong docx
    paragraph_index = 9
    paragraph = document.paragraphs[paragraph_index]
    # Lấy Conscientiousness Comment trong txt
    content = lines[5]
    text = paragraph.add_run(content[27:-1])
    for run in paragraph.runs:
        run.font.name = "Cambria"

    # Lấy đoạn Openness to Experience trong docx
    paragraph_index = 10
    paragraph = document.paragraphs[paragraph_index]
    # Lấy Openness to Experience Comment trong txt
    content = lines[9]
    text = paragraph.add_run(content[32:-1])
    for run in paragraph.runs:
        run.font.name = "Cambria"

    # Lấy đoạn Agreeableness trong docx
    paragraph_index = 11
    paragraph = document.paragraphs[paragraph_index]
    # Lấy Openness to Agreeableness Comment trong txt
    content = lines[3]
    text = paragraph.add_run(content[23:-1])
    for run in paragraph.runs:
        run.font.name = "Cambria"

    # Lấy đoạn Neuroticism trong docx
    paragraph_index = 12
    paragraph = document.paragraphs[paragraph_index]
    # Lấy Openness to Neuroticism Comment trong txt
    content = lines[7]
    text = paragraph.add_run(content[21:-1])
    for run in paragraph.runs:
        run.font.name = "Cambria"

    # =================================================================
    document.save(folder_path + "report.docx")

    # =================================================================
    # Đường dẫn tới file PDF
    pdf_path = folder_path + "report.pdf"

    convert_docx_to_pdf(docx_path, pdf_path)
