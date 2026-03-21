from io import BytesIO

import PyPDF2


def _read_uploaded_bytes(uploaded_file):
    # Reset the file pointer when supported so repeated reads stay reliable.
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    return file_bytes


def _is_pdf(uploaded_file):
    file_type = getattr(uploaded_file, "type", "") or ""
    file_name = getattr(uploaded_file, "name", "") or ""
    return file_type == "application/pdf" or file_name.lower().endswith(".pdf")


def _load_pdf_text(file_bytes):
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages.append(page_text.strip())

    if not pages:
        raise ValueError("The PDF was read successfully, but no extractable text was found.")

    return "\n\n".join(pages)


def _load_text_file(file_bytes):
    # utf-8-sig handles files that start with a UTF-8 BOM.
    text = file_bytes.decode("utf-8-sig", errors="ignore").replace("\x00", "")

    if not text.strip():
        raise ValueError("The text file was read successfully, but it contains no usable text.")

    return text


def load_file(uploaded_file):
    file_bytes = _read_uploaded_bytes(uploaded_file)

    if _is_pdf(uploaded_file):
        return _load_pdf_text(file_bytes)

    return _load_text_file(file_bytes)
