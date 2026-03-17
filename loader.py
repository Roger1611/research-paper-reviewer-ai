from io import BytesIO

import PyPDF2


def load_file(uploaded_file):
    # Read PDF files page by page.
    if uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

        pages = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

        return "\n".join(pages)

    # Read plain text files.
    text_bytes = uploaded_file.read()
    return text_bytes.decode("utf-8", errors="ignore")
