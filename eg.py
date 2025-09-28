import PyPDF2

pdf_path = "data/icici/icici_sample.pdf"

with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for i, page in enumerate(reader.pages):
        print(f"\n--- Page {i+1} ---")
        text = page.extract_text()
        print(repr(text[:1000]))  # print first 1000 chars
        break  # test only first page
