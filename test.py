from modules.pdf_reader import open_and_read_pdf

pdf_path = "data/pdf/modern operating systems, 4th edition.pdf"  # or whatever you're actually using
pages = open_and_read_pdf(pdf_path)

for item in pages[-20:]: 
    print(item["page_number"], "->", item["text"][:80])
