import pdfplumber

def extract_pdf(pdf_path: str) -> str:

    final_output = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_no, page in enumerate(pdf.pages, start=1):

            page_items = []

            tables = page.find_tables()

            table_boxes = []

            for table in tables:

                table_boxes.append(table.bbox)

                table_text = []

                extracted_table = table.extract()

                for row in extracted_table:

                    cleaned = [
                        cell.strip() if cell else ""
                        for cell in row
                    ]

                    table_text.append(" | ".join(cleaned))

                page_items.append({
                    "type": "table",
                    "top": table.bbox[1],
                    "content": "\n".join(table_text)
                })

            words = page.extract_words()

            outside_words = []

            for word in words:

                inside_table = False

                x = word["x0"]
                y = word["top"]

                for bbox in table_boxes:

                    x0, top, x1, bottom = bbox

                    if x0 <= x <= x1 and top <= y <= bottom:

                        inside_table = True
                        break

                if not inside_table:

                    outside_words.append(word)

            lines = {}

            for word in outside_words:

                key = round(word["top"])

                if key not in lines:
                    lines[key] = []

                lines[key].append(word)

            for top in sorted(lines.keys()):

                line = sorted(lines[top], key=lambda w: w["x0"])

                text = " ".join(
                    word["text"]
                    for word in line
                )

                page_items.append({
                    "type": "text",
                    "top": top,
                    "content": text
                })

            page_items.sort(key=lambda item: item["top"])

            final_output.append(
                f"\n========== PAGE {page_no} ==========\n"
            )

            for item in page_items:

                if item["type"] == "table":

                    final_output.append(
                        "\n----- TABLE -----"
                    )

                final_output.append(item["content"])

    return "\n".join(final_output)

   
if __name__ == "__main__":

    pdf_path = r"uploads\my_college_syllabus.pdf"

    text = extract_pdf(pdf_path)

    with open("extracted_text.txt", "w", encoding="utf-8") as file:
        file.write(text)

    print("Extraction completed successfully!")