import json
import os
import re
from itertools import chain

import faiss
import numpy as np
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import ImageDraw
from sentence_transformers import SentenceTransformer

pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract.exe'


class VectorSearch:
    def __init__(self, model_path, metadata_path):
        self.issue_file = []
        self.index = None
        self.metadata_store = {}
        if os.path.exists(model_path):
            self.index = faiss.read_index(model_path)
        else:
            print("Skipping model loading as the file does not exist")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata_store = json.load(f)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def save_model(self, pdf_data, model_path, metadata_path):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [chunk["context"] for chunk in pdf_data]
        # Write the string to a file
        flattened_texts = [str(item) for item in chain.from_iterable(texts)]
        with open('output_set2.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(flattened_texts))
        embeddings = model.encode(flattened_texts, convert_to_tensor=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.numpy())
        faiss.write_index(index, model_path)

        with open(metadata_path, "w") as f:
            json.dump(pdf_data, f)

    def train_vector_search(self, pdf_path_list, model_path, metadata_path):
        pdf_data = []
        for pdf_path in pdf_path_list:
            print(pdf_path)
            file_name = os.path.splitext(os.path.basename(pdf_path))[0]
            print(f"Reading File {file_name}")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        temp_data = []
                        print(f"\n🔸 Page {i + 1}")
                        # Get table bounding boxes to mask
                        table_bboxes = [table.bbox for table in page.find_tables()]
                        page_image = page.to_image(resolution=150)
                        img = page_image.original.convert("RGB")
                        draw = ImageDraw.Draw(img)

                        # Mask tables (convert PDF bbox to image pixels)
                        for bbox in table_bboxes:
                            x0, top, x1, bottom = bbox
                            x0, x1 = [int(x * page_image.scale) for x in (x0, x1)]
                            top, bottom = [int(y * page_image.scale) for y in (
                                top, bottom)]
                            draw.rectangle([x0, top, x1, bottom], fill="white")

                        ocr_text = pytesseract.image_to_string(
                            img, config="--psm 1")

                        pattern = r"""
                        \[\s*None\s*,\s*['"]\s*['"]\s*\]        |  # [None, '']
                        \[\s*None\s*,\s*['"]\s*['"]\s*\]        |  # [None, '']
                        \[\s*['"]\s*['"]\s*,\s*None\s*\]        |  # ['', None]
                        \[\s*None\s*,\s*None\s*\]               |  # [None, None]
                        \[\s*['"]{2}\s*,\s*['"]{2}\s*\]         # ['', '']
                        """

                        formatted_text = re.sub(pattern, '', ocr_text,
                                                flags=re.VERBOSE)

                        formatted_text = re.sub(r'\n\s*\n+', '\n', formatted_text)
                        formatted_text = re.sub(r'[ \t]+', ' ', formatted_text)
                        formatted_text = re.sub(r' +\n', '\n', formatted_text)
                        formatted_text = re.sub(r'\n+', '\n', formatted_text)

                        temp_data.append(formatted_text)


                        # Extract tables with pdfplumber
                        tables = page.extract_tables()
                        if tables:
                            print("\n📊 Tables:")
                            for table in tables:
                                for row in table:
                                    temp_data.extend(row)
                        else:
                            print("\n📊 Tables: [No tables found]")
                        pdf_data.append({"Manual Name": file_name, "Page No.": i+1,
                                        "context": temp_data})

                        print("\n" + "=" * 80)
            except Exception as e:
                print(f"❌ pdfplumber failed for {file_name}. Falling back to OCR only. Error: ")
                # images = convert_from_path(pdf_path, poppler_path=r"F:\Learning\poppler-24.08.0", dpi=150)
                # for i, img in enumerate(images):
                #     ocr_text = pytesseract.image_to_string(img)
                #     formatted_text = re.sub(r'\n\s*\n+', '\n', ocr_text)
                #     pdf_data.append({
                #         "Manual Name": file_name,
                #         "Page No.": i + 1,
                #         "context": formatted_text
                #     })
                self.issue_file.append(file_name)
        self.save_model(pdf_data, model_path, metadata_path)

    def retrieve_contexts(self, query, k=3):
        query_embedding = self.model.encode([query])
        d, ii = self.index.search(np.array(query_embedding), k=k)
        results = [self.metadata_store[i] for i in ii[0]]
        print(results)


def get_file_names(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    print(files)
    return files


directory = r"F:\Learning\Personal GIT\NLP-Gen-AI-Classroom\Assignment 3\RAG based QnA\Case Study_Batch 2\Data\PDF Documents\Set2"
model_path = r"my_index_set2.faiss"
metadata_path = "metadata_set2.json"

pdf_path_list = get_file_names(directory)
vector_search = VectorSearch(model_path, metadata_path)
vector_search.train_vector_search(pdf_path_list, model_path, metadata_path)
vector_search.retrieve_contexts("Hi")
print(vector_search.issue_file)