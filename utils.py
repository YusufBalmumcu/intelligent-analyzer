from typing import List
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from pypdf import PdfReader

def load_batch_texts(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def doc_to_text(file_path):
    loader = UnstructuredFileLoader(file_path, strategy="fast")
    documents = loader.load()
    # Join all text parts into a single string
    return "\n".join(doc.page_content for doc in documents)


def split_document_by_length(content: str, max_size: int, overlap_size: int = 100) -> List[str]:

    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string.")

    if len(content) <= max_size:
        return [content]

    text_splitter = CharacterTextSplitter(
        separator="",
        chunk_size=max_size,
        chunk_overlap=overlap_size,
        length_function=len
    )

    docs = text_splitter.create_documents([content])
    return [doc.page_content for doc in docs]


def split_document_by_structure(content: str, max_size: int, overlap_size: int = 100) -> List[str]:
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string.")

    # Define separators from largest to smallest structure
    separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=max_size,
        chunk_overlap=overlap_size,
        length_function=len
    )

    docs = splitter.create_documents([content])
    return [doc.page_content for doc in docs]

def save_chunks_to_txt(chunks, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.strip() + "\n***\n")
    print(f"Chunks saved to {output_path}")
