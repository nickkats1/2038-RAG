from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document
from typing import List



def chunker(documents: list[str], chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def load_documents(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()




def document_key(document: Document) -> str:
    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")
    return f"{source}|{page}|{document.page_content}"