import json

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import markitdown


if __name__ == "__main__":
    # Parameters
    chunk_size = 200
    chunk_overlap = 50

    # markdown 分割用
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Documentのmarkdown変換
    # 経産省：AI事業者ガイドラインをsample_docsに配置
    # https://www.meti.go.jp/press/2024/04/20240419004/20240419004-1.pdf
    md = markitdown.MarkItDown()
    md_document = md.convert("./sample_docs/20240419004-1.pdf").text_content

    print(f"Lenght of documents converted to markdown: {len(md_document)}")

    # テキスト分割
    # ヘッダーは含む
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    # md分割 -> テキスト分割
    splitted_md_documents = markdown_splitter.split_text(md_document)

    result = text_splitter.split_documents(splitted_md_documents)

    with open("sample_docs/chunks.json", "w") as f:
        json.dump(
            {
                "doc_id": "doc_1",
                "content": md_document,
                "chunks": [
                    {
                        "chunk_id": f"doc_1_chunk_{idx}",
                        "ordinal_index": idx,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for idx, doc in enumerate(result)
                ],
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
