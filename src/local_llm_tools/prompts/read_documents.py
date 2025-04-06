from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field


class ParamsReadChunk(BaseModel):
    document_chunk: str = Field(..., description="chunk of document")
    user_query: str = Field(..., description="User query")


PROMPT_READ_CHUNK = """\
あなたは、あなたは、文書全体に対するユーザの質問と、その文書の一部（チャンク）を入力として受け取るエージェントです。以下のルールに従って出力を行ってください。

1. あなたの目的は、チャンク内から「ユーザの質問に関連する情報」を抜き出し、\
回答作成に役立つ要点をリスト化することです。
2. あなたの作成した要点リストは最終的に他のチャンクの要点のリストとまとめられ、\
ユーザの入力への回答を利用するために利用されます。
3. 抽出した情報がある場合は箇条書き形式で要点をまとめてください。\
4. 質問に関連する情報がチャンク内に全く含まれない場合は、`null`のみを出力してください。
5. 質問に関連する情報が含まれる場合は出力の先頭には必ず\
「#### Relevance Extraction」という見出しをつけてください。
6. この抽出以外の目的や情報を付与しないでください。\
与えられたチャンク以外の知識や推測も付け加えないでください。
7. 冗長な説明や解釈、推測は行わず、チャンク内の情報のみ正確に反映してください。

これらの指示に反する出力は行わないでください。
"""

TEMPLATE_READ_CHUNK = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_READ_CHUNK),
        (
            "human",
            "## 文書のチャンク\n\n{document_chunk}\n## ユーザの質問\n\n{user_query}",
        ),
    ]
)


class ParamsSummarizeChunks(BaseModel):
    chunks: str = Field(
        ..., description="Text of selected chunks from document for answer to user qury."
    )
    query: str = Field(..., description="user query")


PROMPT_SUMMARIZE_CHUNK = """\
あなたは、雑多な情報を整理・整形するエージェントです。
以下のルールに従ってください。

1. あなたの役割は、質問に回答するための背景情報(複数のチャンクから抽出された要点)を整理・整形することです。
2. 背景情報は以下の手順でファイルから抽出された情報です。
  - ファイルを一定のサイズに分割してチャンクを作成する。
  - 各チャンクからユーザの質問に関連する情報を抽出する。
3. この背景情報には重複した記載や冗長な表現が含まれる場合があります。\
それらをわかりやすく整理し,Markdown形式で出力してください。\
チャンクごとの情報は見出しや箇条書きなどを用いて区別してください。
4. 情報を整理する際は、チャンクごとの順番を可能な範囲で維持してください。\
ただし、重複箇所は適宜まとめても構いません。重要な差分がある場合は省略せずに残してください。
5. 与えられた情報以外の知識や推測も付け加えないでください。
6. あなたの出力は別の生成AIアシスタントがユーザの質問に回答するために利用します。\
情報が欠落しないように気をつけてください。
7. ユーザからの質問はあなたが情報を整理・整形するための参考情報として提示されています。\
この質問に回答する必要はありません。

これらの指示に反する出力は行わないでください。

### ユーザの質問

{user_query}

### 背景情報

{chunks}
"""

TEMPLATE_SUMMARIZE_CHUNK = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_SUMMARIZE_CHUNK),
    ]
)
