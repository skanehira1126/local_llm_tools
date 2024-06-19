from local_llm_tools.tagging.prompts import TaggingPrompt
from local_llm_tools.utils.ollama import generate_json


def add_tag_to_all_texts(
    model, df, text_column, dc_prompt: TaggingPrompt, params: dict | None = None
):
    for row in df.itertuples(index=False):
        text = getattr(row, text_column)
        prompt = dc_prompt.build(sentence=text)

        tag_dict = generate_json(
            model=model,
            prompt=prompt,
            options=params,
        )

        tag_dict[text_column] = text

        yield tag_dict
