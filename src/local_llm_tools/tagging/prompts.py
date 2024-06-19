from dataclasses import dataclass

TEMPLATE_PROMPT = (
    "Please assign a tag {tag_list} to the following sentence according to descriptions of tag.\n"
    "If description is not applied, guess from of tag name.\n"
    "Sentence: '{sentence}'.\n"
    "Descriptions of tag:\n"
    "{tag_description}\n"
    "Respond using JSON and use 'tag' as key of json.\n"
)


@dataclass
class TaggingPrompt:
    tag_list: list[str]
    tag_descriptions: dict[str, str]

    def build(self, sentence: str):

        str_tag_descriptions = ""
        for tag, desc in self.tag_descriptions.items():
            str_tag_descriptions += f"- {tag}: {desc}"

        return TEMPLATE_PROMPT.format(
            tag_list="{}".format(" or ".join(self.tag_list)),
            tag_description=str_tag_descriptions,
            sentence=sentence,
        )
