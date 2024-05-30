from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

import conf.config as config


class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: List[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


def get_generate_initial_outline_chain():
    direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Wikipedia writer. Write an outline for a Wikipedia page about a user-provided topic. Be comprehensive and specific.",
            ),
            ("user", "{topic}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
    generate_outline_direct = direct_gen_outline_prompt | llm.with_structured_output(
        Outline
    )

    return generate_outline_direct


def get_initial_outline(topic: str):
    generate_outline_direct = get_generate_initial_outline_chain()
    initial_outline = generate_outline_direct.invoke({"topic": topic})

    return initial_outline


def main():
    example_topic = "Impact of million-plus token context window language models on RAG"
    initial_outline = get_initial_outline(example_topic)
    print(initial_outline.as_str)


if __name__ == "__main__":
    main()
