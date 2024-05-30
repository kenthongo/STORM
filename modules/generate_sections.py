from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from .generate_initial_outline import Subsection


class SubSection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    content: str = Field(
        ...,
        title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
    )

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.content}".strip()


class WikiSection(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    content: str = Field(..., title="Full content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )
    citations: List[str] = Field(default_factory=list)

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
        return (
            f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )


class SectionWriter:
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever()

    async def retrieve(self, inputs: dict):
        docs = await self.retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
        formatted = "\n".join(
            [
                f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )
        return {"docs": formatted, **inputs}

    def get_section_writer_chain(self):
        section_writer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Wikipedia writer. Complete your assigned WikiSection from the following outline:\n\n"
                    "{outline}\n\nCite your sources, using the following references:\n\n<Documents>\n{docs}\n<Documents>",
                ),
                ("user", "Write the full WikiSection for the {section} section."),
            ]
        )
        llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
        section_writer = (
            self.retrieve
            | section_writer_prompt
            | llm.with_structured_output(WikiSection)
        )
        return section_writer
