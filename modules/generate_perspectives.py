import asyncio
import sys
from typing import List

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain as as_runnable
from langchain_openai import ChatOpenAI

import conf.config as config
from .expand_topics import RelatedSubjects

wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=1)


class Editor(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the editor.",
    )
    name: str = Field(
        description="Name of the editor.", pattern=r"^[a-zA-Z0-9_-]{1,64}$"
    )
    role: str = Field(
        description="Role of the editor in the context of the topic.",
    )
    description: str = Field(
        description="Description of the editor's focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )


def format_doc(doc, max_length=1000):
    related = "- ".join(doc.metadata["categories"])
    return f"### {doc.metadata['title']}\n\nSummary: {doc.page_content}\n\nRelated\n{related}"[
        :max_length
    ]


def format_docs(docs):
    return "\n\n".join(format_doc(doc) for doc in docs)


def get_gen_perspectives_chain():
    gen_perspectives_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You need to select a diverse (and distinct) group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.\
        You can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on.

        Wiki page outlines of related topics for inspiration:
        {examples}""",
            ),
            ("user", "Topic of interest: {topic}"),
        ]
    )

    gen_perspectives_chain = gen_perspectives_prompt | ChatOpenAI(
        model="gpt-4o-2024-05-13"
    ).with_structured_output(Perspectives)

    return gen_perspectives_chain


def get_expand_chain():
    gen_related_topics_prompt = ChatPromptTemplate.from_template(
        """I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.

    Please list the as many subjects and urls as you can.

    Topic of interest: {topic}
    """
    )
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
    expand_chain = gen_related_topics_prompt | llm.with_structured_output(
        RelatedSubjects
    )

    return expand_chain


@as_runnable
async def survey_subjects(topic: str):
    expand_chain = get_expand_chain()
    related_subjects = await expand_chain.ainvoke({"topic": topic})
    retrieved_docs = await wikipedia_retriever.abatch(
        related_subjects.topics, return_exceptions=True
    )
    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)
    formatted = format_docs(all_docs)
    gen_perspectives_chain = get_gen_perspectives_chain()
    return await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})


async def main():
    example_topic = "Impact of million-plus token context window language models on RAG"
    perspectives = await survey_subjects.ainvoke(example_topic)
    print(perspectives.dict())


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
