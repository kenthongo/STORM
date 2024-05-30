import asyncio
import sys
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class RelatedSubjects(BaseModel):
    topics: List[str] = Field(
        description="Comprehensive list of related subjects as background research.",
    )


async def get_related_subjects():
    gen_related_topics_prompt = ChatPromptTemplate.from_template(
        """I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.

    Please list the as many subjects and urls as you can.

    Topic of interest: {topic}
    """
    )
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
    example_topic = "Impact of million-plus token context window language models on RAG"
    expand_chain = gen_related_topics_prompt | llm.with_structured_output(
        RelatedSubjects
    )
    related_subjects = await expand_chain.ainvoke({"topic": example_topic})
    return related_subjects


async def main():
    related_subjects = await get_related_subjects()
    print(type(related_subjects))
    print(related_subjects.topics)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
