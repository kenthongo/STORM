from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from modules.generate_initial_outline import Outline


def get_refine_outline_chain():
    refine_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Wikipedia writer. You have gathered information from experts and search engines. Now, you are refining the outline of the Wikipedia page. \
    You need to make sure that the outline is comprehensive and specific. \
    Topic you are writing about: {topic} 

    Old outline:

    {old_outline}""",
            ),
            (
                "user",
                "Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nWrite the refined Wikipedia outline:",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-2024-05-13")
    # Using turbo preview since the context can get quite long
    refine_outline_chain = refine_outline_prompt | llm.with_structured_output(Outline)

    return refine_outline_chain
