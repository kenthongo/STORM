from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


def get_writer_chain():
    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Wikipedia author. Write the complete wiki article on {topic} using the following section drafts:\n\n"
                "{draft}\n\nStrictly follow Wikipedia format guidelines.",
            ),
            (
                "user",
                'Write the complete Wiki article using markdown format. Organize citations using footnotes like "[1]",'
                " avoiding duplicates in the footer. Include URLs in the footer.",
            ),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
    writer = writer_prompt | llm | StrOutputParser()

    return writer
