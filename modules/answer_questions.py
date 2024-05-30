import asyncio
import json
import sys
from typing import List, Optional

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .dialog_roles import get_question, swap_roles
from state.interview_state import InterviewState


class Queries(BaseModel):
    queries: List[str] = Field(
        description="Comprehensive list of search engine queries to answer the user's questions.",
    )


class AnswerWithCitations(BaseModel):
    answer: str = Field(
        description="Comprehensive answer to the user's question with citations.",
    )
    cited_urls: List[str] = Field(
        description="List of urls cited in the answer.",
    )

    @property
    def as_str(self) -> str:
        return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
            f"[{i+1}]: {url}" for i, url in enumerate(self.cited_urls)
        )


@tool
async def search_engine(query: str):
    """Search engine to the internet."""
    results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
    return [{"content": r["body"], "url": r["href"]} for r in results]


def get_gen_queries_chain():
    gen_queries_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant. Query the search engine to answer the user's questions.",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )
    gen_queries_chain = gen_queries_prompt | ChatOpenAI(
        model="gpt-3.5-turbo"
    ).with_structured_output(Queries, include_raw=True)

    return gen_queries_chain


def get_gen_answer_chain():
    gen_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants\
    to write a Wikipedia page on the topic you know. You have gathered the related information and will now use the information to form a response.

    Make your response as informative as possible and make sure every sentence is supported by the gathered information.
    Each response must be backed up by a citation from a reliable source, formatted as a footnote, reproducing the URLS after your response.""",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
    gen_answer_chain = gen_answer_prompt | llm.with_structured_output(
        AnswerWithCitations, include_raw=True
    ).with_config(run_name="GenerateAnswer")

    return gen_answer_chain


async def gen_answer(
    state: InterviewState,
    config: Optional[RunnableConfig] = None,
    name: str = "Subject_Matter_Expert",
    max_str_len: int = 15000,
):
    swapped_state = swap_roles(state, name)  # Convert all other AI messages
    gen_queries_chain = get_gen_queries_chain()
    queries = await gen_queries_chain.ainvoke(swapped_state)
    query_results = await search_engine.abatch(
        queries["parsed"].queries, config, return_exceptions=True
    )
    print("query_results:", query_results)
    print()
    successful_results = [
        res for res in query_results if not isinstance(res, Exception)
    ]
    all_query_results = {
        res["url"]: res["content"] for results in successful_results for res in results
    }
    print("all_query_results:", all_query_results)
    print()
    # We could be more precise about handling max token length if we wanted to here
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].additional_kwargs["tool_calls"][0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    # Only update the shared state with the final answer to avoid
    # polluting the dialogue history with intermediate messages
    gen_answer_chain = get_gen_answer_chain()
    generated = await gen_answer_chain.ainvoke(swapped_state)
    # print(generated)
    # print("------------------------")
    # print()
    # print(generated["parsed"])
    # print("------------------------")
    # print()
    # print(generated["parsed"].cited_urls)
    # exit()
    cited_urls = set(generated["parsed"].cited_urls)
    # Save the retrieved information to a the shared state for future reference
    cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
    print("cited_references:", cited_references)
    print()
    formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
    return {"messages": [formatted_message], "references": cited_references}


async def main():
    example_topic = "Impact of million-plus token context window language models on RAG"
    question = await get_question(example_topic)
    example_answer = await gen_answer(
        {"messages": [HumanMessage(content=question["messages"][0].content)]}
    )
    print(example_answer["messages"][-1].content)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
