import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain as as_runnable
from langchain_openai import ChatOpenAI

from .generate_perspectives import survey_subjects
from state.interview_state import InterviewState


def tag_with_name(ai_message: AIMessage, name: str):
    ai_message.name = name
    return ai_message


def swap_roles(state: InterviewState, name: str):
    converted = []
    for message in state["messages"]:
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.dict(exclude={"type"}))
        converted.append(message)
    return {"messages": converted}


@as_runnable
async def generate_question(state: InterviewState):
    editor = state["editor"]
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
    gen_qn_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an experienced Wikipedia writer and want to edit a specific page. \
    Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic. \
    Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

    When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.\
    Please only ask one question at a time and don't ask what you have asked before.\
    Your questions should be related to the topic you want to write.
    Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

    Stay true to your specific perspective:

    {persona}""",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )
    gn_chain = (
        RunnableLambda(swap_roles).bind(name=editor.name)
        | gen_qn_prompt.partial(persona=editor.persona)
        | llm
        | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    result = await gn_chain.ainvoke(state)
    return {"messages": [result]}


async def get_question(topic):
    messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
    perspectives = await survey_subjects.ainvoke(topic)
    question = await generate_question.ainvoke(
        {
            "editor": perspectives.editors[0],
            "messages": messages,
        }
    )

    return question


async def main():
    example_topic = "Impact of million-plus token context window language models on RAG"
    messages = [
        HumanMessage(f"So you said you were writing an article on {example_topic}?")
    ]
    perspectives = await survey_subjects.ainvoke(example_topic)
    question = await generate_question.ainvoke(
        {
            "editor": perspectives.editors[0],
            "messages": messages,
        }
    )

    print(question["messages"][0].content)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
