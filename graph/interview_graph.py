from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from modules.answer_questions import gen_answer
from modules.dialog_roles import generate_question
from state.interview_state import InterviewState

max_num_turns = 5


def route_messages(state: InterviewState, name: str = "Subject_Matter_Expert"):
    messages = state["messages"]
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    if num_responses >= max_num_turns:
        return END
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    return "ask_question"


def get_interview_graph():
    builder = StateGraph(InterviewState)

    builder.add_node("ask_question", generate_question)
    builder.add_node("answer_question", gen_answer)
    builder.add_conditional_edges("answer_question", route_messages)
    builder.add_edge("ask_question", "answer_question")

    builder.set_entry_point("ask_question")
    interview_graph = builder.compile().with_config(run_name="Conduct Interviews")

    return interview_graph


def visualize_interview_graph():
    interview_graph = get_interview_graph()
    img = interview_graph.get_graph().draw_png()
    with open("graph_image/interview_graph.png", "wb") as f:
        f.write(img)


def main():
    visualize_interview_graph()


if __name__ == "__main__":
    main()
