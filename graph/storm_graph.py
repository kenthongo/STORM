import asyncio

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from graph.interview_graph import get_interview_graph
from modules.generate_final_article import get_writer_chain
from modules.generate_initial_outline import get_generate_initial_outline_chain
from modules.generate_perspectives import survey_subjects
from modules.generate_sections import SectionWriter
from modules.refine_outline import get_refine_outline_chain
from state.research_state import ResearchState


class STORMNode:
    def __init__(self) -> None:
        self.generate_outline_direct = get_generate_initial_outline_chain()
        self.interview_graph = get_interview_graph()
        self.refine_outline_chain = get_refine_outline_chain()
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = SKLearnVectorStore(embedding=self.embeddings)
        self.section_writer = SectionWriter(self.vectorstore).get_section_writer_chain()
        self.refine_outline_chain = get_refine_outline_chain()
        self.writer = get_writer_chain()

    async def initialize_research(self, state: ResearchState):
        topic = state["topic"]
        self.initial_outline = self.generate_outline_direct.ainvoke({"topic": topic})
        coros = (
            self.initial_outline,
            survey_subjects.ainvoke(topic),
        )
        results = await asyncio.gather(*coros)
        return {
            **state,
            "outline": results[0],
            "editors": results[1].editors,
        }

    async def conduct_interviews(self, state: ResearchState):
        topic = state["topic"]
        initial_states = [
            {
                "editor": editor,
                "messages": [
                    AIMessage(
                        content=f"So you said you were writing an article on {topic}?",
                        name="Subject_Matter_Expert",
                    )
                ],
            }
            for editor in state["editors"]
        ]
        # We call in to the sub-graph here to parallelize the interviews
        interview_results = await self.interview_graph.abatch(initial_states)

        return {
            **state,
            "interview_results": interview_results,
        }

    def format_conversation(self, interview_state):
        messages = interview_state["messages"]
        convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
        return f'Conversation with {interview_state["editor"].name}\n\n' + convo

    async def refine_outline(self, state: ResearchState):
        convos = "\n\n".join(
            [
                self.format_conversation(interview_state)
                for interview_state in state["interview_results"]
            ]
        )
        updated_outline = await self.refine_outline_chain.ainvoke(
            {
                "topic": state["topic"],
                "old_outline": state["outline"].as_str,
                "conversations": convos,
            }
        )
        return {**state, "outline": updated_outline}

    async def index_references(self, state: ResearchState):
        all_docs = []
        for interview_state in state["interview_results"]:
            reference_docs = [
                Document(page_content=v, metadata={"source": k})
                for k, v in interview_state["references"].items()
            ]
            all_docs.extend(reference_docs)
        await self.vectorstore.aadd_documents(all_docs)
        return state

    async def write_sections(self, state: ResearchState):
        outline = state["outline"]
        sections = await self.section_writer.abatch(
            [
                {
                    "outline": outline.as_str,
                    "section": section.section_title,
                    "topic": state["topic"],
                }
                for section in outline.sections
            ]
        )
        return {
            **state,
            "sections": sections,
        }

    async def write_article(self, state: ResearchState):
        topic = state["topic"]
        sections = state["sections"]
        draft = "\n\n".join([section.as_str for section in sections])
        article = await self.writer.ainvoke({"topic": topic, "draft": draft})
        return {
            **state,
            "article": article,
        }


def get_storm_graph():
    builder_of_storm = StateGraph(ResearchState)
    storm_node = STORMNode()
    nodes = [
        ("init_research", storm_node.initialize_research),
        ("conduct_interviews", storm_node.conduct_interviews),
        ("refine_outline", storm_node.refine_outline),
        ("index_references", storm_node.index_references),
        ("write_sections", storm_node.write_sections),
        ("write_article", storm_node.write_article),
    ]
    for i in range(len(nodes)):
        name, node = nodes[i]
        builder_of_storm.add_node(name, node)
        if i > 0:
            builder_of_storm.add_edge(nodes[i - 1][0], name)

    builder_of_storm.set_entry_point(nodes[0][0])
    builder_of_storm.set_finish_point(nodes[-1][0])
    storm = builder_of_storm.compile()

    return storm


def visualize_storm_graph():
    storm = get_storm_graph()
    img = storm.get_graph().draw_png()
    with open("graph_image/storm_graph.png", "wb") as f:
        f.write(img)


def main():
    visualize_storm_graph()


if __name__ == "__main__":
    main()
