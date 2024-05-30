import asyncio
import sys

from graph.storm_graph import get_storm_graph


async def main():
    storm = get_storm_graph()
    results = None
    async for step in storm.astream(
        {
            "topic": "Impact of million-plus token context window language models on RAG",
        }
    ):
        name = next(iter(step))
        print(name)
        print("-- ", str(step[name])[:300])
        results = step

    article = results["write_article"]["article"]
    print(article)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
