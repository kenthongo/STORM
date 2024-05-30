from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def get_sklearn_vector_store(final_state):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    reference_docs = [
        Document(page_content=v, metadata={"source": k})
        for k, v in final_state["references"].items()
    ]
    # This really doesn't need to be a vectorstore for this size of data.
    # It could just be a numpy matrix. Or you could store documents
    # across requests if you want.
    if not reference_docs:
        raise ValueError("No reference documents available")

    vectorstore = SKLearnVectorStore.from_documents(
        reference_docs,
        embedding=embeddings,
    )
    num_docs = len(reference_docs)

    # k をドキュメント数以下に設定
    k = min(10, num_docs)
    retriever = vectorstore.as_retriever(k=k)

    return retriever
