# import os

# from dotenv import load_dotenv
# from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
# from injestion import retriever
# from graph.state import GraphState

# load_dotenv()


# def test_retrieval_grader() -> None:s
#     state = GraphState(
#         question="What is the capital of France?", documents=[], web_Search=False
#     )
#     documents = retriever.invoke(state["question"])
#     result = retrieval_grader.invoke(
#         {"document": documents, "question": state["question"]}
#     )
#     assert result.binary_score == "yes"


# def test_retrieval_grader_answer_yes() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#     docs_txt = docs[0].page_content

#     result = retrieval_grader.invoke({"document": docs_txt, "question": question})
#     assert result.binary_score == "yes"

# graph/chains/tests/test_chains.py
import os

from dotenv import load_dotenv

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.state import GraphState
from ingestion import retriever

load_dotenv()


# def test_retrieval_grader() -> None:
#     state = GraphState(
#         question="What is the capital of France?",
#         documents=[],
#         web_Search=False
#     )
#     documents = retriever.invoke(state["question"])
#     result = retrieval_grader.invoke(
#         {"document": documents, "question": state["question"]}
#     )
#     assert result.binary_score == "yes"


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    print(f"\n{generation}\n")


# def test_hallucination_grader_answer_yes() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)

#     generation = generation_chain.invoke({"context": docs, "question": question})
#     res: GradeHallucinations = hallucination_grader.invoke(
#         {"documents": docs, "generation": generation}
#     )
#     assert res.binary_score


# def test_hallucination_grader_answer_no() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)

#     res: GradeHallucinations = hallucination_grader.invoke(
#         {
#             "documents": docs,
#             "generation": "In order to make pizza we need to first start with the dough",
#         }
#     )
#     assert not res.binary_score


# def test_router_to_vectorstore() -> None:
#     question = "agent memory"

#     res: RouteQuery = question_router.invoke({"question": question})
#     assert res.datasource == "vectorstore"


# def test_router_to_websearch() -> None:
#     question = "how to make pizza"

#     res: RouteQuery = question_router.invoke({"question": question})
#     assert res.datasource == "websearch"
