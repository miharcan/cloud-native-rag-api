import os

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class RAGPipeline:
    def __init__(self):
        self.location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.model_name = os.getenv("VERTEX_MODEL", "gemini-2.5-flash-lite")
        self.embedding_model = os.getenv(
            "VERTEX_EMBEDDING_MODEL", "text-embedding-005"
        )

        # LLM
        self.llm = ChatVertexAI(
            model=self.model_name,
            project=self.project_id,
            location=self.location,
            temperature=0.2
        )

        # Embeddings
        self.embeddings = VertexAIEmbeddings(
            model_name=self.embedding_model,
            project=self.project_id,
            location=self.location
        )


        # Vector store
        self.vectorstore = self._create_vector_store()

    def _create_vector_store(self):
        documents = [
            "Cloud-native systems are designed for scalability and resilience.",
            "Retrieval-Augmented Generation improves factual grounding.",
            "Vertex AI provides managed machine learning services on GCP.",
            "Serverless architectures scale automatically and reduce operational overhead."
        ]

        docs = [Document(page_content=text) for text in documents]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        split_docs = splitter.split_documents(docs)

        return FAISS.from_documents(split_docs, self.embeddings)

    def ask(self, question: str):
        # Retrieve relevant documents
        retrieved_docs = self.vectorstore.similarity_search(question, k=3)

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [doc.page_content for doc in retrieved_docs]
        }
