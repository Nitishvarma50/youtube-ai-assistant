from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) system for YouTube transcripts."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200, k=4, temperature=0.2):
        """
        Initialize the RAG system.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            k: Number of chunks to retrieve
            temperature: LLM temperature setting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.temperature = temperature
        
        # Initialize components
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        
        # Prompt template
        self.prompt = PromptTemplate(
            template="""You are an expert AI assistant. Use the following context to answer the question at the end.
      if you don't know the answer, just say that you don't know, don't try to make up an answer.
        Context: {context}
        question: {question}
         """,
            input_variables=["context", "question"]
        )
        
        self.vector_store = None
        self.retriever = None
        self.chain = None
    
    def index_transcript(self, transcript_text: str):
        """
        Index the transcript text by splitting and creating vector store.
        
        Args:
            transcript_text: Full transcript text
        """
        # Split text into chunks
        chunks = self.splitter.create_documents([transcript_text])
        print(f"📄 Created {len(chunks)} chunks")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings_model)
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": self.k}
        )
        
        # Build chain
        self._build_chain()
        print("✅ Indexing complete!")
    
    def _format_docs(self, retrieved_docs):
        """Format retrieved documents into a single context string."""
        context_texts = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return context_texts
    
    def _build_chain(self):
        """Build the RAG chain."""
        parallel_chain = RunnableParallel({
            'context': self.retriever | RunnableLambda(self._format_docs),
            'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        self.chain = parallel_chain | self.prompt | self.llm | parser
    
    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: User question
            
        Returns:
            Answer from the LLM
        """
        if self.chain is None:
            raise RuntimeError("System not indexed yet. Call index_transcript() first.")
        
        return self.chain.invoke(question)
