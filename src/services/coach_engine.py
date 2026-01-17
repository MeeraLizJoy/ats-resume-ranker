import os
import torch
import chromadb
import requests
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

class ResumeCoach:
    def __init__(self, db_path=None, collection_name="resume_vault_v2"):
        # Priority: Env Var > Argument > Default Local Path
        self.db_path = os.getenv("CHROMA_DB_PATH", db_path or "./chroma_db")
        self.db_path = os.path.abspath(self.db_path)
        os.makedirs(self.db_path, exist_ok=True)
            
        self.db = chromadb.PersistentClient(path=self.db_path)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # --- CORRECT SWITCH LOGIC ---
        self.is_cloud = os.getenv("RENDER") or os.getenv("SPACE_ID") or os.getenv("RAILWAY_STATIC_URL")
        
        if self.is_cloud:
            # Use the "Secret" name you will set in HF Settings
            api_key = os.getenv("GEMINI_API_KEY") 
            self.llm = Gemini(model_name="models/gemini-2.0-flash", api_key=api_key)
            self.embed_model = HuggingFaceEmbedding(model_name="./models/all-MiniLM-L6-v2")
            
        else:
            # Local Deployment: Use Ollama
            base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
            self.llm = Ollama(
                model="llama3.2", 
                base_url=base_url,
                request_timeout=360.0,
                additional_kwargs={
                    "num_ctx": 4096,
                    "temperature": 0.1,
                    "num_thread": 8
                }
            )
            # Local uses the standard download or GPU-accelerated version
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5", 
                device=self.device
            )
        
        self.index = None

    def get_status(self):
        if self.is_cloud:
            return {"ollama": "☁️ Cloud Mode (Gemini)", "device": "Cloud API", "model": "Gemini 2.0 Flash"}
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            ollama_ok = response.status_code == 200
        except:
            ollama_ok = False
        
        return {
            "ollama": "🟢 Active" if ollama_ok else "🔴 Offline",
            "device": f"🚀 GPU ({self.device.upper()})" if self.device == "mps" else "🐢 CPU",
            "model": "Llama 3.2 (3B)"
        }

    def add_jd_to_index(self, jd_text):
        """Specifically indexes the Job Description."""
        doc = Document(text=jd_text, metadata={"filename": "CURRENT_JD", "type": "job_description"})
        if not self.index:
            self.index = VectorStoreIndex.from_documents([doc], storage_context=self.storage_context, embed_model=self.embed_model)
        else:
            self.index.insert(doc)

    def add_to_index(self, text, filename):
        doc = Document(text=text, metadata={"filename": filename, "type": "resume"})
        if not self.index:
            self.index = VectorStoreIndex.from_documents([doc], storage_context=self.storage_context, embed_model=self.embed_model)
        else:
            self.index.insert(doc)

    def query_stream(self, user_query, target_filename=None):
        if not self.index:
            yield "Please upload resumes and a JD first."
            return
        
        # Filter to see BOTH the specific resume and the JD
        filters = None
        if target_filename:
            filters = MetadataFilters(filters=[
                ExactMatchFilter(key="filename", value=target_filename),
                ExactMatchFilter(key="filename", value="CURRENT_JD")
            ], condition="or")

        # Create a focused query engine
        streaming_engine = self.index.as_query_engine(
            llm=self.llm, 
            streaming=True,
            filters=filters,
            similarity_top_k=5 # Grab more context chunks
        )
        
        # System Prompt to prevent generic "Entry Level" summaries
        enriched_prompt = (
            f"You are a Senior Technical Recruiter. You have access to the Job Description (labeled CURRENT_JD) "
            f"and the resume of {target_filename}. \n"
            f"Task: {user_query}\n"
            f"Constraint: Only use information from these two documents. Do not give generic career advice. "
            f"If rewriting projects, maintain technical accuracy but align keywords with the JD."
        )
        
        response = streaming_engine.query(enriched_prompt)
        for text in response.response_gen:
            yield text