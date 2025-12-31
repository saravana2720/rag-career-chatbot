import os
import logging
from typing import List, Dict, Any

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# CONSTANTS
# =====================================================
MAX_CONTEXT_CHARS = 6000
DEFAULT_MODEL = "gemini-2.5-flash"  # ✅ UPDATED

# =====================================================
# LLM SERVICE
# =====================================================
class GeminiLLMService:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)

        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048,
        )

    # ================= BASIC / RAG =================
    def generate_response(
        self,
        query: str,
        context: List[Document] | None = None,
    ) -> Dict[str, Any]:

        sources = []
        context_text = ""

        if context:
            for doc in context:
                if len(context_text) + len(doc.page_content) > MAX_CONTEXT_CHARS:
                    break

                context_text += "\n\n" + doc.page_content

                if doc.metadata:
                    sources.append(doc.metadata)

            prompt = (
                "Answer ONLY using the context below.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question:\n{query}\n\nAnswer:"
            )
        else:
            prompt = f"Question:\n{query}\n\nAnswer:"

        response = self.llm.invoke(prompt)

        return {
            "response": response.content,
            "sources": sources,
            "model": self.model_name,
        }

    # ================= CHAT =================
    def chat_completion(self, messages):
        history = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages[-10:]
        )

        prompt = f"Conversation:\n{history}\n\nAssistant:"
        response = self.llm.invoke(prompt)

        return response.content

    # ================= HEALTH =================
    def test_connection(self):
        r = self.llm.invoke("Reply only with: OK")
        return r.content

    # ================= AVAILABLE MODELS =================
    @staticmethod
    def get_available_models() -> List[str]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)

        try:
            models = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    models.append(m.name.replace("models/", ""))
            return models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
