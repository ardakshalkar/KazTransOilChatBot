import streamlit as st
import pickle
import faiss
import numpy as np
import openai
from typing import List, Dict, Any
import uuid
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="KazTransOil Knowledge Chatbot",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
    }
    .source-button {
        background-color: #e8f4fd;
        border: 1px solid #1f77b4;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
    }
    .bot-message {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class RAGChatbot:
    def __init__(self):
        self.model = None
        self.index = None
        self.documents = None
        self.document_list = None
        
    @st.cache_resource
    def load_models(_self):
        """Load OpenAI client for embeddings"""
        try:
            # Get API key from environment or Streamlit secrets
            api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
            if not api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            return client
        except Exception as e:
            st.error(f"Error loading OpenAI client: {e}")
            return None
    
    @st.cache_data
    def load_data(_self):
        """Load FAISS index and document metadata"""
        try:
            # Load FAISS index
            index = faiss.read_index("data/index_kaztransoil.faiss")
            
            # Load document mappings
            with open("data/mapping_kaztransoil.pkl", "rb") as f:
                documents = pickle.load(f)
            
            # Convert to list for easier indexing
            document_list = list(documents.values())
            
            return index, documents, document_list
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None
    
    def initialize(self):
        """Initialize the chatbot components"""
        self.model = self.load_models()
        self.index, self.documents, self.document_list = self.load_data()
        
        if not all([self.model, self.index, self.documents]):
            st.error("Failed to initialize chatbot components")
            return False
        return True
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text using text-embedding-3-large"""
        try:
            response = self.model.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return np.array(response.data[0].embedding, dtype='float32')
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            return None
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using FAISS with OpenAI embeddings"""
        if not self.model or not self.index:
            return []
        
        try:
            # Get OpenAI embedding for the query
            query_vector = self.get_embedding(query)
            if query_vector is None:
                return []
            
            # Reshape for FAISS search (needs 2D array)
            query_vector = query_vector.reshape(1, -1)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_vector, k)
            
            # Retrieve corresponding documents
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_list):
                    doc = self.document_list[idx]
                    results.append({
                        'document': doc,
                        'score': float(score),
                        'rank': i + 1
                    })
            
            return results
        except Exception as e:
            st.error(f"Error during search: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved context"""
        if not context_docs:
            return "Извините, я не смог найти релевантную информацию для вашего вопроса."
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Документ {i+1}: {doc['document'].page_content[:500]}..."
            for i, doc in enumerate(context_docs[:3])
        ])
        
        # Simple template-based response (you can replace this with OpenAI API call)
        answer = f"""
        Основываясь на найденных документах, вот информация по вашему запросу:

        {context}

        Для получения более подробной информации рекомендую ознакомиться с источниками ниже.
        """
        
        return answer.strip()

def main():
    st.title("🛢️ KazTransOil Knowledge Chatbot")
    st.markdown("### Умный помощник по документации КазТрансОйл")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
        if not st.session_state.chatbot.initialize():
            st.stop()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = {}
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # OpenAI API Key input
        if not os.getenv('OPENAI_API_KEY') and not st.secrets.get('OPENAI_API_KEY'):
            st.subheader("🔑 OpenAI API Key")
            api_key_input = st.text_input(
                "Введите ваш OpenAI API ключ:", 
                type="password",
                help="Ключ нужен для генерации эмбеддингов (text-embedding-3-large)"
            )
            if api_key_input:
                os.environ['OPENAI_API_KEY'] = api_key_input
                st.success("✅ API ключ установлен!")
                st.rerun()
        else:
            st.success("✅ OpenAI API ключ настроен")
        
        st.markdown("---")
        
        # Search parameters
        num_results = st.slider("Количество документов для поиска", 1, 10, 5)
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.chatbot.document_list:
            st.metric("Всего документов", len(st.session_state.chatbot.document_list))
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Очистить чат"):
            st.session_state.messages = []
            st.session_state.show_sources = {}
            st.rerun()
    
    # Chat interface
    st.markdown("### 💬 Чат")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>Вы:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Помощник:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources button
            if f"sources_{i}" in st.session_state.show_sources:
                sources = st.session_state.show_sources[f"sources_{i}"]
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    show_sources = st.button(f"📚 Показать источники", key=f"btn_sources_{i}")
                
                if show_sources or f"sources_visible_{i}" in st.session_state:
                    st.session_state[f"sources_visible_{i}"] = True
                    
                    if f"sources_visible_{i}" in st.session_state:
                        st.markdown("**📋 Использованные источники:**")
                        
                        for j, source in enumerate(sources):
                            doc = source['document']
                            metadata = doc.metadata
                            
                            with st.expander(f"📄 Источник {j+1}: {metadata.get('document_source', 'Неизвестный источник')[:50]}..."):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write(f"**Документ:** {metadata.get('document_source', 'N/A')}")
                                    st.write(f"**Раздел:** {metadata.get('article', 'N/A')}")
                                    st.write(f"**Релевантность:** {source['score']:.3f}")
                                
                                with col2:
                                    st.write(f"**Токенов:** {metadata.get('token_count', 'N/A')}")
                                    st.write(f"**Символов:** {metadata.get('char_count', 'N/A')}")
                                
                                st.markdown("**Содержание:**")
                                st.text_area("Содержание документа", doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content, 
                                           height=150, key=f"content_{i}_{j}", label_visibility="hidden")
    
    # Chat input
    if query := st.chat_input("Введите ваш вопрос о документации КазТрансОйл..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process query
        with st.spinner("🔍 Поиск релевантной информации..."):
            # Search for similar documents
            similar_docs = st.session_state.chatbot.search_similar_documents(query, num_results)
            
            if similar_docs:
                # Generate answer
                answer = st.session_state.chatbot.generate_answer(query, similar_docs)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Store sources for this message
                message_idx = len(st.session_state.messages) - 1
                st.session_state.show_sources[f"sources_{message_idx}"] = similar_docs
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Извините, я не смог найти релевантную информацию для вашего вопроса. Попробуйте переформулировать запрос."
                })
        
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Разработано для АО «КазТрансОйл» | Технология RAG с OpenAI Embeddings + FAISS
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 