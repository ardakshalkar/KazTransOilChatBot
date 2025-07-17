#!/usr/bin/env python3
"""
Test script to verify KazTransOil chatbot setup
"""

import sys
import os
from pathlib import Path

def test_data_files():
    """Test if data files exist and are accessible"""
    print("🔍 Checking data files...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return False
    
    # Check FAISS index
    faiss_file = data_dir / "index_kaztransoil.faiss"
    if not faiss_file.exists():
        print("❌ FAISS index file not found!")
        return False
    print(f"✅ FAISS index found: {faiss_file} ({faiss_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Check pickle mapping
    pickle_file = data_dir / "mapping_kaztransoil.pkl"
    if not pickle_file.exists():
        print("❌ Pickle mapping file not found!")
        return False
    print(f"✅ Pickle mapping found: {pickle_file} ({pickle_file.stat().st_size / 1024:.1f} KB)")
    
    return True

def test_dependencies():
    """Test if all required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('faiss', 'faiss'),
        ('sentence_transformers', 'sentence_transformers'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('openai', 'openai'),
    ]
    
    all_good = True
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            all_good = False
    
    return all_good

def test_data_loading():
    """Test loading the actual data"""
    print("\n📊 Testing data loading...")
    
    try:
        import pickle
        import faiss
        
        # Test pickle loading
        with open("data/mapping_kaztransoil.pkl", "rb") as f:
            documents = pickle.load(f)
        print(f"✅ Loaded {len(documents)} documents from pickle")
        
        # Test FAISS loading
        index = faiss.read_index("data/index_kaztransoil.faiss")
        print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
        
        # Test document structure
        sample_doc = list(documents.values())[0]
        print(f"✅ Sample document structure validated")
        print(f"   - Document ID: {sample_doc.metadata.get('document_id', 'N/A')}")
        print(f"   - Source: {sample_doc.metadata.get('document_source', 'N/A')[:50]}...")
        print(f"   - Content length: {len(sample_doc.page_content)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_loading():
    """Test loading the sentence transformer model"""
    print("\n🤖 Testing model loading...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Test encoding
        test_text = "Тест загрузки модели"
        embedding = model.encode([test_text])
        print(f"✅ Model loaded successfully")
        print(f"   - Embedding dimension: {embedding.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 KazTransOil Chatbot Setup Test")
    print("=" * 40)
    
    tests = [
        ("Data Files", test_data_files),
        ("Dependencies", test_dependencies),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Ready to run the chatbot!")
        print("\nTo start the application:")
        print("  1. Double-click 'start_chatbot.bat'")
        print("  2. Or run: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    main() 