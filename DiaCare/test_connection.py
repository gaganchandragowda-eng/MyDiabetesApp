#!/usr/bin/env python3
"""
DiaCare RAG System Connection Test Script
Tests both database and RAG system connectivity
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_database():
    """Test SQLite database connection"""
    print_header("🗄️  TESTING DATABASE CONNECTION")
    
    try:
        import sqlite3
        
        db_file = 'diacare.db'
        
        if os.path.exists(db_file):
            print(f"✅ Database file found: {db_file}")
            file_size = os.path.getsize(db_file)
            print(f"📊 Database size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
            
            # Test connection
            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get table count
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"✅ Found {len(tables)} tables: {', '.join([t[0] for t in tables])}")
            
            # Get record counts
            cursor.execute("SELECT COUNT(*) as count FROM patients")
            patient_count = cursor.fetchone()[0]
            print(f"👥 Patients: {patient_count}")
            
            cursor.execute("SELECT COUNT(*) as count FROM doctors")
            doctor_count = cursor.fetchone()[0]
            print(f"👨‍⚕️ Doctors: {doctor_count}")
            
            cursor.execute("SELECT COUNT(*) as count FROM predictions")
            prediction_count = cursor.fetchone()[0]
            print(f"📊 Predictions: {prediction_count}")
            
            conn.close()
            return True
        else:
            print(f"❌ Database file NOT found: {db_file}")
            print("💡 Run 'python app.py' first to create the database")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_rag_system():
    """Test RAG system connection"""
    print_header("🧠 TESTING RAG SYSTEM CONNECTION")
    
    try:
        # Set paths
        RAG_PROJECT_DIR = Path("/Users/gagan/iCloud Drive (Archive)/Documents/MyDiabetesApp/DiabetesRAGProject")
        KB_PATH = RAG_PROJECT_DIR / "diabetes_kb_data" / "master_kb.json"
        VECTORSTORE_PATH = RAG_PROJECT_DIR / "vectorstore"
        
        print(f"📂 RAG Project Directory: {RAG_PROJECT_DIR}")
        print(f"📖 Knowledge Base Path: {KB_PATH}")
        print(f"🗃️  Vectorstore Path: {VECTORSTORE_PATH}")
        
        # Check if paths exist
        if not RAG_PROJECT_DIR.exists():
            print(f"❌ RAG project directory not found!")
            return False
        print("✅ RAG project directory found")
        
        if not KB_PATH.exists():
            print(f"❌ Knowledge base not found at: {KB_PATH}")
            return False
        print("✅ Knowledge base file found")
        
        if not VECTORSTORE_PATH.exists():
            print(f"⚠️  Vectorstore not found at: {VECTORSTORE_PATH}")
            print("💡 Vectorstore will be created on first use")
        else:
            print("✅ Vectorstore directory found")
        
        # Add to path
        sys.path.insert(0, str(RAG_PROJECT_DIR))
        
        # Try to import RAG module
        print("\n📥 Attempting to import RAG module...")
        from scripts.rag_query import DiabetesRAG, query_rag
        print("✅ RAG module imported successfully!")
        
        # Initialize RAG system
        print("\n🔧 Initializing RAG system...")
        rag = DiabetesRAG(vectorstore_path=str(VECTORSTORE_PATH))
        print("✅ RAG system initialized successfully!")
        
        # Test query
        print("\n🧪 Testing RAG query...")
        test_question = "What is diabetes?"
        response = query_rag(rag, test_question)
        
        print(f"\n📝 Question: {test_question}")
        print(f"💬 Response preview:")
        print("-" * 60)
        print(response[:300] + "..." if len(response) > 300 else response)
        print("-" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure the RAG project has the correct structure:")
        print("   DiabetesRAGProject/")
        print("   ├── scripts/")
        print("   │   └── rag_query.py")
        print("   └── diabetes_kb_data/")
        print("       └── master_kb.json")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_imports():
    """Test Flask and required dependencies"""
    print_header("📦 TESTING DEPENDENCIES")
    
    dependencies = [
        ('flask', 'Flask'),
        ('werkzeug.security', 'Werkzeug Security'),
        ('sqlite3', 'SQLite3'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            all_ok = False
    
    return all_ok

def main():
    """Run all tests"""
    print("="*60)
    print("  🏥 DiaCare System Connection Test")
    print("="*60)
    
    # Test 1: Dependencies
    deps_ok = test_flask_imports()
    
    # Test 2: Database
    db_ok = test_database()
    
    # Test 3: RAG System
    rag_ok = test_rag_system()
    
    # Summary
    print_header("📊 TEST SUMMARY")
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"Database: {'✅ PASS' if db_ok else '❌ FAIL'}")
    print(f"RAG System: {'✅ PASS' if rag_ok else '❌ FAIL'}")
    
    if deps_ok and db_ok and rag_ok:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())