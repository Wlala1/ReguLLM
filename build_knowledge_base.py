import re
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import networkx as nx
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import requests
import time
import os
import getpass
from dotenv import load_dotenv

load_dotenv()


class JurisdictionLevel(Enum):
    """Legal jurisdiction levels"""
    INTERNATIONAL = "international"  
    FEDERAL = "federal"  
    STATE = "state"
    LOCAL = "local"
    REFERENCE = "reference"  


@dataclass
class JurisdictionNode:
    """Jurisdiction node"""
    id: str
    name: str
    level: JurisdictionLevel
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    document_ids: List[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.document_ids is None:
            self.document_ids = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level.value,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'document_ids': self.document_ids,
            'metadata': self.metadata
        }


@dataclass
class LegalDocument:
    """Legal document node"""
    id: str
    title: str
    content: str
    jurisdiction_id: str
    document_type: str
    chunks: List[str] = None
    chunk_embeddings: np.ndarray = None
    metadata: Dict = None
    related_document_ids: List[str] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.metadata is None:
            self.metadata = {}
        if self.related_document_ids is None:
            self.related_document_ids = []
    
    def to_dict(self):
        """Convert to JSON-serializable dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'jurisdiction_id': self.jurisdiction_id,
            'document_type': self.document_type,
            'chunks': self.chunks,
            'metadata': self.metadata,
            'related_document_ids': self.related_document_ids
        }


class DynamicQwenClassifier:
    """Classifier supporting dynamic jurisdiction recognition"""
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = (api_url or os.getenv("DASHSCOPE_API_BASE"))
        self.model_name = "qwen-max-latest"
        
        if api_key:
            self.api_key = api_key
        elif os.getenv("DASHSCOPE_API_KEY"):
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
        else:
            print("API key not found, will try free plan or prompt for input...")
            self.api_key = self._get_api_key_interactively()
        
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def _get_api_key_interactively(self) -> Optional[str]:
        """Interactively get API key"""
        print("Please enter your Qwen API key (press Enter if using free plan):")
        api_key = getpass.getpass("API Key: ").strip()
        return api_key if api_key else None
    
    def classify_document(self, document_content: str, file_path: str = "") -> Dict[str, Any]:
        """Use Qwen to classify documents - supports dynamic jurisdiction recognition"""
        content_sample = document_content[:800] if len(document_content) > 800 else document_content
        
        prompt = f"""Please analyze the following legal document, determine its jurisdiction and document type, and return the result in JSON format.

File path: {file_path}
Document content: {content_sample}

Please return JSON format with the following fields:
{{
    "jurisdiction_code": "jurisdiction code (e.g.: california, new_jersey, texas, etc.)",
    "jurisdiction_name": "full jurisdiction name (e.g.: California, New Jersey, Texas, etc.)", 
    "jurisdiction_level": "jurisdiction level (international/federal/state/local/reference)",
    "parent_jurisdiction": "parent jurisdiction code (e.g., state's parent is usa, country's parent is eu, etc.)",
    "document_type": "document type",
    "confidence": "confidence level (0-1)",
    "title": "document title",
    "year": "year (if identifiable)",
    "bill_number": "bill number (if any)"
}}

Jurisdiction level descriptions:
- "international": International organizations (e.g., EU)
- "federal": National level (e.g., USA, Germany, France, etc.)
- "state": State/provincial level (e.g., California, Texas, New Jersey, etc.)
- "reference": Reference documents (glossaries, definitions, etc.)

Parent jurisdiction inference rules:
- US states' parent is "usa"
- EU countries' parent is "eu" 
- Independent countries' parent can be null

Please identify any jurisdiction, don't limit to preset options. If it's a new jurisdiction, please accurately identify its level and parent relationship.
Please return only JSON, no other explanations."""

        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional legal document analysis assistant, skilled at identifying legal document jurisdictions and types from any country or region. You can identify any jurisdiction worldwide, not limited to preset options."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 800
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                try:
                    # Clean markdown format
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    classification = json.loads(content)
                    
                    # Validate and supplement necessary fields
                    required_fields = {
                        "jurisdiction_code": "unknown",
                        "jurisdiction_name": "Unknown Jurisdiction", 
                        "jurisdiction_level": "reference",
                        "parent_jurisdiction": None,
                        "document_type": "Unknown Document",
                        "confidence": 0.5
                    }
                    
                    for field, default in required_fields.items():
                        if field not in classification:
                            classification[field] = default
                    
                    return classification
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
                    print(f"Original content: {content}")
            else:
                print(f"API call failed: {response.status_code}")
                
        except Exception as e:
            print(f"Classifier call exception: {e}")
        
        # Return default classification
        return {
            "jurisdiction_code": "unknown",
            "jurisdiction_name": "Unknown Jurisdiction",
            "jurisdiction_level": "reference", 
            "parent_jurisdiction": None,
            "document_type": "Unknown Document",
            "confidence": 0.1,
            "title": Path(file_path).stem if file_path else "Unknown"
        }


class DynamicLegalGraphRAG:
    """Legal Graph RAG supporting dynamic jurisdiction expansion"""
    
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-base-en-v1.5",
                 max_chunk_size: int = 800,
                 overlap_size: int = 100,
                 qwen_api_url: str = None,
                 qwen_api_key: str = None):
        
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Use dynamic classifier
        self.classifier = DynamicQwenClassifier(qwen_api_url, qwen_api_key)
        
        # Graph data structure
        self.graph = nx.DiGraph()
        self.jurisdictions: Dict[str, JurisdictionNode] = {}
        self.documents: Dict[str, LegalDocument] = {}
        
        # Vector storage
        self.document_embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize RecursiveCharacterTextSplitter for unified chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
        )
        
        # Initialize base jurisdiction structure
        self._initialize_base_jurisdictions()
    
    def _initialize_base_jurisdictions(self):
        """Initialize minimal base jurisdiction structure"""
        base_jurisdictions = [
            # Reference document category
            ("reference", "Reference Documents", JurisdictionLevel.REFERENCE, None),
            
            # Main root nodes
            ("eu", "European Union", JurisdictionLevel.INTERNATIONAL, None),
            ("usa", "United States", JurisdictionLevel.FEDERAL, None),
            ("canada", "Canada", JurisdictionLevel.FEDERAL, None),
            ("uk", "United Kingdom", JurisdictionLevel.FEDERAL, None),
            ("china", "China", JurisdictionLevel.FEDERAL, None),
        ]
        
        print(f"Initializing {len(base_jurisdictions)} base jurisdiction root nodes...")
        
        # Create jurisdiction nodes
        for jur_id, name, level, parent_id in base_jurisdictions:
            jurisdiction = JurisdictionNode(
                id=jur_id,
                name=name,
                level=level,
                parent_id=parent_id,
                metadata={
                    "is_root_node": True,
                    "initialization_type": "base"
                }
            )
            self.jurisdictions[jur_id] = jurisdiction
            print(f"  Created root node: {name} [{level.value}]")
        
        # Establish parent-child relationships
        for jur_id, jurisdiction in self.jurisdictions.items():
            if jurisdiction.parent_id and jurisdiction.parent_id in self.jurisdictions:
                self.jurisdictions[jurisdiction.parent_id].children_ids.append(jur_id)
        
        # Add to graph
        for jur_id, jurisdiction in self.jurisdictions.items():
            self.graph.add_node(jur_id, node_type="jurisdiction", data=jurisdiction)
            if jurisdiction.parent_id and jurisdiction.parent_id in self.jurisdictions:
                self.graph.add_edge(jurisdiction.parent_id, jur_id, relationship="governs")
        
        print("Base jurisdiction structure initialization complete")

    def _create_default_parent_jurisdiction(self, parent_code: str):
        """Create default parent jurisdiction"""
        default_jurisdictions = {
            "usa": ("United States", JurisdictionLevel.FEDERAL, None),
            "eu": ("European Union", JurisdictionLevel.INTERNATIONAL, None),
            "canada": ("Canada", JurisdictionLevel.FEDERAL, None),
            "uk": ("United Kingdom", JurisdictionLevel.FEDERAL, None),
            "china": ("China", JurisdictionLevel.FEDERAL, None)
        }
        
        if parent_code in default_jurisdictions:
            name, level, grandparent = default_jurisdictions[parent_code]
            
            # Ensure grandparent jurisdiction also exists
            if grandparent and grandparent not in self.jurisdictions:
                self._create_default_parent_jurisdiction(grandparent)
            
            parent_jurisdiction = JurisdictionNode(
                id=parent_code,
                name=name,
                level=level,
                parent_id=grandparent,
                metadata={
                    "auto_created": True, 
                    "type": "default_parent",
                    "creation_reason": "missing_parent_jurisdiction"
                }
            )
            
            self.jurisdictions[parent_code] = parent_jurisdiction
            self.graph.add_node(parent_code, node_type="jurisdiction", data=parent_jurisdiction)
            
            # Establish relationship with grandparent
            if grandparent and grandparent in self.jurisdictions:
                self.jurisdictions[grandparent].children_ids.append(parent_code)
                self.graph.add_edge(grandparent, parent_code, relationship="governs")
            
            print(f"Auto-created parent jurisdiction: {name} -> Parent: {grandparent or 'None'}")
        else:
            # Create generic jurisdiction
            print(f"Warning: Unknown parent jurisdiction {parent_code}, creating as generic jurisdiction")
            
            generic_jurisdiction = JurisdictionNode(
                id=parent_code,
                name=parent_code.replace("_", " ").title(),
                level=JurisdictionLevel.FEDERAL,
                parent_id=None,
                metadata={
                    "auto_created": True,
                    "type": "generic_unknown",
                    "creation_reason": "unknown_parent_jurisdiction",
                    "needs_manual_review": True
                }
            )
            
            self.jurisdictions[parent_code] = generic_jurisdiction
            self.graph.add_node(parent_code, node_type="jurisdiction", data=generic_jurisdiction)

    def ensure_jurisdiction_exists(self, classification: Dict[str, Any]) -> str:
        """Ensure jurisdiction node exists, create dynamically if not exists"""
        jurisdiction_code = classification.get("jurisdiction_code", "unknown")
        jurisdiction_name = classification.get("jurisdiction_name", "Unknown")
        jurisdiction_level_str = classification.get("jurisdiction_level", "reference")
        parent_code = classification.get("parent_jurisdiction")
        
        # If jurisdiction already exists, return directly
        if jurisdiction_code in self.jurisdictions:
            return jurisdiction_code
        
        print(f"Found new jurisdiction: {jurisdiction_code} ({jurisdiction_name})")
        
        # Determine jurisdiction level
        try:
            jurisdiction_level = JurisdictionLevel(jurisdiction_level_str)
        except ValueError:
            print(f"Unknown jurisdiction level: {jurisdiction_level_str}, using default value reference")
            jurisdiction_level = JurisdictionLevel.REFERENCE
        
        # Ensure parent jurisdiction exists
        if parent_code and parent_code not in self.jurisdictions:
            print(f"Parent jurisdiction {parent_code} does not exist, attempting to create...")
            self._create_default_parent_jurisdiction(parent_code)
        
        # Create new jurisdiction node
        new_jurisdiction = JurisdictionNode(
            id=jurisdiction_code,
            name=jurisdiction_name,
            level=jurisdiction_level,
            parent_id=parent_code,
            metadata={
                "auto_created": True,
                "creation_timestamp": time.time(),
                "source_classification": classification
            }
        )
        
        # Add to system
        self.jurisdictions[jurisdiction_code] = new_jurisdiction
        
        # Update parent-child relationships
        if parent_code and parent_code in self.jurisdictions:
            self.jurisdictions[parent_code].children_ids.append(jurisdiction_code)
        
        # Add to graph
        self.graph.add_node(jurisdiction_code, node_type="jurisdiction", data=new_jurisdiction)
        if parent_code and parent_code in self.jurisdictions:
            self.graph.add_edge(parent_code, jurisdiction_code, relationship="governs")
        
        print(f"Successfully created jurisdiction: {jurisdiction_name} -> Parent: {parent_code or 'None'}")
        return jurisdiction_code

    def clean_text(self, text: str) -> str:
        """Clean text content"""
        print("Starting text cleaning...")
        
        # Remove headers and footers
        text = re.sub(r'-{5,}\s*Page\s+\d+\s*-{5,}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'HB\s+\d+\s*\n\s*\d{4}', '', text)
        text = re.sub(r'CODING:\s*Words stricken.*?additions\.', '', text)
        
        # Fix space-separated words
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2\3', text)
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2', text)
        
        # Merge line-broken sentences
        text = re.sub(r'([a-z,;:])\s*\n\s*([a-z])', r'\1 \2', text)
        
        # Normalize whitespace characters
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)
        
        print("Text cleaning complete")
        return text.strip()

    def create_uniform_chunks(self, text: str) -> List[str]:
        """Use unified RecursiveCharacterTextSplitter for chunking"""
        print("Starting unified chunking...")
        chunks = self.text_splitter.split_text(text)
        valid_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        print(f"Chunking complete, generated {len(valid_chunks)} valid chunks")
        return valid_chunks
    
    def process_document(self, text_content: str, file_path: str) -> str:
        """Complete document processing workflow: clean → classify → build graph → chunk → vectorize"""
        print(f"\nStarting document processing: {file_path}")
        
        # Step 1: Clean text
        cleaned_text = self.clean_text(text_content)
        
        # Step 2: Use large model to identify jurisdiction and document type
        print("Using large model for document classification...")
        classification = self.classifier.classify_document(cleaned_text, file_path)
        
        print(f"Classification results:")
        print(f"  - Jurisdiction: {classification.get('jurisdiction_code')} ({classification.get('jurisdiction_name')})")
        print(f"  - Level: {classification.get('jurisdiction_level')}")
        print(f"  - Parent: {classification.get('parent_jurisdiction')}")
        print(f"  - Type: {classification.get('document_type')}")
        print(f"  - Confidence: {classification.get('confidence', 0):.2f}")
        
        jurisdiction_code = classification.get('jurisdiction_code')
        document_type = classification.get('document_type', '').lower()
        
        # For reference documents like glossaries, if jurisdiction is empty or invalid, auto-classify to reference
        if (not jurisdiction_code or jurisdiction_code == "None" or 
            'terminology' in document_type or 'reference' in document_type or
            classification.get('jurisdiction_level') == 'reference'):
            print("Detected reference document/glossary, auto-classifying to reference jurisdiction")
            classification['jurisdiction_code'] = 'reference'
            classification['jurisdiction_name'] = 'Reference Documents'
            classification['jurisdiction_level'] = 'reference'
            classification['parent_jurisdiction'] = None

        # Step 3: Ensure jurisdiction node exists (dynamic creation)
        jurisdiction_id = self.ensure_jurisdiction_exists(classification)
        
        # Step 4: Generate document ID and create document object
        doc_id = hashlib.md5(f"{file_path}_{cleaned_text[:100]}".encode()).hexdigest()[:12]
        
        title = classification.get("title", Path(file_path).stem)
        if classification.get("bill_number"):
            title = f"{classification['bill_number']} - {title}"
        
        document = LegalDocument(
            id=doc_id,
            title=title,
            content=cleaned_text,
            jurisdiction_id=jurisdiction_id,
            document_type=classification.get("document_type", "Unknown"),
            metadata={
                "source_path": file_path,
                "classification_confidence": classification.get("confidence", 0.5),
                "full_classification": classification
            }
        )
        
        # Step 5: Unified chunking
        document.chunks = self.create_uniform_chunks(cleaned_text)
        print(f"Generated {len(document.chunks)} document chunks")
        
        # Step 6: Vectorization
        print("Starting vectorization...")
        if document.chunks:
            chunk_embeddings = []
            for i, chunk in enumerate(document.chunks):
                embedding = self.embedding_model.embed_query(chunk)
                chunk_embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"Completed vectorization of {i + 1}/{len(document.chunks)} chunks")
            
            document.chunk_embeddings = np.array(chunk_embeddings)
            self.document_embeddings[doc_id] = document.chunk_embeddings
        
        # Step 7: Add to graph data structure
        self.documents[doc_id] = document
        self.jurisdictions[jurisdiction_id].document_ids.append(doc_id)
        
        self.graph.add_node(doc_id, node_type="document", data=document)
        self.graph.add_edge(jurisdiction_id, doc_id, relationship="contains")
        
        print(f"Document processing complete: {title} -> Jurisdiction: {jurisdiction_id}")
        return doc_id
    
    def build_from_directory(self, directory_path: str):
        """Batch process documents from directory"""
        print(f"Starting document loading from directory: {directory_path}")
        
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            show_progress=True,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}
        )
        
        documents = loader.load()
        print(f"Found {len(documents)} document files")
        
        processed_docs = []
        for i, doc in enumerate(documents, 1):
            print(f"\nProcessing progress: {i}/{len(documents)}")
            try:
                doc_id = self.process_document(doc.page_content, doc.metadata.get('source', ''))
                processed_docs.append(doc_id)
                time.sleep(1)  # Avoid API calls too fast
                
            except Exception as e:
                print(f"Document processing failed: {doc.metadata.get('source', 'Unknown')}")
                print(f"Error: {e}")
                continue
        
        print(f"\nBatch processing complete, successfully processed {len(processed_docs)} documents")
        
        # Build document relationships
        print("Building document relationships...")
        self.build_document_relationships()
        
        return processed_docs
    
    def build_document_relationships(self):
        """Build relationships between documents"""
        doc_list = list(self.documents.values())
        
        for doc in doc_list:
            related_ids = []
            
            for other_doc in doc_list:
                if other_doc.id == doc.id:
                    continue
                
                # Check title references
                if doc.title in other_doc.content or other_doc.title in doc.content:
                    related_ids.append(other_doc.id)
                
                # Check bill number references
                if (doc.metadata.get('bill_number') and 
                    doc.metadata['bill_number'] in other_doc.content):
                    related_ids.append(other_doc.id)
            
            doc.related_document_ids = related_ids
            
            # Add relationship edges to graph
            for related_id in related_ids:
                self.graph.add_edge(doc.id, related_id, relationship="references")

    def search(self, query: str, jurisdiction_id: Optional[str] = None, 
               top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search relevant legal documents"""
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Determine search scope
        if jurisdiction_id:
            applicable_doc_ids = self.get_applicable_laws(jurisdiction_id)
        else:
            applicable_doc_ids = list(self.documents.keys())
        
        # Calculate similarity
        results = []
        for doc_id in applicable_doc_ids:
            if doc_id not in self.document_embeddings:
                continue
            
            doc = self.documents[doc_id]
            embeddings = self.document_embeddings[doc_id]
            
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            for i, (chunk, sim) in enumerate(zip(doc.chunks, similarities)):
                results.append((
                    chunk,
                    float(sim),
                    {
                        'document_id': doc_id,
                        'document_title': doc.title,
                        'jurisdiction': self.jurisdictions[doc.jurisdiction_id].name,
                        'document_type': doc.document_type,
                        'chunk_index': i,
                        **doc.metadata
                    }
                ))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_applicable_laws(self, jurisdiction_id: str) -> List[str]:
        """Get all laws applicable to a specific jurisdiction"""
        applicable_doc_ids = set()
        
        if jurisdiction_id not in self.jurisdictions:
            return []
        
        # Get documents from current jurisdiction
        applicable_doc_ids.update(self.jurisdictions[jurisdiction_id].document_ids)
        
        # Recursively get documents from parent jurisdictions
        current_id = jurisdiction_id
        visited = set()
        
        while current_id in self.jurisdictions and current_id not in visited:
            visited.add(current_id)
            parent_id = self.jurisdictions[current_id].parent_id
            
            if parent_id and parent_id in self.jurisdictions:
                applicable_doc_ids.update(self.jurisdictions[parent_id].document_ids)
                current_id = parent_id
            else:
                break
        
        return list(applicable_doc_ids)

    def visualize_jurisdiction_creation_stats(self):
        """Display jurisdiction creation statistics"""
        base_nodes = [j for j in self.jurisdictions.values() 
                     if j.metadata.get("is_root_node", False)]
        auto_created = [j for j in self.jurisdictions.values() 
                       if j.metadata.get("auto_created", False)]
        
        print(f"\n=== Jurisdiction Creation Statistics ===")
        print(f"Base root nodes: {len(base_nodes)} nodes")
        for node in base_nodes:
            child_count = len(node.children_ids)
            doc_count = len(node.document_ids)
            print(f"  - {node.name}: {child_count} children, {doc_count} documents")
        
        print(f"\nDynamically created nodes: {len(auto_created)} nodes")
        creation_types = {}
        for node in auto_created:
            node_type = node.metadata.get("type", "unknown")
            creation_types[node_type] = creation_types.get(node_type, 0) + 1
            
            parent_name = (self.jurisdictions[node.parent_id].name 
                          if node.parent_id and node.parent_id in self.jurisdictions 
                          else "None")
            print(f"  - {node.name} [{node.level.value}] -> {parent_name}")
        
        if creation_types:
            print(f"\nCreation type distribution:")
            for creation_type, count in creation_types.items():
                print(f"  - {creation_type}: {count} nodes")

    def save(self, base_path: str = "./legal_graph_db"):
        """Save graph data - using only JSON and Pickle formats"""
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True)
        
        # Save jurisdiction structure
        jurisdictions_data = {
            jur_id: jur.to_dict() 
            for jur_id, jur in self.jurisdictions.items()
        }
        with open(base_path / "jurisdictions.json", 'w', encoding='utf-8') as f:
            json.dump(jurisdictions_data, f, ensure_ascii=False, indent=2)
        
        # Save document data
        documents_data = {
            doc_id: doc.to_dict()
            for doc_id, doc in self.documents.items()
        }
        with open(base_path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        # Save graph structure
        graph_data = {
            'nodes': list(self.graph.nodes()),
            'edges': [(u, v, data) for u, v, data in self.graph.edges(data=True)]
        }
        with open(base_path / "graph_structure.json", 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        # Save vector embeddings - using Pickle
        with open(base_path / "embeddings.pkl", 'wb') as f:
            pickle.dump(self.document_embeddings, f)
        
        # Save configuration information
        config_data = {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "max_chunk_size": self.max_chunk_size,
            "overlap_size": self.overlap_size,
            "version": "dynamic_v1.0"
        }
        with open(base_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Graph data saved to: {base_path}")
        print("Storage format: JSON (metadata) + Pickle (vector embeddings)")

    def load(self, base_path: str = "./legal_graph_db"):
        """Load graph data"""
        base_path = Path(base_path)
        
        # Load configuration
        config_path = base_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.max_chunk_size = config.get("max_chunk_size", 800)
            self.overlap_size = config.get("overlap_size", 100)
        
        # Load jurisdiction structure
        with open(base_path / "jurisdictions.json", 'r', encoding='utf-8') as f:
            jurisdictions_data = json.load(f)
        
        self.jurisdictions = {}
        for jur_id, jur_data in jurisdictions_data.items():
            jur = JurisdictionNode(
                id=jur_data['id'],
                name=jur_data['name'],
                level=JurisdictionLevel(jur_data['level']),
                parent_id=jur_data.get('parent_id'),
                children_ids=jur_data.get('children_ids', []),
                document_ids=jur_data.get('document_ids', []),
                metadata=jur_data.get('metadata', {})
            )
            self.jurisdictions[jur_id] = jur
        
        # Load document data
        with open(base_path / "documents.json", 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
        
        self.documents = {}
        for doc_id, doc_data in documents_data.items():
            doc = LegalDocument(
                id=doc_id,
                title=doc_data['title'],
                content=doc_data['content'],
                jurisdiction_id=doc_data['jurisdiction_id'],
                document_type=doc_data['document_type'],
                chunks=doc_data.get('chunks', []),
                metadata=doc_data.get('metadata', {}),
                related_document_ids=doc_data.get('related_document_ids', [])
            )
            self.documents[doc_id] = doc
        
        # Load graph structure
        with open(base_path / "graph_structure.json", 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        self.graph = nx.DiGraph()
        for node in graph_data['nodes']:
            if node in self.jurisdictions:
                self.graph.add_node(node, node_type="jurisdiction", 
                                  data=self.jurisdictions[node])
            elif node in self.documents:
                self.graph.add_node(node, node_type="document",
                                  data=self.documents[node])
        
        for u, v, data in graph_data['edges']:
            self.graph.add_edge(u, v, **data)
        
        # Load vector embeddings
        with open(base_path / "embeddings.pkl", 'rb') as f:
            self.document_embeddings = pickle.load(f)
        
        print(f"Graph data loaded from {base_path}")

    def visualize_graph_stats(self):
        """Visualize graph statistics"""
        print("\n=== Legal Graph Statistics ===")
        print(f"Jurisdiction nodes: {len(self.jurisdictions)}")
        print(f"Document nodes: {len(self.documents)}")
        print(f"Total nodes in graph: {self.graph.number_of_nodes()}")
        print(f"Total edges in graph: {self.graph.number_of_edges()}")
        print("Storage format: JSON (metadata) + Pickle (vector embeddings)")
        
        print("\n=== Legal System Structure ===")
        # Display system structures
        systems = {"reference": [], "eu": [], "usa": [], "other": []}
        
        for jur_id, jur in self.jurisdictions.items():
            if jur_id == "reference":
                systems["reference"].append(jur)
            elif jur.parent_id == "eu" or jur_id == "eu":
                systems["eu"].append(jur)
            elif jur.parent_id == "usa" or jur_id == "usa":
                systems["usa"].append(jur)
            else:
                systems["other"].append(jur)
        
        for system_name, jurs in systems.items():
            if not jurs:
                continue
                
            system_display_names = {
                "reference": "Reference Document System",
                "eu": "European Union Legal System", 
                "usa": "United States Legal System",
                "other": "Other Jurisdiction Systems"
            }
            
            print(f"\n{system_display_names.get(system_name, system_name)}:")
            
            # Build tree display
            def print_tree(jur_id, level=1):
                if jur_id not in self.jurisdictions:
                    return
                jur = self.jurisdictions[jur_id]
                doc_count = len(jur.document_ids)
                indent = "  " * level
                doc_info = f" ({doc_count} documents)" if doc_count > 0 else ""
                auto_flag = " [auto-created]" if jur.metadata.get("auto_created") else ""
                print(f"{indent}├─ {jur.name} [{jur.level.value}]{doc_info}{auto_flag}")
                
                for child_id in jur.children_ids:
                    print_tree(child_id, level + 1)
            
            # Find root nodes for the system
            if system_name in ["reference", "eu", "usa"]:
                print_tree(system_name)
            else:
                # Display root nodes of other systems
                root_nodes = [j for j in jurs if j.parent_id is None or j.parent_id not in self.jurisdictions]
                for root in root_nodes:
                    print_tree(root.id)
        
        print("\n=== Document Type Distribution ===")
        doc_types = {}
        for doc in self.documents.values():
            doc_type = doc.document_type
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{doc_type}: {count} documents")


def build_dynamic_legal_graph_rag(knowledge_dir: str = "knowledge",
                                 qwen_api_url: str = None,
                                 qwen_api_key: str = None):
    """Build Legal Graph RAG system supporting dynamic jurisdiction expansion"""
    
    print("=== Building Dynamic Legal Graph RAG System ===")
    print("Feature: Automatic recognition and creation of new jurisdictions")
    
    # Use dynamic expansion version
    graph_rag = DynamicLegalGraphRAG(
        embedding_model_name="BAAI/bge-base-en-v1.5",
        max_chunk_size=300,
        overlap_size=50,
        qwen_api_url=qwen_api_url,
        qwen_api_key=qwen_api_key
    )
    
    processed_docs = graph_rag.build_from_directory(knowledge_dir)
    
    # Display statistics
    graph_rag.visualize_graph_stats()
    graph_rag.visualize_jurisdiction_creation_stats()
    
    # Save graph
    graph_rag.save("./dynamic_legal_graph_db")
    
    print(f"\n=== Dynamic Legal Graph RAG System Build Complete! Processed {len(processed_docs)} documents ===")
    
    return graph_rag


def main():
    """Main function - Build dynamic Graph RAG knowledge base"""
    knowledge_dir = "knowledge"
    qwen_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    qwen_api_key = None
    
    print("=== Dynamic Graph RAG Knowledge Base Builder ===")
    print(f"Knowledge base directory: {knowledge_dir}")
    print(f"API URL: {qwen_api_url}")
    print("Storage format: JSON + Pickle")
    print("Feature: Dynamic jurisdiction expansion")
    
    # Check if knowledge base directory exists
    if not Path(knowledge_dir).exists():
        print(f"Error: Knowledge base directory '{knowledge_dir}' does not exist")
        print(f"Please create directory and add txt document files")
        return
    
    # Build dynamic Graph RAG knowledge base
    try:
        graph_rag = build_dynamic_legal_graph_rag(
            knowledge_dir=knowledge_dir,
            qwen_api_url=qwen_api_url,
            qwen_api_key=qwen_api_key
        )
        
        print("\n=== Dynamic Knowledge Base Build Complete ===")
        print("Files saved to: ./dynamic_legal_graph_db/")
        print("Included files:")
        print("  - jurisdictions.json (jurisdiction information)")
        print("  - documents.json (document metadata)")  
        print("  - graph_structure.json (graph structure)")
        print("  - embeddings.pkl (vector embeddings)")
        print("  - config.json (configuration information)")
        
    except Exception as e:
        print(f"Error occurred during build process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()