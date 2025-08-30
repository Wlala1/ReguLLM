import re
import json
from typing import Dict, List, Tuple, Optional, Any
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load environment variables
load_dotenv()


class JargonTranslator:
    """Jargon Translator - Load terminology from Graph RAG knowledge base"""
    
    def __init__(self, graph_db_path: str = None):
        self.graph_db_path = graph_db_path
        self.jargon_dict = {}
        if graph_db_path:
            self._load_jargon_from_graph_db()
    
    def _load_jargon_from_graph_db(self):
        """Load terminology from Graph RAG JSON files"""
        try:
            print("Loading terminology from Graph RAG database...")
            
            # Load documents.json
            documents_path = Path(self.graph_db_path) / "documents.json"
            if not documents_path.exists():
                print(f"Warning: Document file not found: {documents_path}")
                return
            
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            print(f"Found {len(documents_data)} documents, searching for terminology...")
            
            # Search for terminology documents
            terminology_found = False
            for doc_id, doc_data in documents_data.items():
                content = doc_data.get("content", "")
                doc_type = doc_data.get("document_type", "")
                title = doc_data.get("title", "")
                
                # Check if it's a terminology table
                if (doc_type == "Terminology Table" or 
                    "terminology" in title.lower() or
                    "术语" in content or  # Chinese term detection
                    self._looks_like_terminology(content)):
                    
                    print(f"Found terminology document: {title}")
                    print(f"Content preview: {content[:200]}...")
                    
                    # Extract terms
                    extracted_jargon = self._extract_jargon_comprehensive(content)
                    
                    if extracted_jargon:
                        print(f"Successfully extracted {len(extracted_jargon)} terms:")
                        for jargon, definition in extracted_jargon.items():
                            self.jargon_dict[jargon] = definition
                            print(f"  {jargon}: {definition}")
                        terminology_found = True
                        break
            
            if not terminology_found:
                print("Warning: No terminology found, using default terms")
                
            else:
                print(f"Terminology loading complete, total {len(self.jargon_dict)} terms")
                        
        except Exception as e:
            print(f"Terminology loading failed: {e}")
            
    
    def _looks_like_terminology(self, content: str) -> bool:
        """Determine if content looks like terminology"""
        # Check for multiple uppercase abbreviations with colon definitions
        abbreviation_pattern = r'\b[A-Z]{2,}\s*[:-]'
        matches = re.findall(abbreviation_pattern, content)
        return len(matches) >= 3  # At least 3 abbreviation definitions
    
    
    def _extract_jargon_comprehensive(self, content: str) -> Dict[str, str]:
        """Comprehensive jargon term extraction"""
        jargon_dict = {}
        
        # Multiple matching patterns
        patterns = [
            # Standard format: ASL: Age-sensitive logic
            r'([A-Z]{2,}|[A-Z][a-z]+(?:[A-Z][a-z]*)*)\s*[:-]\s*([^\n\r]+)',
            # Parenthetical format: ASL (Age-sensitive logic)
            r'([A-Z]{2,})\s*\(([^)]+)\)',
            # Definition format: ASL means Age-sensitive logic  
            r'([A-Z]{2,})\s+(?:means?|is|stands for|refers to)\s+([^\n\r\.]+)',
            # Quoted format: "ASL": Age-sensitive logic
            r'["\']([A-Z]{2,})["\']?\s*[:-]\s*([^\n\r]+)',
            # Hyphen format: ASL - Age-sensitive logic
            r'([A-Z]{2,})\s*[-–—]\s*([^\n\r]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            for jargon, definition in matches:
                jargon = jargon.strip()
                definition = definition.strip().rstrip('.,;:|')
                
                # Quality filtering
                if (len(jargon) >= 2 and len(definition) > 5 and 
                    len(definition) < 200 and
                    not definition.isupper() and
                    jargon.isupper()):  # Ensure term is uppercase
                    jargon_dict[jargon] = definition
        
        return jargon_dict
    
    def translate_jargon(self, text: str) -> Tuple[str, List[str]]:
        """Translate jargon in text"""
        if not self.jargon_dict:
            return text, []
            
        translated_text = text
        found_jargons = []
        
        # Sort by length to avoid partial matching issues
        sorted_jargon = sorted(self.jargon_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        for jargon, definition in sorted_jargon:
            pattern = r'\b' + re.escape(jargon) + r'\b'
            if re.search(pattern, translated_text, re.IGNORECASE):
                found_jargons.append(jargon)
                replacement = f"{jargon} ({definition})"
                translated_text = re.sub(pattern, replacement, translated_text, flags=re.IGNORECASE)
        
        return translated_text, found_jargons


class OptimizedGraphRAGRetriever:
    """Optimized Graph RAG Retriever"""
    
    def __init__(self, graph_db_path: str):
        self.graph_db_path = Path(graph_db_path)
        self.documents = {}
        self.jurisdictions = {}
        self.document_embeddings = {}
        self.embedding_model = None
        self._load_graph_data()
    
    def _load_graph_data(self):
        """Load Graph RAG data"""
        try:
            print(f"Loading Graph RAG data from: {self.graph_db_path}")
            
            # Load document data
            documents_path = self.graph_db_path / "documents.json"
            if documents_path.exists():
                with open(documents_path, 'r', encoding='utf-8') as f:
                    documents_data = json.load(f)
                self.documents = documents_data
                print(f"✓ Loaded {len(documents_data)} documents")
            else:
                raise FileNotFoundError(f"Document file not found: {documents_path}")
            
            # Load jurisdiction data
            jurisdictions_path = self.graph_db_path / "jurisdictions.json"  
            if jurisdictions_path.exists():
                with open(jurisdictions_path, 'r', encoding='utf-8') as f:
                    self.jurisdictions = json.load(f)
                print(f"✓ Loaded {len(self.jurisdictions)} jurisdictions")
            
            # Load vector embeddings
            embeddings_path = self.graph_db_path / "embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, 'rb') as f:
                    self.document_embeddings = pickle.load(f)
                print(f"✓ Loaded embeddings for {len(self.document_embeddings)} documents")
                
                # Initialize embedding model
                self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
                print("✓ Embedding model initialization complete")
            else:
                print("⚠️ Embedding file not found, will use keyword search")
            
        except Exception as e:
            print(f"Failed to load Graph RAG data: {e}")
            raise e
    
    def similarity_search(self, query: str, k: int = 5, jurisdictions: List[str] = None) -> List[Dict]:
        """Optimized semantic similarity search"""
        if not self.embedding_model or not self.document_embeddings:
            print("Using keyword search mode")
            return self._keyword_search(query, k, jurisdictions)
        
        try:
            print(f"Executing vector search: {query}")
            
            # Query vectorization
            query_embedding = self.embedding_model.embed_query(query)
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            results = []
            
            # Filter target documents
            if jurisdictions:
                target_doc_ids = self._filter_documents_by_jurisdiction(jurisdictions)
                print(f"Filtered to {len(target_doc_ids)} documents based on jurisdictions {jurisdictions}")
            else:
                target_doc_ids = list(self.documents.keys())
                print(f"Searching all {len(target_doc_ids)} documents")
            
            # Calculate similarity
            for doc_id in target_doc_ids:
                if doc_id not in self.document_embeddings:
                    continue
                
                doc_data = self.documents[doc_id]
                doc_embeddings = self.document_embeddings[doc_id]
                chunks = doc_data.get('chunks', [])
                
                if len(chunks) == 0 or doc_embeddings.shape[0] == 0:
                    continue
                
                # Batch calculate similarity
                similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
                
                # Collect high-quality results
                for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
                    if similarity > 0.1:  # Similarity threshold
                        results.append({
                            'page_content': chunk,
                            'similarity_score': float(similarity),
                            'metadata': {
                                'document_id': doc_id,
                                'document_title': doc_data.get('title', 'Unknown'),
                                'document_type': doc_data.get('document_type', 'Unknown'),
                                'jurisdiction_id': doc_data.get('jurisdiction_id', 'Unknown'),
                                'chunk_index': i,
                                'source': doc_data.get('metadata', {}).get('source_path', 'Graph RAG DB'),
                                'classification_confidence': doc_data.get('metadata', {}).get('classification_confidence', 0.5)
                            }
                        })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = results[:k]
            
            print(f"Found {len(final_results)} high-quality results")
            return final_results
            
        except Exception as e:
            print(f"Vector search failed, switching to keyword search: {e}")
            return self._keyword_search(query, k, jurisdictions)
    
    def _filter_documents_by_jurisdiction(self, jurisdictions: List[str]) -> List[str]:
        """Filter documents by jurisdiction (including hierarchical relationships)"""
        filtered_doc_ids = set()
        
        for jurisdiction in jurisdictions:
            # Documents directly containing this jurisdiction
            if jurisdiction in self.jurisdictions:
                jur_data = self.jurisdictions[jurisdiction]
                doc_ids = jur_data.get('document_ids', [])
                filtered_doc_ids.update(doc_ids)
                
                # Documents containing parent jurisdiction (legal hierarchy inheritance)
                parent_id = jur_data.get('parent_id')
                if parent_id and parent_id in self.jurisdictions:
                    parent_doc_ids = self.jurisdictions[parent_id].get('document_ids', [])
                    filtered_doc_ids.update(parent_doc_ids)
        
        return list(filtered_doc_ids)
    
    def _keyword_search(self, query: str, k: int = 5, jurisdictions: List[str] = None) -> List[Dict]:
        """Optimized keyword search"""
        print(f"Executing keyword search: {query}")
        
        # Preprocess query terms
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        results = []
        
        # Filter documents
        if jurisdictions:
            target_doc_ids = self._filter_documents_by_jurisdiction(jurisdictions)
        else:
            target_doc_ids = list(self.documents.keys())
        
        print(f"Searching in {len(target_doc_ids)} documents")
        
        for doc_id in target_doc_ids:
            doc_data = self.documents[doc_id]
            chunks = doc_data.get('chunks', [])
            
            # Search in chunks
            if chunks:
                for i, chunk in enumerate(chunks):
                    chunk_lower = chunk.lower()
                    
                    # Calculate keyword matching score
                    exact_matches = sum(1 for word in query_words if word in chunk_lower)
                    partial_matches = sum(0.5 for word in query_words 
                                        if any(word in token for token in chunk_lower.split()))
                    
                    total_score = exact_matches + partial_matches
                    
                    if total_score > 0:
                        results.append({
                            'page_content': chunk,
                            'keyword_score': total_score,
                            'metadata': {
                                'document_id': doc_id,
                                'document_title': doc_data.get('title', 'Unknown'),
                                'document_type': doc_data.get('document_type', 'Unknown'), 
                                'jurisdiction_id': doc_data.get('jurisdiction_id', 'Unknown'),
                                'chunk_index': i,
                                'source': doc_data.get('metadata', {}).get('source_path', 'Graph RAG DB'),
                                'classification_confidence': doc_data.get('metadata', {}).get('classification_confidence', 0.5)
                            }
                        })
            else:
                # Search in entire document
                content = doc_data.get('content', '').lower()
                exact_matches = sum(1 for word in query_words if word in content)
                
                if exact_matches > 0:
                    # Extract relevant excerpt containing keywords
                    content_excerpt = self._extract_relevant_excerpt(
                        doc_data.get('content', ''), query_words[0] if query_words else query
                    )
                    
                    results.append({
                        'page_content': content_excerpt,
                        'keyword_score': exact_matches,
                        'metadata': {
                            'document_id': doc_id,
                            'document_title': doc_data.get('title', 'Unknown'),
                            'document_type': doc_data.get('document_type', 'Unknown'),
                            'jurisdiction_id': doc_data.get('jurisdiction_id', 'Unknown'),
                            'source': doc_data.get('metadata', {}).get('source_path', 'Graph RAG DB'),
                            'classification_confidence': doc_data.get('metadata', {}).get('classification_confidence', 0.5)
                        }
                    })
        
        # Sort and return
        score_key = 'keyword_score' if results and 'keyword_score' in results[0] else 'similarity_score'
        results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
        
        final_results = results[:k]
        print(f"Keyword search found {len(final_results)} results")
        return final_results
    
    def _extract_relevant_excerpt(self, content: str, keyword: str, max_length: int = 800) -> str:
        """Extract relevant excerpt containing keywords"""
        keyword_lower = keyword.lower()
        content_lower = content.lower()
        
        # Find keyword position
        pos = content_lower.find(keyword_lower)
        if pos == -1:
            return content[:max_length]
        
        # Extract content around keyword
        start = max(0, pos - max_length // 2)
        end = min(len(content), pos + max_length // 2)
        
        excerpt = content[start:end]
        
        # Clean truncated sentences
        if start > 0:
            first_period = excerpt.find('.')
            if first_period > 0:
                excerpt = excerpt[first_period + 1:]
        
        if end < len(content):
            last_period = excerpt.rfind('.')
            if last_period > 0:
                excerpt = excerpt[:last_period + 1]
        
        return excerpt.strip()


class GeographicJurisdictionDetector:
    """Geographic Jurisdiction Detector"""
    
    def __init__(self, llm):
        self.llm = llm
        node = json.load(open("dynamic_legal_graph_db/graph_structure.json"))["nodes"]
        self.allowlist = [n for n in node if not re.search(r'\d', n)]
    
    def detect_jurisdictions(self, text: str) -> List[str]:
        """Detect jurisdictions involved in the text"""
        prompt = (
            "You are a world-knowledge jurisdiction extractor.\n"
            f"Allowed values (lowercase only): {json.dumps(self.allowlist)}\n"
            "Task: From the text, extract ALL jurisdictions that are mentioned or clearly implied.\n"
            "Rules:\n"
            "- Use ONLY items from the allowed list.\n"
            "- If a child jurisdiction is mentioned (e.g., a US state or an EU member state), "
            "  ALSO include its commonly accepted parent jurisdiction(s) **based on your own knowledge** "
            "  when those parents are present in the allowed list (e.g., 'california' -> 'usa', 'france' -> 'eu').\n"
            "- Do NOT infer children from a parent-only mention.\n"
            "- Normalize synonyms and language variants (e.g., 'united states'/'u.s.'/'america' -> 'usa'; "
            "'european union' -> 'eu'; 'holland' -> 'netherlands').\n"
            "- If an abbreviation is ambiguous (e.g., 'ca'), include it only if context clearly indicates "
            "  the intended jurisdiction in the allowed list; otherwise omit it.\n"
            "- Output ONLY a array of strings (e.g., [\"france\",\"eu\"]). No extra text.\n\n"
            f"Text:\n{text}"
        )
        resp = self.llm.invoke(prompt)
        arr = json.loads(resp.content)               # Parse
        arr = [x for x in arr if x in self.allowlist]         # Client-side allowlist filtering
        detected_jurisdictions = list(dict.fromkeys(arr))               # Remove duplicates and preserve order
        
        state_to_country = {
            "california": "usa", "utah": "usa", "florida": "usa", "texas": "usa",
            "germany": "eu", "france": "eu", "italy": "eu", "spain": "eu", "netherlands": "eu"
        }
        
        for jurisdiction in detected_jurisdictions.copy():
            if jurisdiction in state_to_country:
                parent = state_to_country[jurisdiction]
                if parent not in detected_jurisdictions:
                    detected_jurisdictions.append(parent)
        
        return detected_jurisdictions


class OptimizedLegalClassifier:
    """Optimized Legal Classifier - Pure Graph RAG Version"""
    
    def __init__(self, 
                 graph_db_path: str = "./dynamic_legal_graph_db",
                 model_name: str = "qwen-max-latest"):
        
        print("=== Initializing Optimized Legal Classifier (Pure Graph RAG Version) ===")
        
        # Check database path
        if not Path(graph_db_path).exists():
            raise FileNotFoundError(f"Graph RAG database not found: {graph_db_path}")
        
        # Initialize components
        print("1. Initializing Large Language Model...")
        self.llm = ChatTongyi(model=model_name, temperature=0.1)
        
        print("2. Loading Graph RAG Retriever...")
        self.retriever = OptimizedGraphRAGRetriever(graph_db_path)
        
        print("3. Initializing Jargon Translator...")
        self.jargon_translator = JargonTranslator(graph_db_path)
        
        print("4. Initializing Geographic Jurisdiction Detector...")
        self.geo_detector = GeographicJurisdictionDetector(self.llm)
        
        print("5. Initializing Prompt Templates...")
        self.parser = JsonOutputParser()
        self.prompt = self._create_enhanced_prompt()
        
        print("✓ Initialization complete!\n")
    
    def _create_enhanced_prompt(self) -> PromptTemplate:
        """Create enhanced prompt template"""
        template = """
You are a senior legal compliance analyst with Graph RAG legal knowledge graph search capabilities.

Task: Accurately classify feature characteristics into one of the following three categories. Your analysis must clearly link the feature to specific, quoted legal provisions from the provided evidence.
- "LegalRequirement"      (Legal/regulatory/compliance requirement enforcement)
- "BusinessDriven"        (Product strategy/experimentation/security choice; non-legally mandated requirements)
- "UnspecifiedNeedsHuman" (Unclear intent or missing/conflicting evidence)

Analysis Input:
Original Feature: {original_feature}

Translated Feature (with jargon explanations): {translated_feature}

Detected Jurisdictions: {detected_jurisdictions}

Found Jargon Terms: {found_jargons}

Graph RAG Searched Legal Evidence:
{context}

Decision Framework:
LegalRequirement (score 0-1):
+0.40 Context cites specific legal provisions + jurisdiction match
+0.20 Feature behavior clearly corresponds to legal requirements (e.g., age restrictions ↔ child protection laws)
+0.20 Geographic restrictions align with legal jurisdiction boundaries
+0.20 At least one credible legal provision citation

BusinessDriven (score 0-1):
+0.50 Clear business motivation: A/B testing, experimentation, performance optimization, growth
+0.30 Jurisdictions detected but lack supporting legal evidence
+0.20 Geographic differences primarily for product rollout strategy

UnspecifiedNeedsHuman: If LegalRequirement or BusinessDriven score is below 0.8, classify as this category. Score for this category is 1-LegalRequirement score or 1-BusinessDriven score.

Output Requirements:
{{
  "assessment": "LegalRequirement" | "BusinessDriven" | "UnspecifiedNeedsHuman",
  "needs_compliance_logic": true | false | null,
  "reasoning": "Evidence-based detailed analysis (≤200 words). If 'LegalRequirement', you MUST explicitly cite the specific regulation and article (e.g., 'Based on GDPR Art. 7...') from the evidence that mandates the feature's behavior.",
  "detected_jurisdictions": {detected_jurisdictions},
  "translated_jargon": {found_jargons},
  "jurisdictions": ["determined applicable jurisdictions"],
  "regulations": [
    {{
      "id": "regulation ID or null",
      "title": "regulation name", 
      "jurisdiction": "jurisdiction",
      "relevance": 0.0-1.0,
      "passages": [
        {{"quote": "The verbatim key provision (≤200 chars) from the legal evidence that directly mandates or influences the feature.", "source_id": "document ID"}}
      ],
      "decision": "Constrained" | "NotConstrained" | "Unclear",
      "reason": "Explain precisely how the quoted passage constrains this feature. You must reference the specific legal concept from the text (e.g., 'This passage on the right to erasure requires a data deletion mechanism')."
    }}
  ],
  "triggers": {{
    "legal": ["identified legal keywords"],
    "business": ["identified business keywords"],
    "ambiguity": ["ambiguous or conflicting statements"]
  }},
  "scores": {{
    "LegalRequirement": 0.0-1.0,
    "BusinessDriven": 0.0-1.0,
    "UnspecifiedNeedsHuman": 0.0-1.0
  }},
  "citations": ["cited source document IDs"],
  "confidence": 0.10-0.99
}}

Constraints:
- Only use actual legal evidence from context
- When assessment ≠ "LegalRequirement", regulations=[]
- Do not fabricate legal citations
- Jurisdiction codes must match database
"""
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "original_feature", "translated_feature", "detected_jurisdictions", 
                "found_jargons", "context"
            ]
        )
    
    def _search_legal_evidence(self, query: str, jurisdictions: List[str] = None) -> List[str]:
        """Search legal evidence"""
        try:
            print(f"Starting Graph RAG legal evidence search...")
            
            # Build diverse search queries
            search_queries = [query]
            
            # Jurisdiction-specific queries
            if jurisdictions:
                jurisdiction_specific = {
                    "utah": ["Utah Social Media Act", "minor curfew", "parental consent"],
                    "california": ["California SB976", "addictive feed", "teen protection"],
                    "florida": ["Florida minor protection", "harmful content"],
                    "eu": ["Digital Services Act", "DSA", "content moderation"],
                    "usa": ["NCMEC", "COPPA", "child protection", "federal reporting"]
                }
                
                for jurisdiction in jurisdictions:
                    if jurisdiction in jurisdiction_specific:
                        for term in jurisdiction_specific[jurisdiction]:
                            search_queries.append(f"{term}")
                            search_queries.append(f"{term} {query}")
            
            # Topic-related queries
            query_lower = query.lower()
            if any(term in query_lower for term in ["minor", "child", "underage", "youth"]):
                search_queries.extend(["child protection law", "minor safety", "parental consent"])
            
            if "feed" in query_lower or "personalization" in query_lower:
                search_queries.extend(["algorithmic feed", "personalized content", "addictive design"])
                
            if "curfew" in query_lower:
                search_queries.extend(["time restrictions", "access limitations"])
            
            # Execute searches
            all_results = []
            seen_docs = set()
            
            print(f"Executing {len(search_queries)} search queries")
            
            for i, search_query in enumerate(search_queries[:8]):  # Limit query count
                try:
                    print(f"  Query {i+1}: {search_query}")
                    docs = self.retriever.similarity_search(
                        search_query, k=3, jurisdictions=jurisdictions
                    )
                    
                    for doc in docs:
                        doc_id = doc['metadata'].get('document_id')
                        chunk_id = f"{doc_id}_{doc['metadata'].get('chunk_index', 0)}"
                        
                        # Avoid duplicates
                        if chunk_id in seen_docs:
                            continue
                        seen_docs.add(chunk_id)
                        
                        # Build result information
                        score_info = ""
                        if 'similarity_score' in doc:
                            score_info = f"Similarity: {doc['similarity_score']:.3f}"
                        elif 'keyword_score' in doc:
                            score_info = f"KeywordScore: {doc['keyword_score']}"
                        
                        source_info = f"[GraphRAG-{len(all_results)+1}] "
                        source_info += f"Query: {search_query[:30]}... | "
                        source_info += f"{score_info} | "
                        source_info += f"Doc: {doc['metadata'].get('document_title', 'Unknown')} | "
                        source_info += f"Type: {doc['metadata'].get('document_type', 'Unknown')} | "
                        source_info += f"Jurisdiction: {doc['metadata'].get('jurisdiction_id', 'Unknown')}"
                        
                        content = f"{source_info}\nContent: {doc['page_content'][:1000]}"
                        all_results.append(content)
                        
                        print(f"    ✓ Added result: {doc['metadata'].get('document_title', 'Unknown')[:30]}")
                        
                        if len(all_results) >= 8:
                            break
                    
                except Exception as e:
                    print(f"    Query failed: {e}")
                
                if len(all_results) >= 8:
                    break
            
            print(f"Graph RAG search complete, found {len(all_results)} legal evidence items")
            
            # If insufficient results, perform supplementary search
            if len(all_results) < 3:
                print("Insufficient results, performing supplementary search...")
                fallback_queries = ["law", "regulation", "requirement", "compliance"]
                for fallback_query in fallback_queries:
                    try:
                        docs = self.retriever.similarity_search(fallback_query, k=2)
                        for doc in docs[:1]:  # Only take one
                            content = f"[GraphRAG-Fallback] {fallback_query}\n{doc['page_content'][:800]}"
                            all_results.append(content)
                            print(f"  Added supplementary result: {doc['metadata'].get('document_title', 'Unknown')}")
                            if len(all_results) >= 5:
                                break
                    except:
                        continue
                    if len(all_results) >= 5:
                        break
            
            return all_results
            
        except Exception as e:
            print(f"Graph RAG search failed: {e}")
            import traceback
            traceback.print_exc()
            return [f"Search failed: {str(e)}"]
    
    def classify_feature(self, feature_description: str) -> Dict[str, Any]:
        """
        Complete feature classification workflow
        """
        print(f"\n{'='*60}")
        print(f"Starting analysis of feature: {feature_description[:80]}...")
        print(f"{'='*60}")
        
        # Step 1: Jargon identification and translation
        print("Step 1: Jargon identification and translation")
        translated_text, found_jargons = self.jargon_translator.translate_jargon(feature_description)
        
        if found_jargons:
            print(f"  ✓ Found jargon: {found_jargons}")
            print(f"  ✓ Translation result: {translated_text}")
        else:
            print("  - No jargon terms found")
        
        # Step 2: Geographic jurisdiction detection
        print("Step 2: Geographic jurisdiction detection")
        detected_jurisdictions = self.geo_detector.detect_jurisdictions(feature_description)
        if detected_jurisdictions:
            print(f"  ✓ Detected jurisdictions: {detected_jurisdictions}")
            print("Step 3: Graph RAG legal evidence search")
            search_query = f"{feature_description} legal requirements compliance regulation"
            
            legal_evidence = self._search_legal_evidence(search_query, detected_jurisdictions)
            print(f"  ✓ Found {len(legal_evidence)} legal evidence items")
        else:
            print("  - No specific jurisdictions detected")
            print("Step 3: Graph RAG legal evidence search")
            legal_evidence = []
            print("  - Skipping legal evidence search")
        
        # Step 4: Build context
        print("Step 4: Build context")
        context = "\n\n".join(legal_evidence) if legal_evidence else "NO_LEGAL_EVIDENCE_FOUND"
        
        # Step 5: LLM analysis and classification
        print("Step 5: Large Language Model analysis and classification")
        
        try:
            inputs = {
                "original_feature": feature_description,
                "translated_feature": translated_text,
                "detected_jurisdictions": detected_jurisdictions,
                "found_jargons": found_jargons,
                "context": context[:5000]  # Control context length
            }

            formatted_prompt = self.prompt.format(**inputs)
            response = self.llm.invoke(formatted_prompt)
            try:
                result = self.parser.parse(response)
            except Exception:
                # If failed, append JSON constraints and call again
                fix_prompt = (
                    f"{formatted_prompt}\n\n"
                    "Please strictly output **valid JSON only**: use double quotes, no comments, no extra text/Markdown code blocks; "
                    "use null for missing values, don't omit fields. Output JSON only."
                )
                response = self.llm.invoke(fix_prompt)
                result = self.parser.parse(response.content)
            
            # Add processing metadata
            result["processing_metadata"] = {
                "workflow_completed": True,
                "data_source": "Pure Graph RAG",
                "jargons_found": found_jargons,
                "jargons_translated": bool(found_jargons),
                "jurisdictions_detected": detected_jurisdictions,
                "legal_evidence_count": len(legal_evidence),
                "context_length": len(context),
                "embedding_search_available": bool(self.retriever.embedding_model),
                "search_method": "Vector" if self.retriever.embedding_model else "Keyword"
            }
            
            print(f"  ✓ Classification complete: {result.get('assessment', 'Unknown')}")
            print(f"  ✓ Confidence: {result.get('confidence', 'Unknown')}")
            if result.get('needs_compliance_logic') is not None:
                print(f"  ✓ Needs compliance logic: {result.get('needs_compliance_logic')}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Classification failed: {e}")
            return {
                "assessment": "UnspecifiedNeedsHuman",
                "needs_compliance_logic": None,
                "reasoning": f"Classification process error: {str(e)}",
                "detected_jurisdictions": detected_jurisdictions,
                "translated_jargon": found_jargons,
                "error": str(e),
                "processing_metadata": {
                    "workflow_completed": False,
                    "data_source": "Pure Graph RAG",
                    "error": str(e),
                    "jargons_found": found_jargons,
                    "jurisdictions_detected": detected_jurisdictions
                }
            }
    
    def batch_classify(self, feature_list: List[str]) -> List[Dict[str, Any]]:
        """Batch classify features"""
        results = []
        total = len(feature_list)
        
        print(f"Starting batch classification of {total} feature characteristics")
        
        for i, feature in enumerate(feature_list, 1):
            print(f"\nBatch progress: {i}/{total}")
            try:
                result = self.classify_feature(feature)
                results.append({
                    'index': i,
                    'input': feature,
                    'output': result,
                    'success': True
                })
            except Exception as e:
                print(f"Feature {i} classification failed: {e}")
                results.append({
                    'index': i,
                    'input': feature,
                    'output': {'error': str(e)},
                    'success': False
                })
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\nBatch classification complete: {success_count}/{total} successful")
        
        return results


def main():
    """Main function demonstration"""
    print("=== Optimized Legal Classification System - Pure Graph RAG Version ===\n")
    
    # Initialize classifier
    try:
        # Modify to your actual path
        graph_db_path = "./dynamic_legal_graph_db"
        
        classifier = OptimizedLegalClassifier(graph_db_path=graph_db_path)
        print("✓ Classifier initialization successful\n")
        
    except Exception as e:
        print(f"✗ Classifier initialization failed: {e}")
        print("\nPlease check:")
        print("1. Is the Graph RAG database path correct?")
        print("2. Have you run the first code to generate the database?")
        print("3. Does the database contain necessary files: documents.json, jurisdictions.json, embeddings.pkl?")
        return
    
    # Test data
    test_features = [
        {
            "name": "Copyright Enforcement Feature",
            "description": "Feature reads user location to enforce France's copyright rules"
        }
    ]
    
    # Execute classification tests
    results = []
    for i, test_case in enumerate(test_features):
        print(f"\n{'='*80}")
        print(f"Test Case {i+1}/{len(test_features)}: {test_case['name']}")
        print(f"{'='*80}")
        
        # Execute classification
        result = classifier.classify_feature(test_case['description'])
        
        # Save result
        test_result = {
            'case_name': test_case['name'],
            'input': test_case['description'],
            'output': result
        }
        results.append(test_result)
        
        # Display key results
        print(f"\n📋 Classification Result Summary:")
        print(f"  🏷️  Classification: {result.get('assessment', 'Unknown')}")
        print(f"  🔧 Needs compliance logic: {result.get('needs_compliance_logic', 'Unknown')}")
        print(f"  📊 Confidence: {result.get('confidence', 'Unknown')}")
        
        if result.get('detected_jurisdictions'):
            print(f"  🌍 Detected jurisdictions: {result.get('detected_jurisdictions')}")
        
        if result.get('translated_jargon'):
            print(f"  🔤 Found jargon: {result.get('translated_jargon')}")
        
        reasoning = result.get('reasoning', '')
        if reasoning:
            print(f"  💭 Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
        
        # Display processing metadata
        if result.get('processing_metadata'):
            metadata = result['processing_metadata']
            print(f"\n🔍 Processing Statistics:")
            print(f"  ✅ Workflow completed: {metadata.get('workflow_completed', False)}")
            print(f"  🔍 Search method: {metadata.get('search_method', 'Unknown')}")
            print(f"  📄 Legal evidence count: {metadata.get('legal_evidence_count', 0)}")
            print(f"  📝 Context length: {metadata.get('context_length', 0)}")
        
        print(f"\n{'-'*60}")
    
    # Save results to file
    try:
        output_file = 'optimized_graph_rag_classification_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"📊 Total processed: {len(results)} test cases")
        
        # Count classification results
        assessments = [r['output'].get('assessment', 'Error') for r in results]
        from collections import Counter
        assessment_counts = Counter(assessments)
        
        print(f"\n📈 Classification Statistics:")
        for assessment, count in assessment_counts.items():
            print(f"  {assessment}: {count} cases")
        
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")


def interactive_mode():
    """Interactive mode"""
    try:
        classifier = OptimizedLegalClassifier()
        print("\n🔍 Interactive Legal Feature Classifier")
        print("💡 Enter 'quit' to exit the program")
        print("📍 Available jurisdictions: usa, california, utah, florida, texas, eu, germany, france, italy, spain, netherlands, reference")
        
        while True:
            print(f"\n{'-'*50}")
            try:
                feature_input = input("🎯 Please enter feature description: ").strip()
                if feature_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not feature_input:
                    print("⚠️ Please enter a valid feature description")
                    continue
                
                # Execute classification
                result = classifier.classify_feature(feature_input)
                
                # Display results
                print(f"\n📋 Classification Result:")
                print(f"  🏷️  {result.get('assessment', 'Unknown')}")
                print(f"  📊 Confidence: {result.get('confidence', 'N/A')}")
                print(f"  💭 Reasoning: {result.get('reasoning', 'N/A')}")
                
                if result.get('regulations'):
                    print(f"  ⚖️  Related regulations: {len(result['regulations'])} items")
                
            except KeyboardInterrupt:
                print("\n\n👋 Program interrupted, goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Processing error: {e}")
                
    except Exception as e:
        print(f"❌ Cannot start interactive mode: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        main()