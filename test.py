from brainex_memory import RuleBasedRAGAgent, RuleBasedReasoner

def demo_update_delete():
    """Test the update and delete functionality"""
    agent = RuleBasedRAGAgent()
    
    print("=== Update and Delete Demo ===\n")
    
    # Create a test document first
    print("1. Creating a test document:")
    reasoner = RuleBasedReasoner()
    save_result = reasoner.save_knowledge._run(
        name="Test CRUD Document",
        content="# Test Document\n\nThis is a test document for CRUD operations.\n\n## Original Content\nThis content will be updated.",
        topic="Testing",
        tags=["test", "crud"],
        source_query="test crud operations"
    )
    
    save_data = json.loads(save_result)
    if save_data.get("success"):
        test_doc_id = save_data["document_id"]
        print(f"✅ Test document created with ID: {test_doc_id}")
        
        print(f"\n{'-'*50}\n")
        
        # Test update
        print("2. Testing update operation:")
        updates = {
            "name": "Updated CRUD Test Document",
            "topic": "Advanced Testing",
            "tags": ["test", "crud", "updated"],
            "lecture_content": "# Updated Test Document\n\nThis document has been updated!\n\n## Updated Content\nThis is the new content after update.\n\n## Additional Section\nMore information added during update."
        }
        print(agent.update_document(test_doc_id, updates))
        
        print(f"\n{'-'*50}\n")
        
        # Test delete without confirmation
        print("3. Testing delete without confirmation:")
        print(agent.delete_document(test_doc_id))
        
        print(f"\n{'-'*50}\n")
        
        # Test delete with confirmation
        print("4. Testing delete with confirmation:")
        print(agent.delete_document(test_doc_id, confirm=True))
        
    else:
        print(f"❌ Failed to create test document: {save_data.get('error')}")

if __name__ == "__main__":
    # You can call either the original main() or the new demo
    # main()  # Original functionality
    demo_update_delete()  # Test new CRUD operations# Additional tools for updating and deleting documents in Firestore
# Add these to your existing test.py file

from typing import Dict, List, Optional, TypedDict, Annotated, Literal, Tuple
from datetime import datetime
import json
import re
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from google.cloud import firestore

#=================================================================
# Update Tool
#=================================================================

class UpdateDocumentInput(BaseModel):
    document_id: str = Field(description="The ID of the document to update")
    collection: str = Field(default="lecture_notes", description="Collection name (lecture_notes or resources)")
    updates: Dict = Field(description="Dictionary of fields to update with their new values")
    merge: bool = Field(default=True, description="Whether to merge with existing data or overwrite")

class UpdateDocument(BaseTool):
    name: str = "update_document"
    description: str = """Update an existing document in Firestore database. Can update lecture notes or resources."""
    args_schema = UpdateDocumentInput

    def _run(self, document_id: str, collection: str = "lecture_notes", updates: Dict = None, merge: bool = True) -> str:
        print(f"Updating document {document_id} in collection {collection}")
        
        if not updates:
            return json.dumps({"success": False, "error": "No updates provided"})
        
        try:
            db = firestore.Client()
            doc_ref = db.collection(collection).document(document_id)
            
            # Check if document exists
            doc = doc_ref.get()
            if not doc.exists:
                return json.dumps({
                    "success": False, 
                    "error": f"Document with ID {document_id} not found in collection {collection}"
                })
            
            # Add/update timestamp
            updates['updated_at'] = datetime.now().isoformat()
            
            # For lecture_notes, update key_concepts if content is updated
            if collection == "lecture_notes" and "lecture_content" in updates:
                key_concepts = self._extract_key_concepts(updates["lecture_content"])
                updates["key_concepts"] = key_concepts
                updates["content_length"] = len(updates["lecture_content"])
            
            # Perform the update
            doc_ref.set(updates, merge=merge)
            
            # Get updated document to return current state
            updated_doc = doc_ref.get().to_dict()
            
            return json.dumps({
                "success": True,
                "message": f"Document {document_id} updated successfully",
                "document_id": document_id,
                "updated_fields": list(updates.keys()),
                "updated_document": updated_doc
            })
            
        except Exception as e:
            return json.dumps({"success": False, "error": f"Failed to update document: {str(e)}"})
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Simple key concept extraction - same as in SaveKnowledge"""
        concepts = set()
        
        # Extract from markdown headers
        headers = re.findall(r'#+\s*([^#\n]+)', content)
        for header in headers:
            words = re.findall(r'\b[A-Za-z]{3,}\b', header)
            concepts.update(word.lower() for word in words if len(word) > 3)
        
        # Extract capitalized terms
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
        concepts.update(word.lower() for word in capitalized if len(word) > 3)
        
        # Remove common words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        concepts -= common_words
        
        return list(concepts)[:10]

#=================================================================
# Delete Tool
#=================================================================

class DeleteDocumentInput(BaseModel):
    document_id: str = Field(description="The ID of the document to delete")
    collection: str = Field(default="lecture_notes", description="Collection name (lecture_notes or resources)")
    confirm: bool = Field(default=False, description="Confirmation flag to prevent accidental deletions")

class DeleteDocument(BaseTool):
    name: str = "delete_document"
    description: str = """Delete a document from Firestore database. Requires confirmation to prevent accidental deletions."""
    args_schema = DeleteDocumentInput

    def _run(self, document_id: str, collection: str = "lecture_notes", confirm: bool = False) -> str:
        print(f"Attempting to delete document {document_id} from collection {collection}")
        
        if not confirm:
            return json.dumps({
                "success": False, 
                "error": "Deletion requires explicit confirmation. Set confirm=True to proceed.",
                "warning": "This action cannot be undone."
            })
        
        try:
            db = firestore.Client()
            doc_ref = db.collection(collection).document(document_id)
            
            # Check if document exists and get its data before deletion
            doc = doc_ref.get()
            if not doc.exists:
                return json.dumps({
                    "success": False, 
                    "error": f"Document with ID {document_id} not found in collection {collection}"
                })
            
            # Store document data for confirmation
            doc_data = doc.to_dict()
            doc_name = doc_data.get('name', 'Unnamed Document')
            
            # Delete the document
            doc_ref.delete()
            
            return json.dumps({
                "success": True,
                "message": f"Document '{doc_name}' (ID: {document_id}) deleted successfully from {collection}",
                "deleted_document": {
                    "id": document_id,
                    "name": doc_name,
                    "collection": collection,
                    "deleted_at": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            return json.dumps({"success": False, "error": f"Failed to delete document: {str(e)}"})

#=================================================================
# List Documents Tool (Helper for finding IDs)
#=================================================================

class ListDocumentsInput(BaseModel):
    collection: str = Field(default="lecture_notes", description="Collection name to list documents from")
    limit: int = Field(default=20, description="Maximum number of documents to return")
    order_by: str = Field(default="updated_at", description="Field to order by")
    order_direction: str = Field(default="desc", description="Order direction (asc or desc)")

class ListDocuments(BaseTool):
    name: str = "list_documents"
    description: str = """List documents from a Firestore collection with their IDs and basic info."""
    args_schema = ListDocumentsInput

    def _run(self, collection: str = "lecture_notes", limit: int = 20, order_by: str = "updated_at", order_direction: str = "desc") -> str:
        print(f"Listing documents from collection {collection}")
        
        try:
            db = firestore.Client()
            collection_ref = db.collection(collection)
            
            # Apply ordering if field exists
            try:
                if order_direction.lower() == "desc":
                    query = collection_ref.order_by(order_by, direction=firestore.Query.DESCENDING)
                else:
                    query = collection_ref.order_by(order_by, direction=firestore.Query.ASCENDING)
            except Exception:
                # If ordering fails, just get documents without ordering
                query = collection_ref
            
            # Apply limit
            docs = query.limit(limit).stream()
            
            documents = []
            for doc in docs:
                data = doc.to_dict()
                doc_info = {
                    "id": doc.id,
                    "name": data.get('name', 'Unnamed'),
                    "type": collection,
                    "created_at": data.get('created_at', ''),
                    "updated_at": data.get('updated_at', ''),
                }
                
                # Add collection-specific fields
                if collection == "lecture_notes":
                    doc_info.update({
                        "topic": data.get('topic', ''),
                        "course_title": data.get('course_title', ''),
                        "content_length": data.get('content_length', 0),
                        "tags": data.get('tags', [])
                    })
                elif collection == "resources":
                    doc_info.update({
                        "description": data.get('description', ''),
                        "link": data.get('link', ''),
                        "tags": data.get('tags', [])
                    })
                
                documents.append(doc_info)
            
            return json.dumps({
                "success": True,
                "collection": collection,
                "count": len(documents),
                "documents": documents,
                "query_info": {
                    "limit": limit,
                    "order_by": order_by,
                    "order_direction": order_direction
                }
            })
            
        except Exception as e:
            return json.dumps({"success": False, "error": f"Failed to list documents: {str(e)}"})

#=================================================================
# Get Document by ID Tool
#=================================================================

class GetDocumentInput(BaseModel):
    document_id: str = Field(description="The ID of the document to retrieve")
    collection: str = Field(default="lecture_notes", description="Collection name (lecture_notes or resources)")

class GetDocument(BaseTool):
    name: str = "get_document"
    description: str = """Retrieve a specific document by its ID from Firestore."""
    args_schema = GetDocumentInput

    def _run(self, document_id: str, collection: str = "lecture_notes") -> str:
        print(f"Getting document {document_id} from collection {collection}")
        
        try:
            db = firestore.Client()
            doc_ref = db.collection(collection).document(document_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return json.dumps({
                    "success": False, 
                    "error": f"Document with ID {document_id} not found in collection {collection}"
                })
            
            doc_data = doc.to_dict()
            doc_data["id"] = doc.id  # Add the ID to the returned data
            
            return json.dumps({
                "success": True,
                "document": doc_data,
                "document_id": document_id,
                "collection": collection
            })
            
        except Exception as e:
            return json.dumps({"success": False, "error": f"Failed to get document: {str(e)}"})

#=================================================================
# Enhanced RuleBasedReasoner with CRUD operations
#=================================================================

class EnhancedRuleBasedReasoner(RuleBasedReasoner):
    """Extended rule-based reasoner with CRUD operations"""
    
    def __init__(self):
        super().__init__()
        self.update_document = UpdateDocument()
        self.delete_document = DeleteDocument()
        self.list_documents = ListDocuments()
        self.get_document = GetDocument()

    def update_knowledge(self, document_id: str, updates: Dict, collection: str = "lecture_notes") -> Dict:
        """Update existing knowledge in the database"""
        try:
            result = self.update_document._run(document_id, collection, updates)
            return json.loads(result)
        except Exception as e:
            return {"success": False, "error": f"Update failed: {str(e)}"}

    def delete_knowledge(self, document_id: str, collection: str = "lecture_notes", confirm: bool = False) -> Dict:
        """Delete knowledge from the database"""
        try:
            result = self.delete_document._run(document_id, collection, confirm)
            return json.loads(result)
        except Exception as e:
            return {"success": False, "error": f"Delete failed: {str(e)}"}

    def list_knowledge(self, collection: str = "lecture_notes", limit: int = 20) -> Dict:
        """List documents in the knowledge base"""
        try:
            result = self.list_documents._run(collection, limit)
            return json.loads(result)
        except Exception as e:
            return {"success": False, "error": f"List failed: {str(e)}"}

    def get_knowledge_by_id(self, document_id: str, collection: str = "lecture_notes") -> Dict:
        """Get a specific document by ID"""
        try:
            result = self.get_document._run(document_id, collection)
            return json.loads(result)
        except Exception as e:
            return {"success": False, "error": f"Get failed: {str(e)}"}

    def find_documents_by_name(self, name_pattern: str, collection: str = "lecture_notes") -> List[Dict]:
        """Find documents by name pattern"""
        try:
            # First list all documents
            list_result = self.list_knowledge(collection, limit=100)
            if not list_result.get("success", False):
                return []
            
            # Filter by name pattern
            documents = list_result.get("documents", [])
            pattern_lower = name_pattern.lower()
            
            matching_docs = []
            for doc in documents:
                doc_name = doc.get("name", "").lower()
                if pattern_lower in doc_name:
                    matching_docs.append(doc)
            
            return matching_docs
        except Exception as e:
            return []

#=================================================================
# Enhanced RAG Agent with CRUD operations
#=================================================================

class EnhancedRAGAgent(RuleBasedRAGAgent):
    """Enhanced RAG Agent with CRUD operations"""
    
    def __init__(self):
        super().__init__()
        self.reasoner = EnhancedRuleBasedReasoner()

    def update_document(self, document_id: str, updates: Dict, collection: str = "lecture_notes") -> str:
        """Update a document in the knowledge base"""
        result = self.reasoner.update_knowledge(document_id, updates, collection)
        
        if result.get("success", False):
            return f"✅ Document updated successfully!\n\nUpdated fields: {', '.join(result.get('updated_fields', []))}\nDocument ID: {document_id}"
        else:
            return f"❌ Update failed: {result.get('error', 'Unknown error')}"

    def delete_document(self, document_id: str, collection: str = "lecture_notes", confirm: bool = False) -> str:
        """Delete a document from the knowledge base"""
        if not confirm:
            return "⚠️  Deletion requires confirmation. This action cannot be undone. Use confirm=True to proceed."
        
        result = self.reasoner.delete_knowledge(document_id, collection, confirm)
        
        if result.get("success", False):
            deleted_doc = result.get("deleted_document", {})
            return f"✅ Document deleted successfully!\n\nDeleted: {deleted_doc.get('name', 'Unknown')}\nID: {deleted_doc.get('id', 'Unknown')}\nCollection: {deleted_doc.get('collection', 'Unknown')}"
        else:
            return f"❌ Delete failed: {result.get('error', 'Unknown error')}"

    def list_documents(self, collection: str = "lecture_notes", limit: int = 20) -> str:
        """List documents in the knowledge base"""
        result = self.reasoner.list_knowledge(collection, limit)
        
        if not result.get("success", False):
            return f"❌ Failed to list documents: {result.get('error', 'Unknown error')}"
        
        documents = result.get("documents", [])
        if not documents:
            return f"📭 No documents found in collection '{collection}'"
        
        # Format the list
        response = [f"📚 **{collection.title()} ({len(documents)} documents)**\n"]
        
        for i, doc in enumerate(documents, 1):
            response.append(f"**{i}. {doc.get('name', 'Unnamed')}**")
            response.append(f"   ID: `{doc.get('id')}`")
            
            if collection == "lecture_notes":
                response.append(f"   Topic: {doc.get('topic', 'N/A')}")
                response.append(f"   Course: {doc.get('course_title', 'N/A')}")
                response.append(f"   Content Length: {doc.get('content_length', 0)} chars")
            elif collection == "resources":
                response.append(f"   Description: {doc.get('description', 'N/A')}")
                if doc.get('link'):
                    response.append(f"   Link: {doc.get('link')}")
            
            response.append(f"   Updated: {doc.get('updated_at', 'N/A')}")
            response.append("")
        
        return "\n".join(response)

    def get_document(self, document_id: str, collection: str = "lecture_notes") -> str:
        """Get a specific document by ID"""
        result = self.reasoner.get_knowledge_by_id(document_id, collection)
        
        if not result.get("success", False):
            return f"❌ Failed to get document: {result.get('error', 'Unknown error')}"
        
        doc_data = result.get("document", {})
        
        # Format the document info
        response = [f"📄 **Document Details**\n"]
        response.append(f"**Name:** {doc_data.get('name', 'Unnamed')}")
        response.append(f"**ID:** `{doc_data.get('id')}`")
        response.append(f"**Collection:** {collection}")
        
        if collection == "lecture_notes":
            response.append(f"**Topic:** {doc_data.get('topic', 'N/A')}")
            response.append(f"**Course:** {doc_data.get('course_title', 'N/A')}")
            response.append(f"**Content Length:** {doc_data.get('content_length', 0)} characters")
            
            if doc_data.get('key_concepts'):
                response.append(f"**Key Concepts:** {', '.join(doc_data['key_concepts'])}")
            
            if doc_data.get('tags'):
                response.append(f"**Tags:** {', '.join(doc_data['tags'])}")
        
        elif collection == "resources":
            response.append(f"**Description:** {doc_data.get('description', 'N/A')}")
            if doc_data.get('link'):
                response.append(f"**Link:** {doc_data.get('link')}")
            if doc_data.get('tags'):
                response.append(f"**Tags:** {', '.join(doc_data['tags'])}")
        
        response.append(f"**Created:** {doc_data.get('created_at', 'N/A')}")
        response.append(f"**Updated:** {doc_data.get('updated_at', 'N/A')}")
        
        # Show content preview for lecture notes
        if collection == "lecture_notes" and doc_data.get('lecture_content'):
            content = doc_data['lecture_content']
            preview = content[:500] + "..." if len(content) > 500 else content
            response.append(f"\n**Content Preview:**\n{preview}")
        
        return "\n".join(response)

    def find_documents(self, name_pattern: str, collection: str = "lecture_notes") -> str:
        """Find documents by name pattern"""
        matching_docs = self.reasoner.find_documents_by_name(name_pattern, collection)
        
        if not matching_docs:
            return f"🔍 No documents found matching '{name_pattern}' in collection '{collection}'"
        
        response = [f"🔍 **Found {len(matching_docs)} document(s) matching '{name_pattern}'**\n"]
        
        for i, doc in enumerate(matching_docs, 1):
            response.append(f"**{i}. {doc.get('name', 'Unnamed')}**")
            response.append(f"   ID: `{doc.get('id')}`")
            if collection == "lecture_notes":
                response.append(f"   Topic: {doc.get('topic', 'N/A')}")
            response.append("")
        
        return "\n".join(response)

#=================================================================
# Example Usage
#=================================================================

def demo_crud_operations():
    """Demonstrate CRUD operations"""
    agent = EnhancedRAGAgent()
    
    print("=== CRUD Operations Demo ===\n")
    
    # 1. List existing documents
    print("1. Listing existing documents:")
    print(agent.list_documents("lecture_notes", limit=5))
    print("\n" + "="*50 + "\n")
    
    # 2. Create a test document (using existing save functionality)
    print("2. Creating a test document:")
    test_content = """# Test Document
    
This is a test document for demonstrating CRUD operations.

## Key Points
- Point 1: This is important
- Point 2: This is also important

## Conclusion
This document will be updated and potentially deleted.
"""
    
    result = agent.reasoner.save_knowledge._run(
        name="CRUD Test Document",
        content=test_content,
        topic="Testing",
        tags=["test", "crud", "demo"],
        source_query="crud operations demo"
    )
    
    result_data = json.loads(result)
    if result_data.get("success"):
        test_doc_id = result_data["document_id"]
        print(f"✅ Test document created with ID: {test_doc_id}")
    else:
        print(f"❌ Failed to create test document: {result_data.get('error')}")
        return
    
    print("\n" + "="*50 + "\n")
    
    # 3. Get the document
    print("3. Retrieving the test document:")
    print(agent.get_document(test_doc_id))
    print("\n" + "="*50 + "\n")
    
    # 4. Update the document
    print("4. Updating the test document:")
    updates = {
        "name": "Updated CRUD Test Document",
        "topic": "Advanced Testing",
        "tags": ["test", "crud", "demo", "updated"],
        "lecture_content": test_content + "\n\n## Update\nThis document has been updated successfully!"
    }
    
    print(agent.update_document(test_doc_id, updates))
    print("\n" + "="*50 + "\n")
    
    # 5. Get updated document
    print("5. Retrieving updated document:")
    print(agent.get_document(test_doc_id))
    print("\n" + "="*50 + "\n")
    
    # 6. Search for the document
    print("6. Finding documents by name:")
    print(agent.find_documents("CRUD", "lecture_notes"))
    print("\n" + "="*50 + "\n")
    
    # 7. Delete the document (with confirmation)
    print("7. Deleting the test document:")
    print(agent.delete_document(test_doc_id, confirm=True))
    print("\n" + "="*50 + "\n")
    
    # 8. Verify deletion
    print("8. Verifying deletion:")
    print(agent.get_document(test_doc_id))

if __name__ == "__main__":
    demo_crud_operations()