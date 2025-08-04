from flask import Flask, request, jsonify
from brainex_memory import RuleBasedRAGAgent, HumanMessage
from google.cloud import firestore
from datetime import datetime
import os
import json

app = Flask(__name__)

# Initialize the RAG Agent and Firestore client
agent = RuleBasedRAGAgent()
db = firestore.Client()

@app.route('/')
def health_check():
    return jsonify({'status': 'ok', 'message': 'Flask RAG CRUD API is running'})

# =================================================================
# RAG Query Endpoints
# =================================================================

@app.route('/query', methods=['POST'])
def query_rag():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Please provide a "question" field in JSON body.'}), 400

    question = data['question']
    thread_id = data.get('thread_id', 'default')

    try:
        answer = agent.query(question, thread_id=thread_id)
        return jsonify({'question': question, 'answer': answer, 'thread_id': thread_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    thread_id = request.args.get('thread_id', 'default')
    try:
        history = agent.get_conversation_history(thread_id)
        # Serialize messages
        serialized = []
        for msg in history:
            serialized.append({
                'type': msg.__class__.__name__,
                'content': msg.content
            })
        return jsonify({'thread_id': thread_id, 'history': serialized})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =================================================================
# Lecture Notes CRUD Operations
# =================================================================

@app.route('/lecture_notes', methods=['POST'])
def create_lecture_note():
    """Create a new lecture note"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Please provide JSON data'}), 400
    
    required_fields = ['name', 'lecture_content']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        # Prepare document data
        doc_data = {
            'name': data['name'],
            'description': data.get('description', ''),
            'lecture_content': data['lecture_content'],
            'course_title': data.get('course_title', ''),
            'topic': data.get('topic', ''),
            'tags': data.get('tags', []),
            'key_concepts': data.get('key_concepts', []),
            'content_type': 'lecture_notes',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'content_length': len(data['lecture_content'])
        }
        
        # Add to Firestore
        doc_ref = db.collection('lecture_notes').document()
        doc_ref.set(doc_data)
        
        # Return created document with ID
        doc_data['id'] = doc_ref.id
        return jsonify({
            'message': 'Lecture note created successfully',
            'lecture_note': doc_data
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Failed to create lecture note: {str(e)}'}), 500

@app.route('/lecture_notes', methods=['GET'])
def get_all_lecture_notes():
    """Get all lecture notes with optional filtering"""
    try:
        # Get query parameters for filtering
        topic = request.args.get('topic')
        course_title = request.args.get('course_title')
        tag = request.args.get('tag')
        limit = request.args.get('limit', type=int)
        
        # Start with base query
        query = db.collection('lecture_notes')
        
        # Apply filters
        if topic:
            query = query.where('topic', '==', topic)
        if course_title:
            query = query.where('course_title', '==', course_title)
        if tag:
            query = query.where('tags', 'array_contains', tag)
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        # Execute query
        docs = query.stream()
        
        lecture_notes = []
        for doc in docs:
            note_data = doc.to_dict()
            note_data['id'] = doc.id
            lecture_notes.append(note_data)
        
        return jsonify({
            'lecture_notes': lecture_notes,
            'count': len(lecture_notes)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve lecture notes: {str(e)}'}), 500

@app.route('/lecture_notes/<note_id>', methods=['GET'])
def get_lecture_note(note_id):
    """Get a single lecture note by ID"""
    try:
        doc_ref = db.collection('lecture_notes').document(note_id)
        doc = doc_ref.get()

        print(doc)
        
        if not doc.exists:
            return jsonify({'error': 'Lecture note not found'}), 404
        
        note_data = doc.to_dict()
        note_data['id'] = doc.id
        
        return jsonify({'lecture_note': note_data})
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve lecture note: {str(e)}'}), 500

@app.route('/lecture_notes/<note_id>', methods=['PUT'])
def update_lecture_note(note_id):
    """Update a lecture note"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Please provide JSON data'}), 400
    
    try:
        doc_ref = db.collection('lecture_notes').document(note_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({'error': 'Lecture note not found'}), 404
        
        # Prepare update data
        update_data = {}
        updatable_fields = ['name', 'description', 'lecture_content', 'course_title', 
                           'topic', 'tags', 'key_concepts']
        
        for field in updatable_fields:
            if field in data:
                update_data[field] = data[field]
        
        # Always update the timestamp and content length
        update_data['updated_at'] = datetime.now().isoformat()
        if 'lecture_content' in update_data:
            update_data['content_length'] = len(update_data['lecture_content'])
        
        # Update document
        doc_ref.update(update_data)
        
        # Return updated document
        updated_doc = doc_ref.get()
        updated_data = updated_doc.to_dict()
        updated_data['id'] = note_id
        
        return jsonify({
            'message': 'Lecture note updated successfully',
            'lecture_note': updated_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to update lecture note: {str(e)}'}), 500

@app.route('/lecture_notes/<note_id>', methods=['DELETE'])
def delete_lecture_note(note_id):
    """Delete a lecture note"""
    try:
        doc_ref = db.collection('lecture_notes').document(note_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({'error': 'Lecture note not found'}), 404
        
        # Delete the document
        doc_ref.delete()
        
        return jsonify({'message': 'Lecture note deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete lecture note: {str(e)}'}), 500

# =================================================================
# Resources CRUD Operations
# =================================================================

@app.route('/resources', methods=['POST'])
def create_resource():
    """Create a new resource"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Please provide JSON data'}), 400
    
    required_fields = ['name']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        # Prepare document data
        doc_data = {
            'name': data['name'],
            'description': data.get('description', ''),
            'link': data.get('link', ''),
            'tags': data.get('tags', []),
            'resource_type': data.get('resource_type', 'general'),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Add to Firestore
        doc_ref = db.collection('resources').document()
        doc_ref.set(doc_data)
        
        # Return created document with ID
        doc_data['id'] = doc_ref.id
        return jsonify({
            'message': 'Resource created successfully',
            'resource': doc_data
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Failed to create resource: {str(e)}'}), 500

@app.route('/resources', methods=['GET'])
def get_all_resources():
    """Get all resources with optional filtering"""
    try:
        # Get query parameters for filtering
        resource_type = request.args.get('resource_type')
        tag = request.args.get('tag')
        limit = request.args.get('limit', type=int)
        
        # Start with base query
        query = db.collection('resources')
        
        # Apply filters
        if resource_type:
            query = query.where('resource_type', '==', resource_type)
        if tag:
            query = query.where('tags', 'array_contains', tag)
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        # Execute query
        docs = query.stream()
        
        resources = []
        for doc in docs:
            resource_data = doc.to_dict()
            resource_data['id'] = doc.id
            resources.append(resource_data)
        
        return jsonify({
            'resources': resources,
            'count': len(resources)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve resources: {str(e)}'}), 500

@app.route('/resources/<resource_id>', methods=['GET'])
def get_resource(resource_id):
    """Get a single resource by ID"""
    try:
        doc_ref = db.collection('resources').document(resource_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({'error': 'Resource not found'}), 404
        
        resource_data = doc.to_dict()
        resource_data['id'] = doc.id
        
        return jsonify({'resource': resource_data})
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve resource: {str(e)}'}), 500

@app.route('/resources/<resource_id>', methods=['PUT'])
def update_resource(resource_id):
    """Update a resource"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Please provide JSON data'}), 400
    
    try:
        doc_ref = db.collection('resources').document(resource_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({'error': 'Resource not found'}), 404
        
        # Prepare update data
        update_data = {}
        updatable_fields = ['name', 'description', 'link', 'tags', 'resource_type']
        
        for field in updatable_fields:
            if field in data:
                update_data[field] = data[field]
        
        # Always update the timestamp
        update_data['updated_at'] = datetime.now().isoformat()
        
        # Update document
        doc_ref.update(update_data)
        
        # Return updated document
        updated_doc = doc_ref.get()
        updated_data = updated_doc.to_dict()
        updated_data['id'] = resource_id
        
        return jsonify({
            'message': 'Resource updated successfully',
            'resource': updated_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to update resource: {str(e)}'}), 500

@app.route('/resources/<resource_id>', methods=['DELETE'])
def delete_resource(resource_id):
    """Delete a resource"""
    try:
        doc_ref = db.collection('resources').document(resource_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({'error': 'Resource not found'}), 404
        
        # Delete the document
        doc_ref.delete()
        
        return jsonify({'message': 'Resource deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete resource: {str(e)}'}), 500

# =================================================================
# Search and Analytics Endpoints
# =================================================================

@app.route('/search', methods=['POST'])
def search_knowledge():
    """Search across both lecture notes and resources"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a "query" field in JSON body.'}), 400
    
    query = data['query']
    
    try:
        # Use the existing search functionality from the RAG agent
        firestore_result = agent.reasoner.search_firestore._run(query)
        search_results = json.loads(firestore_result)
        
        return jsonify(search_results)
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get basic analytics about the knowledge base"""
    try:
        # Count lecture notes
        lecture_notes_count = len(list(db.collection('lecture_notes').stream()))
        
        # Count resources
        resources_count = len(list(db.collection('resources').stream()))
        
        # Get recent activity (last 10 items)
        recent_notes = db.collection('lecture_notes').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).stream()
        recent_resources = db.collection('resources').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).stream()
        
        recent_activity = []
        
        for note in recent_notes:
            note_data = note.to_dict()
            recent_activity.append({
                'type': 'lecture_note',
                'id': note.id,
                'name': note_data.get('name'),
                'created_at': note_data.get('created_at')
            })
        
        for resource in recent_resources:
            resource_data = resource.to_dict()
            recent_activity.append({
                'type': 'resource',
                'id': resource.id,
                'name': resource_data.get('name'),
                'created_at': resource_data.get('created_at')
            })
        
        # Sort by creation date
        recent_activity.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            'total_lecture_notes': lecture_notes_count,
            'total_resources': resources_count,
            'total_items': lecture_notes_count + resources_count,
            'recent_activity': recent_activity[:10]
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get analytics: {str(e)}'}), 500

# =================================================================
# Bulk Operations
# =================================================================

@app.route('/lecture_notes/bulk', methods=['POST'])
def bulk_create_lecture_notes():
    """Create multiple lecture notes at once"""
    data = request.get_json()
    if not data or 'lecture_notes' not in data:
        return jsonify({'error': 'Please provide a "lecture_notes" array in JSON body.'}), 400
    
    lecture_notes = data['lecture_notes']
    if not isinstance(lecture_notes, list):
        return jsonify({'error': 'lecture_notes must be an array'}), 400
    
    created_notes = []
    errors = []
    
    for i, note_data in enumerate(lecture_notes):
        try:
            if 'name' not in note_data or 'lecture_content' not in note_data:
                errors.append(f'Item {i}: Missing required fields (name, lecture_content)')
                continue
            
            # Prepare document data
            doc_data = {
                'name': note_data['name'],
                'description': note_data.get('description', ''),
                'lecture_content': note_data['lecture_content'],
                'course_title': note_data.get('course_title', ''),
                'topic': note_data.get('topic', ''),
                'tags': note_data.get('tags', []),
                'key_concepts': note_data.get('key_concepts', []),
                'content_type': 'lecture_notes',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'content_length': len(note_data['lecture_content'])
            }
            
            # Add to Firestore
            doc_ref = db.collection('lecture_notes').document()
            doc_ref.set(doc_data)
            
            doc_data['id'] = doc_ref.id
            created_notes.append(doc_data)
            
        except Exception as e:
            errors.append(f'Item {i}: {str(e)}')
    
    return jsonify({
        'message': f'Bulk operation completed. Created {len(created_notes)} notes.',
        'created_notes': created_notes,
        'errors': errors
    }), 201 if created_notes else 400

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)