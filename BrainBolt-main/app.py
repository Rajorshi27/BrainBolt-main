from flask import Flask, render_template, request, session, jsonify
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
import io

# Load environment variables (OpenAI API key)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Set up OpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file):
    """Extract text content from various file types"""
    if file.filename == '':
        return ''

    file_extension = file.filename.rsplit('.', 1)[1].lower()
    text_content = ''

    if file_extension == 'txt':
        text_content = file.read().decode('utf-8')
    elif file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
    elif file_extension in ['doc', 'docx']:
        doc = Document(io.BytesIO(file.read()))
        text_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    return text_content

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/form')
def form_page():
    return render_template('form.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/generate', methods=['POST'])
def generate_ideas():
    # Get data from the form
    topic = request.form['topic']
    members = request.form['members']
    goal = request.form['goal']

    # Handle file uploads
    file_contents = []
    if 'fileUpload' in request.files:
        files = request.files.getlist('fileUpload')
        for file in files:
            if file and file.filename != '' and allowed_file(file.filename):
                text_content = extract_text_from_file(file)
                if text_content:
                    file_contents.append(text_content)

    # Store session data
    session['topic'] = topic
    session['members'] = members
    session['goal'] = goal
    session['file_contents'] = file_contents

    # Render the session template with the data
    return render_template('session.html',
                           topic=topic,
                           members=members,
                           goal=goal)

@app.route('/initialize_session', methods=['POST'])
def initialize_session():
    data = request.json
    topic = data.get('topic', '')
    members = data.get('members', '')
    goal = data.get('goal', '')

    # Store initial session context
    session['context'] = {
        'topic': topic,
        'members': members,
        'goal': goal,
        'history': []  # Store conversation history
    }

    messages = [
        SystemMessage(content="""You are an expert brainstorming facilitator. Given the topic, participants, and goal:
        1. Choose the most appropriate brainstorming method for this specific situation
        2. Explain briefly why this method is ideal for their needs
        3. Outline just the FIRST step they should take
        4. Ask them to complete just that first step
        
        Keep your response encouraging but concise. Wait for their completion of each step before moving on."""),
        HumanMessage(content=f"We're starting a brainstorming session about '{topic}' with {members} participants. Our goal is: {goal}")
    ]

    response = model.invoke(messages)
    session['context']['history'].append(("assistant", response.content))

    return jsonify({"response": response.content})

@app.route('/message', methods=['POST'])
def process_message():
    message = request.json.get('message', '')
    context = session.get('context', {})

    # Add user message to history
    context['history'].append(("user", message))

    # Create messages list including full conversation history for context
    messages = [
        SystemMessage(content="""You are an expert brainstorming facilitator guiding a session. For each response:
        1. Acknowledge and engage with what the user has shared
        2. If they've completed the current step satisfactorily, introduce the next logical step
        3. If they need more guidance or their response is insufficient, provide specific help for the current step
        4. If they ask questions, answer them clearly and return focus to the current step
        5. Keep the brainstorming momentum while ensuring quality completion of each step
        
        Stay encouraging but focused. Only move forward when genuine progress is made."""),
        HumanMessage(content=f"""Topic: {context['topic']}
Goal: {context['goal']}
Participants: {context['members']}

Previous conversation:
{chr(10).join(f'{"Assistant" if role == "assistant" else "User"}: {msg}' for role, msg in context['history'][-5:])}

New message: {message}""")
    ]

    response = model.invoke(messages)

    # Update history
    context['history'].append(("assistant", response.content))
    session['context'] = context

    return jsonify({"response": response.content})

@app.route('/save_notes', methods=['POST'])
def save_notes():
    notes = request.json.get('notes', '')
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)