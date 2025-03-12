import io
import base64
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from report_generator import generate_reports_in_memory
import logging
import threading

# Set up basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dfm_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dfm_analysis")

# File storage in memory
MEMORY_STORAGE = {}
ALLOWED_EXTENSIONS = {'stp', 'step', 'stl'}

# Track running tasks to allow cancellation
RUNNING_TASKS = {}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file_task(file_content, filename, material, density, draft_angle, pull_direction, task_id):
    """Background task to process files"""
    try:
        logger.info(f"Starting background task {task_id} for file: {filename}")
        
        # Generate reports in memory
        reports = generate_reports_in_memory(file_content, filename, material, density, draft_angle, pull_direction)
        
        # Store generated reports in memory if task wasn't cancelled
        if task_id in RUNNING_TASKS:
            MEMORY_STORAGE[f'pdf_{filename}'] = reports['pdf_content']
            MEMORY_STORAGE[f'html_{filename}'] = reports['html_content']
            MEMORY_STORAGE[f'task_status_{task_id}'] = 'completed'
            logger.info(f"Task {task_id} completed successfully")
        else:
            logger.info(f"Task {task_id} was cancelled, not storing results")
            
    except Exception as e:
        logger.error(f"Error in background task {task_id}: {str(e)}", exc_info=True)
        if task_id in RUNNING_TASKS:
            MEMORY_STORAGE[f'task_status_{task_id}'] = f'error: {str(e)}'
    finally:
        # Clean up task tracking
        if task_id in RUNNING_TASKS:
            RUNNING_TASKS.pop(task_id)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            logger.info("Processing form submission")
            # Check if the file part exists
            if 'file' not in request.files:
                logger.warning("No file part in request")
                return render_template('index.html', error="No file part in the request")
            
            file = request.files['file']
            
            # Check if the user submitted an empty file input
            if file.filename == '':
                logger.warning("No file selected")
                return render_template('index.html', error="No file selected")
            
            # Check if the file has an allowed extension
            if not allowed_file(file.filename):
                logger.warning(f"Invalid file type: {file.filename}")
                return render_template('index.html', error="File type not supported. Please upload .stp, .step, or .stl files.")
            
            # Get remaining form data
            material = request.form.get('material')
            density = request.form.get('density')
            
            # Add validation for draft angle
            try:
                draft_angle = float(request.form.get('draft_angle'))
            except (ValueError, TypeError):
                logger.warning("Invalid draft angle")
                return render_template('index.html', error="Draft angle must be a valid number")
                
            pull_direction = request.form.get('pull_direction')
            
            logger.info(f"Processing file: {file.filename}, Material: {material}, Density: {density}, " 
                       f"Draft Angle: {draft_angle}, Pull Direction: {pull_direction}")
            
            # Read file into memory
            file_content = file.read()
            filename = secure_filename(file.filename)
            
            # Store input file in memory
            MEMORY_STORAGE[f'input_{filename}'] = file_content
            
            # Create a unique task ID
            task_id = f"task_{filename}_{hash(file_content)}"
            
            # Store initial task status
            MEMORY_STORAGE[f'task_status_{task_id}'] = 'processing'
            
            # Start processing in a background thread
            thread = threading.Thread(
                target=process_file_task,
                args=(file_content, filename, material, density, draft_angle, pull_direction, task_id)
            )
            RUNNING_TASKS[task_id] = thread
            thread.daemon = True  # Thread will be terminated when the main process exits
            thread.start()
            
            # Generate file IDs for the template
            file_ids = {
                'input': f'input_{filename}',
                'task_id': task_id,
                'filename': filename
            }
            
            logger.info(f"Started background task {task_id}, rendering processing page")
            # Render results page with processing indicator
            return render_template('processing.html', file_ids=file_ids)
            
        except Exception as e:
            logger.error(f"Error processing form: {str(e)}", exc_info=True)
            return render_template('index.html', error=f"Processing error: {str(e)}")
    
    return render_template('index.html')

@app.route('/check_status/<task_id>')
def check_status(task_id):
    """Check the status of a processing task"""
    status_key = f'task_status_{task_id}'
    if status_key in MEMORY_STORAGE:
        status = MEMORY_STORAGE[status_key]
        return {'status': status}
    return {'status': 'unknown'}

@app.route('/results/<task_id>/<filename>')
def results(task_id, filename):
    """Show results after processing is complete"""
    status_key = f'task_status_{task_id}'
    if status_key in MEMORY_STORAGE:
        status = MEMORY_STORAGE[status_key]
        
        if status == 'completed':
            # Generate file IDs for the template
            file_ids = {
                'input': f'input_{filename}',
                'pdf': f'pdf_{filename}',
                'html': f'html_{filename}'
            }
            
            logger.info("Rendering results page")
            return render_template('results.html', reports=file_ids, filename=filename)
        elif status.startswith('error:'):
            error_message = status[6:]  # Remove 'error:' prefix
            return render_template('index.html', error=f"Processing error: {error_message}")
        else:
            # Still processing
            return render_template('processing.html', file_ids={'task_id': task_id, 'filename': filename})
    
    # Task not found
    return render_template('index.html', error="Task not found or expired")

@app.route('/download/<file_id>')
def download(file_id):
    try:
        if file_id not in MEMORY_STORAGE:
            logger.warning(f"File not found: {file_id}")
            return "File not found", 404
        
        file_content = MEMORY_STORAGE[file_id]
        file_type = file_id.split('_')[0]
        filename = '_'.join(file_id.split('_')[1:])
        
        # Set appropriate MIME type
        if file_type == 'pdf':
            mimetype = 'application/pdf'
            download_name = f'DFM_Analysis_{filename}.pdf'
        elif file_type == 'html':
            mimetype = 'text/html'
            download_name = f'3D_Visualization_{filename}.html'
        else:  # input file
            # Determine mimetype based on file extension
            ext = filename.lower().split('.')[-1]
            if ext == 'stl':
                mimetype = 'model/stl'
            else:  # stp or step
                mimetype = 'application/step'
            download_name = filename
        
        logger.info(f"Downloading file: {download_name}, type: {mimetype}")
        
        return send_file(
            io.BytesIO(file_content),
            mimetype=mimetype,
            as_attachment=True,
            download_name=download_name
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        return f"Error downloading file: {str(e)}", 500

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup(session_id):
    try:
        # Clear any files associated with this session
        keys_to_remove = [key for key in MEMORY_STORAGE if session_id in key]
        for key in keys_to_remove:
            MEMORY_STORAGE.pop(key, None)
        logger.info(f"Cleaned up session: {session_id}")
        return "OK"
    except Exception as e:
        logger.error(f"Error cleaning up session: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))  # Use Render's assigned port or default to 5000
    app.run(debug=True, host="0.0.0.0", port=port)