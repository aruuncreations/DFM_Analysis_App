import io
import base64
import os
import threading
import logging
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from report_generator import generate_reports_in_memory

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dfm_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dfm_analysis")

# File storage in memory
MEMORY_STORAGE = {}
ALLOWED_EXTENSIONS = {'stp', 'step', 'stl'}
RUNNING_TASKS = {}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file_task(file_content, filename, material, density, draft_angle, pull_direction, task_id):
    """Background task to process files"""
    try:
        logger.info(f"Starting background task {task_id} for file: {filename}")
        
        reports = generate_reports_in_memory(file_content, filename, material, density, draft_angle, pull_direction)
        
        # Verify the reports have content
        if not reports['pdf_content'] or len(reports['pdf_content']) == 0:
            raise ValueError("Generated PDF report is empty")
            
        if not reports['html_content'] or len(reports['html_content']) == 0:
            raise ValueError("Generated HTML report is empty")
        
        if task_id in RUNNING_TASKS:
            MEMORY_STORAGE[f'pdf_{filename}'] = reports['pdf_content']
            MEMORY_STORAGE[f'html_{filename}'] = reports['html_content']
            MEMORY_STORAGE[f'task_status_{task_id}'] = 'completed'
            logger.info(f"Task {task_id} completed successfully. PDF size: {len(reports['pdf_content'])} bytes, HTML size: {len(reports['html_content'])} bytes")
    except MemoryError:
        logger.error(f"MemoryError in task {task_id}")
        MEMORY_STORAGE[f'task_status_{task_id}'] = 'error: Memory limit exceeded'
    except Exception as e:
        logger.error(f"Error in task {task_id}: {str(e)}", exc_info=True)
        MEMORY_STORAGE[f'task_status_{task_id}'] = f'error: {str(e)}'
    finally:
        RUNNING_TASKS.pop(task_id, None)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            logger.info("Processing form submission")

            if 'file' not in request.files:
                logger.warning("No file part in request")
                return render_template('index.html', error="No file part in the request")

            file = request.files['file']

            if file.filename == '':
                logger.warning("No file selected")
                return render_template('index.html', error="No file selected")

            if not allowed_file(file.filename):
                logger.warning(f"Invalid file type: {file.filename}")
                return render_template('index.html', error="Unsupported file type. Allowed: .stp, .step, .stl")

            # Get form data
            material = request.form.get('material')
            density = request.form.get('density')
            try:
                draft_angle = float(request.form.get('draft_angle'))
            except (ValueError, TypeError):
                logger.warning("Invalid draft angle")
                return render_template('index.html', error="Draft angle must be a valid number")

            pull_direction = request.form.get('pull_direction')

            logger.info(f"Received file: {file.filename}, Material: {material}, Density: {density}, Draft Angle: {draft_angle}, Pull Direction: {pull_direction}")

            # Read file into memory completely
            file_content = file.read()
            
            # Log file content type and size for debugging
            logger.debug(f"File size: {len(file_content)} bytes, type: {type(file_content)}")

            filename = secure_filename(file.filename)
            MEMORY_STORAGE[f'input_{filename}'] = file_content

            # Generate unique task ID
            task_id = f"task_{filename}_{hash(str(file_content))}"  # Make sure hash input is string
            MEMORY_STORAGE[f'task_status_{task_id}'] = 'processing'

            # Start processing in a background thread
            thread = threading.Thread(
                target=process_file_task,
                args=(file_content, filename, material, density, draft_angle, pull_direction, task_id)
            )
            RUNNING_TASKS[task_id] = thread
            thread.daemon = True
            thread.start()

            file_ids = {
                'input': f'input_{filename}',
                'task_id': task_id,
                'filename': filename
            }

            logger.info(f"Started background task {task_id}")
            return render_template('processing.html', file_ids=file_ids)

        except Exception as e:
            logger.error(f"Error processing form: {str(e)}", exc_info=True)
            return render_template('index.html', error=f"Processing error: {str(e)}")

    return render_template('index.html')

@app.route('/check_status/<task_id>')
def check_status(task_id):
    status_key = f'task_status_{task_id}'
    if status_key in MEMORY_STORAGE:
        return {'status': MEMORY_STORAGE[status_key]}
    return {'status': 'unknown'}

@app.route('/results/<task_id>/<filename>')
def results(task_id, filename):
    status = MEMORY_STORAGE.get(f'task_status_{task_id}', 'unknown')

    if status == 'completed':
        file_ids = {
            'input': f'input_{filename}',
            'pdf': f'pdf_{filename}',
            'html': f'html_{filename}'
        }
        return render_template('results.html', reports=file_ids, filename=filename)
    
    if status.startswith('error:'):
        return render_template('index.html', error=status[6:])

    return render_template('processing.html', file_ids={'task_id': task_id, 'filename': filename})

@app.route('/download/<file_id>')
def download(file_id):
    try:
        file_content = MEMORY_STORAGE.get(file_id)
        if not file_content:
            logger.warning(f"File not found: {file_id}")
            return "File not found", 404

        file_type, filename = file_id.split('_', 1)
        ext = filename.rsplit('.', 1)[-1].lower()

        if file_type == 'pdf':
            mimetype = 'application/pdf'
            download_name = f'DFM_Analysis_{filename}.pdf'
        elif file_type == 'html':
            mimetype = 'text/html'
            download_name = f'3D_Visualization_{filename}.html'
        else:
            mimetype = 'application/step' if ext in {'stp', 'step'} else 'model/stl'
            download_name = filename

        logger.info(f"Downloading file: {download_name}, Size: {len(file_content)} bytes")
        return send_file(io.BytesIO(file_content), mimetype=mimetype, as_attachment=True, download_name=download_name)

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", 500

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup(session_id):
    try:
        keys_to_remove = [key for key in MEMORY_STORAGE if session_id in key]
        for key in keys_to_remove:
            MEMORY_STORAGE.pop(key, None)
        logger.info(f"Cleaned up session: {session_id}")
        return "OK"
    except Exception as e:
        logger.error(f"Error cleaning up session: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)