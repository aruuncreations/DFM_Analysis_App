import io
import numpy as np
import trimesh
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64
from PIL import Image as PILImage
import logging
import concurrent.futures

# Set up logging for this module
logger = logging.getLogger("dfm_analysis.report_generator")

def parse_file_in_memory(file_content, filename, draft_angle, pull_direction):
    """
    Parse 3D model file and analyze for draft angle compliance
    """
    logger.info(f"Parsing file: {filename}")
    # Determine file type from filename
    file_extension = filename.lower().split('.')[-1]
    
    # Map extensions to trimesh file types
    file_type_map = {
        'stl': 'stl',
        'step': 'step',
        'stp': 'step'
    }
    
    file_type = file_type_map.get(file_extension)
    if not file_type:
        error_msg = f"Unsupported file extension: {file_extension}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Create a new BytesIO object for each attempt to ensure position is at start
        file_stream = io.BytesIO(file_content)
        logger.info(f"Loading mesh with file type: {file_type}")
        try:
            # First try with explicit file_type
            mesh = trimesh.load(file_stream, file_type=file_type)
        except Exception as e:
            logger.warning(f"Failed to load with explicit file_type: {str(e)}")
            # Create a new BytesIO to ensure we start from the beginning
            file_stream = io.BytesIO(file_content)
            try:
                logger.info("Attempting to load without explicit file_type")
                mesh = trimesh.load(file_stream)
            except Exception as e2:
                error_msg = f"Failed to load 3D model: {str(e2)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if mesh is loaded correctly
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        error_msg = "The loaded file doesn't contain a valid mesh with vertices and faces"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Mesh loaded successfully. Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Define pull direction vector
    if pull_direction == 'X':
        axis = np.array([1, 0, 0])
    elif pull_direction == 'Y':
        axis = np.array([0, 1, 0])
    else:  # Z
        axis = np.array([0, 0, 1])
    
    logger.info(f"Analyzing faces for draft angle compliance. Draft angle: {draft_angle}°")
    
    # Analyze each face for draft angle compliance
    compliant_faces = []
    non_compliant_faces = []
    
    for i, face in enumerate(faces):
        normal = mesh.face_normals[i]
        
        # Calculate angle between normal and pull direction
        dot_product = np.abs(np.dot(normal, axis))
        angle = np.degrees(np.arccos(dot_product))
        
        # Check if the angle is within the draft angle requirement
        if angle <= (90 - draft_angle):
            compliant_faces.append(i)
        else:
            non_compliant_faces.append(i)
    
    logger.info(f"Analysis complete. Compliant faces: {len(compliant_faces)}, " 
               f"Non-compliant faces: {len(non_compliant_faces)}")
    
    return mesh, compliant_faces, non_compliant_faces

def generate_3d_plot(mesh, compliant_faces, non_compliant_faces, pull_direction, draft_angle):
    """
    Generate a 3D plot of the mesh showing compliant and non-compliant faces
    with downsampling for large meshes
    """
    logger.info("Generating 3D plot")
    
    try:
        # Check if the mesh is very large and needs downsampling
        face_count = len(mesh.faces)
        logger.info(f"Total face count: {face_count}")
        
        # Apply downsampling for large meshes to improve performance
        if face_count > 10000:
            logger.info(f"Mesh has {face_count} faces. Downsampling for visualization...")
            # Calculate a reasonable downsampling factor
            downsample_factor = max(1, int(face_count / 5000))  # Aim for ~5000 faces
            
            # Downsample by taking every nth face
            if compliant_faces:
                compliant_faces = compliant_faces[::downsample_factor]
            if non_compliant_faces:
                non_compliant_faces = non_compliant_faces[::downsample_factor]
            
            logger.info(f"Downsampled to approximately {len(compliant_faces) + len(non_compliant_faces)} faces")
        
        # Create a figure
        fig = go.Figure()
        
        # Create mesh with compliant faces (green)
        if compliant_faces:
            logger.info("Adding compliant faces to plot")
            compliant_triangles = mesh.faces[compliant_faces]
            vertices = mesh.vertices
            
            i, j, k = compliant_triangles.T
            x, y, z = vertices.T
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color='green',
                opacity=0.7,
                name="Compliant"
            ))
        
        # Create mesh with non-compliant faces (red)
        if non_compliant_faces:
            logger.info("Adding non-compliant faces to plot")
            non_compliant_triangles = mesh.faces[non_compliant_faces]
            vertices = mesh.vertices
            
            i, j, k = non_compliant_triangles.T
            x, y, z = vertices.T
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color='red',
                opacity=0.7,
                name="Non-Compliant"
            ))
        
        # Add pull direction indicator
        center = np.mean(mesh.vertices, axis=0)
        scale = np.max(np.ptp(mesh.vertices, axis=0)) * 0.5
        
        logger.info(f"Adding pull direction indicator: {pull_direction}")
        if pull_direction == 'X':
            end = center + np.array([scale, 0, 0])
            fig.add_trace(go.Scatter3d(
                x=[center[0], end[0]], 
                y=[center[1], end[1]], 
                z=[center[2], end[2]],
                line=dict(color='blue', width=7),
                name="Pull Direction (X)"
            ))
        elif pull_direction == 'Y':
            end = center + np.array([0, scale, 0])
            fig.add_trace(go.Scatter3d(
                x=[center[0], end[0]], 
                y=[center[1], end[1]], 
                z=[center[2], end[2]],
                line=dict(color='blue', width=7),
                name="Pull Direction (Y)"
            ))
        else:  # Z
            end = center + np.array([0, 0, scale])
            fig.add_trace(go.Scatter3d(
                x=[center[0], end[0]], 
                y=[center[1], end[1]], 
                z=[center[2], end[2]],
                line=dict(color='blue', width=7),
                name="Pull Direction (Z)"
            ))
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data'
            ),
            title=f"Draft Angle Analysis (Min Angle: {draft_angle}°)",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            # Add copyright annotation
            annotations=[
                dict(
                    text="Copyrights reserved - aruuncreations",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0,
                    xanchor="center",
                    yanchor="auto",
                    font=dict(size=10)
                )
            ]
        )
        
        logger.info("3D plot generation complete")
        return fig
    except Exception as e:
        error_msg = f"Error generating 3D plot: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Create a simple fallback figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error generating plot: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Error in Plot Generation")
        return fig

# MODIFIED: Skip image generation completely
def generate_static_image(fig):
    """Generate a static image of the 3D model for the PDF report - now always returns None"""
    logger.info("Static image generation skipped to avoid timeouts")
    return None

def create_pdf_report(data, image_bytes=None):
    """Create a PDF report in memory using ReportLab"""
    logger.info("Creating PDF report")
    try:
        # Create a BytesIO object for the PDF
        pdf_buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading2_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create a custom style for centered text
        centered_style = ParagraphStyle(
            'centered',
            parent=normal_style,
            alignment=1  # 0=left, 1=center, 2=right
        )
        
        # Create a custom style for the footer
        footer_style = ParagraphStyle(
            'footer',
            parent=normal_style,
            fontSize=8,
            textColor=colors.gray,
            alignment=1,
            fontName='Helvetica-Oblique'
        )
        
        # Start building the document
        elements = []
        
        # Add title
        elements.append(Paragraph("DFM Analysis Report", title_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Add part information
        elements.append(Paragraph("Part Information", heading2_style))
        elements.append(Paragraph(f"Material: {data['material']}", normal_style))
        elements.append(Paragraph(f"Density: {data['density']} g/cm³", normal_style))
        elements.append(Paragraph(f"Volume: {data['volume']:.2f} mm³", normal_style))
        elements.append(Paragraph(f"Estimated Mass: {data['mass']:.2f} g", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add analysis parameters
        elements.append(Paragraph("Analysis Parameters", heading2_style))
        elements.append(Paragraph(f"Draft Angle: {data['draft_angle']}°", normal_style))
        elements.append(Paragraph(f"Pull Direction: {data['pull_direction']}", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # MODIFIED: Add model visualization section without image
        elements.append(Paragraph("Model Visualization", heading2_style))
        elements.append(Paragraph("3D visualization is available in the interactive HTML report only.", normal_style))
        elements.append(Paragraph("Static visualization was skipped due to model complexity.", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add analysis results
        elements.append(Paragraph("Draft Angle Analysis Results", heading2_style))
        elements.append(Paragraph(f"Total Faces: {data['total_faces']}", normal_style))
        elements.append(Paragraph(f"Compliant Faces: {data['compliant_count']} ({data['compliance_percentage']:.2f}%)", normal_style))
        elements.append(Paragraph(f"Non-Compliant Faces: {data['non_compliant_count']} ({100-data['compliance_percentage']:.2f}%)", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add recommendation
        elements.append(Paragraph("Recommendation", heading2_style))
        if data['compliance_percentage'] > 90:
            recommendation = "The part is suitable for manufacturing with the specified draft angle and pull direction."
        elif data['compliance_percentage'] > 70:
            recommendation = "The part requires some modifications to improve manufacturability."
        else:
            recommendation = "The part requires significant redesign to be manufacturable with the specified parameters."
        elements.append(Paragraph(recommendation, normal_style))
        
        # Build the document
        doc.build(elements, onFirstPage=lambda canvas, doc: add_footer(canvas, doc),
                 onLaterPages=lambda canvas, doc: add_footer(canvas, doc))
        
        # Get the PDF data from the buffer
        pdf_data = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        logger.info("PDF report creation complete")
        return pdf_data
    except Exception as e:
        logger.error(f"Error creating PDF report: {str(e)}", exc_info=True)
        # Create a simple error PDF
        try:
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = [
                Paragraph("DFM Analysis Error Report", styles['Title']),
                Spacer(1, 0.3*inch),
                Paragraph("An error occurred during report generation:", styles['Heading2']),
                Paragraph(str(e), styles['Normal'])
            ]
            doc.build(elements)
            pdf_data = pdf_buffer.getvalue()
            pdf_buffer.close()
            return pdf_data
        except Exception as e2:
            logger.error(f"Error creating error PDF: {str(e2)}", exc_info=True)
            # If all else fails, return an empty PDF
            return b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n'

def add_footer(canvas, doc):
    """Add footer with copyright to each page"""
    footer_text = "Copyrights reserved - aruuncreations"
    canvas.saveState()
    canvas.setFont('Helvetica-Oblique', 8)
    canvas.setFillColor(colors.gray)
    # Position at the bottom center of the page
    canvas.drawCentredString(letter[0]/2, 0.5*inch, footer_text)
    canvas.restoreState()

def generate_reports_in_memory(file_content, filename, material, density, draft_angle, pull_direction):
    """
    Main function to generate reports based on the 3D model
    """
    logger.info(f"Generating reports for {filename}")
    try:
        # Parse the file and analyze draft angles
        mesh, compliant_faces, non_compliant_faces = parse_file_in_memory(file_content, filename, draft_angle, pull_direction)
        
        # Calculate metrics
        total_faces = len(mesh.faces)
        compliant_count = len(compliant_faces)
        non_compliant_count = len(non_compliant_faces)
        compliance_percentage = (compliant_count / total_faces) * 100 if total_faces > 0 else 0
        
        # Calculate volume and mass
        volume = mesh.volume  # in cubic units (likely mm³)
        mass = volume * float(density) / 1000  # convert to g if density is in g/cm³
        
        logger.info(f"Analysis metrics - Volume: {volume:.2f}mm³, Mass: {mass:.2f}g, " 
                   f"Compliance: {compliance_percentage:.2f}%")
        
        # Generate 3D Plot for HTML only
        fig = generate_3d_plot(mesh, compliant_faces, non_compliant_faces, pull_direction, draft_angle)
        
        # MODIFIED: Skip image generation completely - it's causing timeouts
        img_bytes = None  # Don't try to generate the static image
        
        # Create PDF report without image
        data = {
            'material': material,
            'density': density,
            'volume': volume,
            'mass': mass,
            'draft_angle': draft_angle,
            'pull_direction': pull_direction,
            'total_faces': total_faces,
            'compliant_count': compliant_count,
            'non_compliant_count': non_compliant_count,
            'compliance_percentage': compliance_percentage
        }
        
        pdf_content = create_pdf_report(data, img_bytes)
        
        # Generate HTML content
        try:
            html_content = fig.to_html().encode('utf-8')
        except Exception as e:
            logger.error(f"Error generating HTML: {str(e)}", exc_info=True)
            html_content = f"""
            <html>
            <head><title>DFM Analysis Error</title></head>
            <body>
                <h1>DFM Analysis Error</h1>
                <p>Error generating HTML visualization: {str(e)}</p>
            </body>
            </html>
            """.encode('utf-8')
        
        logger.info("Report generation complete")
        
        return {
            'html_content': html_content,
            'pdf_content': pdf_content
        }
        
    except Exception as e:
        logger.error(f"Error in report generation: {str(e)}", exc_info=True)
        
        # Create error reports
        error_message = f"Error processing file: {str(e)}"
        
        # Create simple PDF with error
        try:
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = [
                Paragraph("DFM Analysis Error Report", styles['Title']),
                Spacer(1, 0.3*inch),
                Paragraph("An error occurred during analysis:", styles['Heading2']),
                Paragraph(error_message, styles['Normal'])
            ]
            doc.build(elements)
            pdf_content = pdf_buffer.getvalue()
            pdf_buffer.close()
        except Exception as e2:
            logger.error(f"Error creating error PDF: {str(e2)}", exc_info=True)
            pdf_content = b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n'
        
        # Create simple HTML with error
        html_content = f"""
        <html>
        <head><title>DFM Analysis Error</title></head>
        <body>
            <h1>DFM Analysis Error</h1>
            <p>{error_message}</p>
        </body>
        </html>
        """.encode('utf-8')
        
        return {
            'html_content': html_content,
            'pdf_content': pdf_content
        }

# In report_generator.py - DELETE this entire function as it's duplicated from app.py
# and references undefined variables MEMORY_STORAGE and RUNNING_TASKS
def process_file_task(file_content, filename, material, density, draft_angle, pull_direction, task_id):
    """Background task to process files with enhanced error handling"""
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
            # Create a more user-friendly error message
            error_msg = str(e)
            if "processing.html" in error_msg.lower():
                # Specific error for missing template
                user_friendly_msg = "Missing template file. Please check your application installation."
            elif "timeout" in error_msg.lower():
                user_friendly_msg = "Processing timed out. Your model may be too complex."
            elif "memory" in error_msg.lower():
                user_friendly_msg = "Not enough memory to process this model. Try a simpler model."
            else:
                user_friendly_msg = f"Processing error: {error_msg}"
                
            MEMORY_STORAGE[f'task_status_{task_id}'] = f'error: {user_friendly_msg}'
    finally:
        # Clean up task tracking
        if task_id in RUNNING_TASKS:
            RUNNING_TASKS.pop(task_id)