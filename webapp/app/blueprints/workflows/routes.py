"""
Workflow templates and history routes.
"""
import logging
from flask import render_template, request, jsonify, send_file
from . import bp
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

try:
    from app.services.workflow_service import WorkflowService
    from app.services.export_service import ExportService
    WORKFLOW_SERVICE_AVAILABLE = True
except ImportError:
    WORKFLOW_SERVICE_AVAILABLE = False
    logger.warning("Workflow services not available")

logger = logging.getLogger(__name__)

# Initialize services
workflow_service = WorkflowService() if WORKFLOW_SERVICE_AVAILABLE else None
export_service = ExportService() if WORKFLOW_SERVICE_AVAILABLE else None


@bp.route("/")
def workflows_home():
    """Workflow templates home page."""
    return render_template("workflows/index.html")


@bp.route("/api/templates", methods=["GET"])
def list_templates():
    """List all workflow templates."""
    if not workflow_service:
        return jsonify({'error': 'Workflow service not available'}), 503
    
    category = request.args.get('category')
    templates = workflow_service.list_templates(category=category)
    
    return jsonify({'templates': templates})


@bp.route("/api/templates", methods=["POST"])
def create_template():
    """Create a new workflow template."""
    if not workflow_service:
        return jsonify({'error': 'Workflow service not available'}), 503
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    name = data.get('name')
    workflow = data.get('workflow')
    description = data.get('description')
    category = data.get('category')
    
    if not name or not workflow:
        return jsonify({'error': 'name and workflow are required'}), 400
    
    template_id = workflow_service.save_template(
        name=name,
        workflow=workflow,
        description=description,
        category=category
    )
    
    return jsonify({'template_id': template_id}), 201


@bp.route("/api/templates/<template_id>", methods=["GET"])
def get_template(template_id):
    """Get a workflow template."""
    if not workflow_service:
        return jsonify({'error': 'Workflow service not available'}), 503
    
    try:
        template = workflow_service.load_template(template_id)
        return jsonify(template)
    except FileNotFoundError:
        return jsonify({'error': 'Template not found'}), 404


@bp.route("/api/templates/<template_id>", methods=["DELETE"])
def delete_template(template_id):
    """Delete a workflow template."""
    if not workflow_service:
        return jsonify({'error': 'Workflow service not available'}), 503
    
    deleted = workflow_service.delete_template(template_id)
    
    if deleted:
        return jsonify({'message': 'Template deleted'}), 200
    else:
        return jsonify({'error': 'Template not found'}), 404


@bp.route("/api/history", methods=["GET"])
def get_history():
    """Get workflow execution history."""
    if not workflow_service:
        return jsonify({'error': 'Workflow service not available'}), 503
    
    template_id = request.args.get('template_id')
    limit = int(request.args.get('limit', 50))
    
    history = workflow_service.get_execution_history(
        template_id=template_id,
        limit=limit
    )
    
    return jsonify({'history': history})


@bp.route("/api/execute", methods=["POST"])
def execute_workflow():
    """Execute a workflow template."""
    if not workflow_service:
        return jsonify({'error': 'Workflow service not available'}), 503
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    template_id = data.get('template_id')
    inputs = data.get('inputs', {})
    
    if not template_id:
        return jsonify({'error': 'template_id is required'}), 400
    
    try:
        template = workflow_service.load_template(template_id)
        workflow = template['workflow']
        
        # Execute workflow steps (simplified - would need full workflow engine)
        # For now, just save execution record
        outputs = {'status': 'executed', 'workflow': workflow}
        
        execution_id = workflow_service.save_execution(
            template_id=template_id,
            inputs=inputs,
            outputs=outputs
        )
        
        return jsonify({
            'execution_id': execution_id,
            'outputs': outputs
        }), 200
    except FileNotFoundError:
        return jsonify({'error': 'Template not found'}), 404


@bp.route("/api/export", methods=["POST"])
def export_results():
    """Export analysis results."""
    if not export_service:
        return jsonify({'error': 'Export service not available'}), 503
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    analysis_type = data.get('analysis_type', 'analysis')
    inputs = data.get('inputs', {})
    results = data.get('results', {})
    format_type = data.get('format', 'json')
    
    try:
        if format_type == 'pdf':
            file_content = export_service.export_to_pdf_report(
                title=f"{analysis_type.title()} Analysis Report",
                sections=[
                    {'title': 'Input Parameters', 'content': str(inputs)},
                    {'title': 'Results', 'content': str(results)}
                ]
            )
            mimetype = 'application/pdf'
            filename = f"{analysis_type}_report.pdf"
        elif format_type == 'excel':
            file_content = export_service.export_to_excel(results)
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename = f"{analysis_type}_results.xlsx"
        elif format_type == 'csv':
            file_content = export_service.export_to_csv(results)
            mimetype = 'text/csv'
            filename = f"{analysis_type}_results.csv"
        else:
            file_content = export_service.export_to_json(results)
            mimetype = 'application/json'
            filename = f"{analysis_type}_results.json"
        
        return send_file(
            io.BytesIO(file_content),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({'error': str(e)}), 500

