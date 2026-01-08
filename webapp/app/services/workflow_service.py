"""
Workflow templates and history management service.

Provides functionality for saving, loading, and managing workflow templates
and execution history.
"""
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkflowService:
    """Service for managing workflow templates and history."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize workflow service.
        
        Parameters
        ----------
        storage_dir : str, optional
            Directory for storing workflows. Defaults to ./workflows
        """
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'workflows')
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates_dir = self.storage_dir / 'templates'
        self.history_dir = self.storage_dir / 'history'
        
        self.templates_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
    
    def save_template(
        self,
        name: str,
        workflow: Dict[str, Any],
        description: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Save a workflow template.
        
        Parameters
        ----------
        name : str
            Template name
        workflow : dict
            Workflow definition (steps, parameters, etc.)
        description : str, optional
            Template description
        category : str, optional
            Template category (e.g., 'petrophysics', 'geomechanics')
            
        Returns
        -------
        str
            Template ID
        """
        template_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        template_data = {
            'id': template_id,
            'name': name,
            'description': description,
            'category': category,
            'workflow': workflow,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        template_file = self.templates_dir / f"{template_id}.json"
        
        with open(template_file, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"Saved workflow template: {template_id}")
        return template_id
    
    def load_template(self, template_id: str) -> Dict[str, Any]:
        """
        Load a workflow template.
        
        Parameters
        ----------
        template_id : str
            Template ID
            
        Returns
        -------
        dict
            Template data
        """
        template_file = self.templates_dir / f"{template_id}.json"
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template {template_id} not found")
        
        with open(template_file, 'r') as f:
            return json.load(f)
    
    def list_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all workflow templates.
        
        Parameters
        ----------
        category : str, optional
            Filter by category
            
        Returns
        -------
        list of dict
            List of template metadata
        """
        templates = []
        
        for template_file in self.templates_dir.glob('*.json'):
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                
                if category is None or template_data.get('category') == category:
                    templates.append({
                        'id': template_data['id'],
                        'name': template_data['name'],
                        'description': template_data.get('description'),
                        'category': template_data.get('category'),
                        'created_at': template_data.get('created_at')
                    })
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")
        
        return sorted(templates, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def save_execution(
        self,
        template_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save workflow execution history.
        
        Parameters
        ----------
        template_id : str
            Template ID used
        inputs : dict
            Input parameters
        outputs : dict
            Output results
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        str
            Execution ID
        """
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        execution_data = {
            'id': execution_id,
            'template_id': template_id,
            'inputs': inputs,
            'outputs': outputs,
            'metadata': metadata or {},
            'executed_at': datetime.now().isoformat()
        }
        
        execution_file = self.history_dir / f"{execution_id}.json"
        
        with open(execution_file, 'w') as f:
            json.dump(execution_data, f, indent=2)
        
        logger.info(f"Saved workflow execution: {execution_id}")
        return execution_id
    
    def get_execution_history(
        self,
        template_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get workflow execution history.
        
        Parameters
        ----------
        template_id : str, optional
            Filter by template ID
        limit : int, default 50
            Maximum number of executions to return
            
        Returns
        -------
        list of dict
            List of execution records
        """
        executions = []
        
        for execution_file in sorted(self.history_dir.glob('*.json'), reverse=True):
            try:
                with open(execution_file, 'r') as f:
                    execution_data = json.load(f)
                
                if template_id is None or execution_data.get('template_id') == template_id:
                    executions.append({
                        'id': execution_data['id'],
                        'template_id': execution_data.get('template_id'),
                        'executed_at': execution_data.get('executed_at'),
                        'metadata': execution_data.get('metadata', {})
                    })
                
                if len(executions) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Failed to load execution {execution_file}: {e}")
        
        return executions
    
    def delete_template(self, template_id: str) -> bool:
        """
        Delete a workflow template.
        
        Parameters
        ----------
        template_id : str
            Template ID to delete
            
        Returns
        -------
        bool
            True if deleted, False if not found
        """
        template_file = self.templates_dir / f"{template_id}.json"
        
        if template_file.exists():
            template_file.unlink()
            logger.info(f"Deleted template: {template_id}")
            return True
        
        return False

