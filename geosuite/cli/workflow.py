"""
CLI command for running workflows from YAML/JSON files.
"""

import argparse
import logging
import sys
from pathlib import Path

from ..workflows import run_workflow, WorkflowOrchestrator
from ..config import load_config


def main():
    """Command-line entry point for workflow execution."""
    parser = argparse.ArgumentParser(
        description="Execute GeoSuite workflows from YAML/JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a workflow
  geosuite-workflow petrophysical_workflow.yaml
  
  # Run with custom config
  geosuite-workflow workflow.yaml --config custom_config.yaml
  
  # Run with verbose output
  geosuite-workflow workflow.yaml --verbose
  
  # Dry run (validate without executing)
  geosuite-workflow workflow.yaml --dry-run
        """
    )
    
    parser.add_argument(
        "workflow_file",
        type=Path,
        help="Path to workflow YAML or JSON file"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML/JSON file (optional)"
    )
    
    parser.add_argument(
        "--working-dir",
        type=Path,
        help="Working directory for relative paths (default: workflow file directory)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate workflow without executing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate workflow file exists
    if not args.workflow_file.exists():
        logger.error(f"Workflow file not found: {args.workflow_file}")
        sys.exit(1)
    
    # Determine working directory
    working_dir = args.working_dir or args.workflow_file.parent
    
    # Load config if provided
    config = None
    if args.config:
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(config=config, working_dir=working_dir)
    
    if args.dry_run:
        # Just validate the workflow
        try:
            workflow = orchestrator.load_workflow_file(args.workflow_file)
            logger.info("✓ Workflow file is valid")
            logger.info(f"  Name: {workflow.get('name', 'Unnamed')}")
            logger.info(f"  Steps: {len(workflow.get('steps', []))}")
            print("\nWorkflow validation successful!")
        except Exception as e:
            logger.error(f"✗ Workflow validation failed: {e}")
            sys.exit(1)
    else:
        # Execute workflow
        try:
            logger.info(f"Executing workflow: {args.workflow_file}")
            results = orchestrator.execute_file(args.workflow_file)
            
            logger.info("\n" + "=" * 60)
            logger.info("Workflow Execution Summary")
            logger.info("=" * 60)
            logger.info(f"Steps executed: {len(results)}")
            for step_name, result in results.items():
                if isinstance(result, (int, float, str)):
                    logger.info(f"  {step_name}: {result}")
                elif hasattr(result, 'shape'):
                    logger.info(f"  {step_name}: array shape {result.shape}")
                else:
                    logger.info(f"  {step_name}: completed")
            
            print("\n✓ Workflow completed successfully!")
            
        except Exception as e:
            logger.error(f"\n✗ Workflow execution failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

