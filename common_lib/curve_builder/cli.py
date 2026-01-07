"""
CLI tool for curve_builder package.
Provides commands for validation, hash checking, and migration management.
"""

import click
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from common_lib.curve_builder.python.validator import validate_sql_python_parity, ParityValidator
from common_lib.curve_builder.sql.hash_validator import SQLHashValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
def cli(debug):
    """Curve Builder CLI - Manage SQL/Python parity and migrations."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--db-url', envvar='PG_DSN', help='Database connection URL')
@click.option('--sample-time', help='Sample time for validation (ISO format)')
def validate(db_url, sample_time):
    """Validate SQL/Python parity for curve calculations."""
    if not db_url:
        click.echo("Error: Database URL required (set PG_DSN or use --db-url)")
        sys.exit(1)
    
    # Parse sample time if provided
    if sample_time:
        try:
            sample_ts = pd.Timestamp(sample_time)
        except Exception as e:
            click.echo(f"Error parsing sample time: {e}")
            sys.exit(1)
    else:
        sample_ts = pd.Timestamp.now(tz='UTC')
    
    click.echo(f"Validating SQL/Python parity at {sample_ts}...")
    
    try:
        is_valid = validate_sql_python_parity(db_url, sample_ts)
        
        if is_valid:
            click.echo("✅ All validations passed!")
            sys.exit(0)
        else:
            click.echo("❌ Validation failed! Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error during validation: {e}")
        logger.exception("Validation error")
        sys.exit(1)


@cli.command()
@click.option('--update/--no-update', default=False, help='Update registry with current hashes')
def check_hashes(update):
    """Check SQL file hashes against registry."""
    validator = SQLHashValidator(project_root)
    
    # Load registry
    registry_path = project_root / 'common_lib' / 'curve_builder' / 'sql' / 'registry.yaml'
    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)
    
    # Validate all files
    all_valid, changed_files, updated_hashes = validator.validate_all(
        registry['queries']
    )
    
    # Generate report
    report = validator.generate_hash_report(registry['queries'])
    click.echo(report)
    
    if changed_files:
        click.echo(f"\n⚠️  Changed files detected: {', '.join(changed_files)}")
        click.echo("Python implementations may need to be updated!")
    
    if update and updated_hashes:
        click.echo("\nUpdating registry with current hashes...")
        
        # Update registry
        for query_name, hash_info in updated_hashes.items():
            registry['queries'][query_name].update(hash_info)
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            yaml.safe_dump(registry, f, default_flow_style=False)
        
        click.echo("✅ Registry updated successfully!")
    
    sys.exit(0 if all_valid else 1)


@cli.command()
@click.argument('query_name')
def show_sql(query_name):
    """Display the SQL for a registered query."""
    # Load registry
    registry_path = project_root / 'common_lib' / 'curve_builder' / 'sql' / 'registry.yaml'
    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)
    
    if query_name not in registry['queries']:
        click.echo(f"Error: Query '{query_name}' not found in registry")
        click.echo(f"Available queries: {', '.join(registry['queries'].keys())}")
        sys.exit(1)
    
    query_info = registry['queries'][query_name]
    if query_info.get('path') is None:
        click.echo(f"Query '{query_name}' has no associated SQL file")
        sys.exit(1)
    
    sql_path = project_root / query_info['path']
    if not sql_path.exists():
        click.echo(f"Error: SQL file not found: {query_info['path']}")
        sys.exit(1)
    
    # Display SQL
    click.echo(f"\n{query_name}: {query_info.get('description', 'No description')}")
    click.echo(f"File: {query_info['path']}")
    click.echo("-" * 80)
    
    with open(sql_path, 'r') as f:
        click.echo(f.read())


@cli.command()
@click.option('--db-url', envvar='PG_DSN', help='Database connection URL')
@click.option('--start-time', required=True, help='Start time (ISO format)')
@click.option('--end-time', required=True, help='End time (ISO format)')
def test_python(db_url, start_time, end_time):
    """Test Python curve calculation on real data."""
    if not db_url:
        click.echo("Error: Database URL required (set PG_DSN or use --db-url)")
        sys.exit(1)
    
    try:
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
    except Exception as e:
        click.echo(f"Error parsing times: {e}")
        sys.exit(1)
    
    click.echo(f"Testing Python curve calculation from {start_ts} to {end_ts}...")
    
    try:
        import psycopg2
        from common_lib.curve_builder.python.curve_calculator import CurveCalculator
        
        calculator = CurveCalculator()
        
        # Get data from database
        with psycopg2.connect(db_url) as conn:
            # Get bar data
            bar_query = """
                SELECT ts, instrument_id, high, low, close, volume
                FROM market_data.bars_5s
                WHERE ts >= %s AND ts <= %s
            """
            bar_df = pd.read_sql(bar_query, conn, params=(start_ts, end_ts), parse_dates=['ts'])
            
            # Get continuous contract mapping
            mapping_query = """
                SELECT bucket, continuous_position, instrument_id
                FROM market_data.v_continuous_contracts_5s
                WHERE bucket >= %s AND bucket <= %s
            """
            mapping_df = pd.read_sql(mapping_query, conn, params=(start_ts, end_ts), parse_dates=['bucket'])
        
        # Calculate curve
        result_df = calculator.calculate_curve_mid_5s(bar_df, mapping_df, end_ts)
        
        click.echo(f"\nCalculated {len(result_df)} curve points")
        if len(result_df) > 0:
            click.echo("\nSample results (first 5 rows):")
            click.echo(result_df.head().to_string())
            
            # Validation
            is_valid, quality_score, msg = calculator.validate_curve_data(result_df)
            click.echo(f"\nValidation: {'✅ Passed' if is_valid else '❌ Failed'}")
            click.echo(f"Quality Score: {quality_score:.3f}")
            click.echo(f"Message: {msg}")
        
    except Exception as e:
        click.echo(f"Error during test: {e}")
        logger.exception("Test error")
        sys.exit(1)


@cli.command()
def init():
    """Initialize curve_builder for a new project."""
    click.echo("Initializing curve_builder...")
    
    # Check if already initialized
    registry_path = project_root / 'common_lib' / 'curve_builder' / 'sql' / 'registry.yaml'
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f)
        
        # Calculate initial hashes
        validator = SQLHashValidator(project_root)
        _, _, updated_hashes = validator.validate_all(registry['queries'])
        
        # Update registry with hashes
        for query_name, hash_info in updated_hashes.items():
            registry['queries'][query_name].update(hash_info)
        
        with open(registry_path, 'w') as f:
            yaml.safe_dump(registry, f, default_flow_style=False)
        
        click.echo("✅ Initialized SQL file hashes in registry")
    else:
        click.echo("❌ Registry file not found!")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()