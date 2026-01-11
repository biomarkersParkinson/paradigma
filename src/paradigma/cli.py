"""
Command Line Interface for ParaDigMa toolbox.

This module provides CLI commands for running ParaDigMa pipelines on sensor data.
"""

import argparse
import json
import sys
from pathlib import Path

from paradigma.pipeline import list_available_pipelines, run_pipeline


def main():
    """Main entry point for the ParaDigMa CLI."""
    parser = argparse.ArgumentParser(
        description="ParaDigMa - Parkinson's Disease Digital Markers Toolbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run gait pipeline on TSDF data
  paradigma run data/tsdf/ --pipelines gait --output results/

  # Run on prepared dataframes (parquet files)
  paradigma run data/prepared/ --pipelines gait tremor --data-format prepared --file-pattern "*.parquet" --output results/

  # Run on Axivity CWA files with verbose output
  paradigma run data/axivity/ --pipelines gait --data-format axivity --verbose --output results/

  # Use column mapping for different naming conventions
  paradigma run data/tsdf/ --pipelines gait --column-mapping '{"acceleration_x": "accelerometer_x", "rotation_x": "gyroscope_x"}'

  # Auto-detect data format
  paradigma run data/mixed/ --pipelines gait tremor --output results/

  # List available pipelines
  paradigma list-pipelines
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Run ParaDigMa pipelines on sensor data"
    )
    run_parser.add_argument(
        "data_path", type=str, help="Path to directory or file containing sensor data"
    )
    run_parser.add_argument(
        "--pipelines",
        nargs="+",
        required=True,
        choices=list_available_pipelines(),
        help="Pipeline(s) to run",
    )
    run_parser.add_argument(
        "--output", "-o", type=str, help="Output directory for results (optional)"
    )
    run_parser.add_argument(
        "--data-format",
        type=str,
        choices=["tsdf", "empatica", "axivity", "prepared"],
        help="Data format (auto-detected if not specified). Options: tsdf, empatica, axivity, prepared",
    )
    run_parser.add_argument(
        "--file-pattern",
        type=str,
        help='File pattern for prepared dataframes (e.g., "*.parquet", "*.pkl")',
    )
    run_parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file (optional)"
    )
    run_parser.add_argument(
        "--column-mapping",
        type=str,
        help='Column mapping as JSON string or path to JSON file (e.g., \'{"acceleration_x": "accelerometer_x"}\')',
    )
    run_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing where supported",
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # List pipelines command
    subparsers.add_parser("list-pipelines", help="List available ParaDigMa pipelines")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        run_command(args)
    elif args.command == "list-pipelines":
        list_pipelines_command()


def run_command(args):
    """Execute the run command."""
    try:
        # Load configuration if provided
        config = "default"
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"Error: Configuration file not found: {config_path}")
                sys.exit(1)

            with open(config_path, "r") as f:
                config = json.load(f)

        # Parse column mapping if provided
        column_mapping = None
        if args.column_mapping:
            if args.column_mapping.startswith("{"):
                # JSON string
                try:
                    column_mapping = json.loads(args.column_mapping)
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in column mapping: {e}")
                    sys.exit(1)
            else:
                # File path
                mapping_path = Path(args.column_mapping)
                if not mapping_path.exists():
                    print(f"Error: Column mapping file not found: {mapping_path}")
                    sys.exit(1)
                with open(mapping_path, "r") as f:
                    column_mapping = json.load(f)

        # Run the pipelines
        print(f"Running ParaDigMa pipelines: {', '.join(args.pipelines)}")
        print(f"Data path: {args.data_path}")
        if args.output:
            print(f"Output directory: {args.output}")

        results = run_pipeline(
            data_path=args.data_path,
            pipelines=args.pipelines,
            config=config,
            output_dir=args.output,
            data_format=args.data_format,
            file_pattern=args.file_pattern,
            column_mapping=column_mapping,
            parallel=args.parallel,
            verbose=args.verbose,
        )

        print("\\nPipeline execution completed successfully!")

        # Print summary of results
        for pipeline_name, result_df in results.items():
            if not result_df.empty:
                print(f"  {pipeline_name}: {len(result_df)} rows of results")
            else:
                print(f"  {pipeline_name}: No results generated")

    except KeyboardInterrupt:
        print("\\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def list_pipelines_command():
    """Execute the list-pipelines command."""
    pipelines = list_available_pipelines()
    print("Available ParaDigMa pipelines:")
    for pipeline in pipelines:
        print(f"  - {pipeline}")


if __name__ == "__main__":
    main()
