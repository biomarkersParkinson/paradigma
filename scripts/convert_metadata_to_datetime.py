"""
Script to convert old arm swing metadata format to new format with datetime support.

This script handles the conversion of old metadata files, converting them to the new
format with independent filtered/unfiltered metadata and optional datetime information.

Old format (combined filtered/unfiltered):
{
    'aggregated': {'all': {'duration_s': X}},
    'per_segment': {
        segment_nr: {
            'start_time_s': float,
            'end_time_s': float,
            'duration_unfiltered_segment_s': float,
            'duration_filtered_segment_s': float (optional)
        }
    }
}

New format (completely independent):

gait_segment_meta_unfiltered.json:
{
    'combined': {
        'duration_s': float,
        'start_dt': '2019-08-20T10:39:16Z' (optional),
        'end_dt': '2019-08-20T20:11:36Z' (optional)
    },
    'per_segment': {
        segment_nr: {
            'start_s': float,
            'end_s': float,
            'duration_s': float,
            'start_dt': '2019-08-20T10:40:01.123Z' (optional),
            'end_dt': '2019-08-20T10:41:21.456Z' (optional)
        }
    }
}

gait_segment_meta_filtered.json:
{
    'combined': {
        'duration_s': float,
        'start_dt': '2019-08-20T10:39:16Z' (optional),
        'end_dt': '2019-08-20T20:11:36Z' (optional)
    },
    'per_segment': {
        segment_nr: {
            'start_s': float,
            'end_s': float,
            'duration_s': float,
            'start_dt': '2019-08-20T10:40:01.123Z' (optional),
            'end_dt': '2019-08-20T10:41:21.456Z' (optional)
        }
    }
}

Note: Filtered and unfiltered metadata are now completely independent.
Each segment's duration_s is the actual duration of that segment only.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path


def convert_metadata_to_new_format(
    old_meta: dict,
    start_dt: datetime | None = None,
) -> dict:
    """
    Convert old metadata format to new format.

    Parameters
    ----------
    old_meta : dict
        Old metadata dictionary
    start_dt : datetime.datetime, optional
        Start datetime of the recording. If provided, absolute datetime
        values will be included in the output.

    Returns
    -------
    dict
        New format metadata dictionary
    """
    # Initialize new metadata structure
    new_meta = {
        "combined": {},
        "per_segment": {},
    }

    # Copy duration_s to combined section
    if "aggregated" in old_meta and "all" in old_meta["aggregated"]:
        new_meta["combined"]["duration_s"] = old_meta["aggregated"]["all"]["duration_s"]
    elif "all" in old_meta:
        new_meta["combined"]["duration_s"] = old_meta["all"]["duration_s"]
    else:
        # Calculate from per_segment data if needed
        total_duration = 0
        if "per_segment" in old_meta:
            for seg_data in old_meta["per_segment"].values():
                if "duration_unfiltered_segment_s" in seg_data:
                    total_duration = max(total_duration, seg_data.get("end_time_s", 0))
        new_meta["combined"]["duration_s"] = total_duration

    # Add datetime to combined if available
    if start_dt is not None and "per_segment" in old_meta:
        # Find min/max times from all segments
        min_time_s = None
        max_time_s = None
        for seg_data in old_meta["per_segment"].values():
            start_s = seg_data.get("start_time_s") or seg_data.get("start_s")
            end_s = seg_data.get("end_time_s") or seg_data.get("end_s")
            if start_s is not None:
                min_time_s = start_s if min_time_s is None else min(min_time_s, start_s)
            if end_s is not None:
                max_time_s = end_s if max_time_s is None else max(max_time_s, end_s)

        if min_time_s is not None:
            new_meta["combined"]["start_dt"] = (
                start_dt + timedelta(seconds=float(min_time_s))
            ).isoformat() + "Z"
        if max_time_s is not None:
            new_meta["combined"]["end_dt"] = (
                start_dt + timedelta(seconds=float(max_time_s))
            ).isoformat() + "Z"

    # Convert per_segment data
    if "per_segment" in old_meta:
        for segment_nr, seg_data in old_meta["per_segment"].items():
            new_seg_data = {}

            # Rename start/end time fields
            start_time_key = "start_time_s" if "start_time_s" in seg_data else "start_s"
            end_time_key = "end_time_s" if "end_time_s" in seg_data else "end_s"

            new_seg_data["start_s"] = seg_data[start_time_key]
            new_seg_data["end_s"] = seg_data[end_time_key]

            # Determine appropriate duration field
            # Prefer duration_filtered_segment_s if present, otherwise use duration_unfiltered_segment_s
            if "duration_filtered_segment_s" in seg_data:
                new_seg_data["duration_s"] = seg_data["duration_filtered_segment_s"]
            elif "duration_unfiltered_segment_s" in seg_data:
                new_seg_data["duration_s"] = seg_data["duration_unfiltered_segment_s"]
            else:
                # Fallback: calculate from start/end times
                new_seg_data["duration_s"] = (
                    seg_data[end_time_key] - seg_data[start_time_key]
                )

            # Add datetime if available
            if start_dt is not None:
                start_s = new_seg_data["start_s"]
                end_s = new_seg_data["end_s"]
                new_seg_data["start_dt"] = (
                    start_dt + timedelta(seconds=float(start_s))
                ).isoformat() + "Z"
                new_seg_data["end_dt"] = (
                    start_dt + timedelta(seconds=float(end_s))
                ).isoformat() + "Z"
                # Remove relative time fields when datetime is available
                del new_seg_data["start_s"]
                del new_seg_data["end_s"]

            new_meta["per_segment"][segment_nr] = new_seg_data

    return new_meta


def convert_file(
    input_file: Path,
    output_file: Path,
    start_dt: datetime | None = None,
) -> None:
    """
    Convert a single metadata file from old to new format.

    Parameters
    ----------
    input_file : Path
        Path to input metadata file
    output_file : Path
        Path to output metadata file
    start_dt : datetime.datetime, optional
        Start datetime of the recording
    """
    # Load old metadata
    with open(input_file) as f:
        old_meta = json.load(f)

    # Convert to new format
    new_meta = convert_metadata_to_new_format(old_meta, start_dt)

    # Save new metadata
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(new_meta, f, indent=2)

    print(f"Converted: {input_file} → {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert old arm swing metadata to new format with datetime support"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input metadata JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output metadata file (default: input_file with '_new' suffix)",
    )
    parser.add_argument(
        "-s",
        "--start-datetime",
        type=str,
        default=None,
        help="Start datetime in ISO8601 format (e.g., '2019-08-20T10:39:16Z')",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=None,
        help="Convert all .json files in directory",
    )

    args = parser.parse_args()

    # Parse start datetime if provided
    start_dt = None
    if args.start_datetime:
        start_dt = datetime.fromisoformat(args.start_datetime.rstrip("Z"))

    # Handle directory conversion
    if args.directory:
        print(f"Converting all metadata files in {args.directory}...")
        for json_file in args.directory.glob("*_meta.json"):
            output_file = json_file.parent / json_file.name.replace(
                "_meta.json", "_meta_converted.json"
            )
            convert_file(json_file, output_file, start_dt)
    else:
        # Handle single file conversion
        input_file = args.input_file
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Determine output file
        if args.output is None:
            output_file = input_file.parent / (input_file.stem + "_converted.json")
        else:
            output_file = args.output

        convert_file(input_file, output_file, start_dt)
        print("\nConversion complete!")
        print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
