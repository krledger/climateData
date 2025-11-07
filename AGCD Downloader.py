#!/usr/bin/env python3
"""
Download ALL AGCD data from NCI THREDDS server
- Precipitation: 1900-2024 (125 files)
- Temperature: 1910-2024 (115 files each for tmax/tmin)
Total: ~355 files, ~25-30 GB
"""

import os
import sys
import time
import ssl
from pathlib import Path
import urllib.request
import urllib.error

# Configuration
OUTPUT_DIR = Path("agcd_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Create SSL context that doesn't verify certificates
# This is needed for some institutional/corporate networks
ssl_context = ssl._create_unverified_context()

# Variable definitions
VARIABLES = {
    'precip': {
        'base_url': 'https://thredds.nci.org.au/thredds/fileServer/zv2/agcd/v1-0-3/precip/total/r005/01day',
        'start_year': 1900,
        'end_year': 2024,
        'filename_pattern': 'agcd_v1_precip_total_r005_daily_{year}.nc'
    },
    'tmax': {
        'base_url': 'https://thredds.nci.org.au/thredds/fileServer/zv2/agcd/v1-0-3/tmax/mean/r005/01day',
        'start_year': 1910,
        'end_year': 2024,
        'filename_pattern': 'agcd_v1_tmax_mean_r005_daily_{year}.nc'
    },
    'tmin': {
        'base_url': 'https://thredds.nci.org.au/thredds/fileServer/zv2/agcd/v1-0-3/tmin/mean/r005/01day',
        'start_year': 1910,
        'end_year': 2024,
        'filename_pattern': 'agcd_v1_tmin_mean_r005_daily_{year}.nc'
    },
    'vprp09': {
        'base_url': 'https://thredds.nci.org.au/thredds/fileServer/zv2/agcd/v1-0-3/vapourpres_h09/mean/r005/01day',
        'start_year': 1971,
        'end_year': 2024,
        'filename_pattern': 'agcd_v1_vapourpres_h09_mean_r005_daily_{year}.nc'
    }
}


def download_file(url, output_path):
    """Download file with progress and SSL handling"""
    try:
        print(f"      Downloading...", end="", flush=True)

        # Open URL with SSL context
        with urllib.request.urlopen(url, context=ssl_context) as response:
            # Write to file
            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())

        # Check file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f" ✓ {size_mb:.1f} MB", flush=True)
        return True, size_mb

    except urllib.error.HTTPError as e:
        print(f" ✗ HTTP {e.code}", flush=True)
        return False, f"HTTP {e.code}"
    except Exception as e:
        print(f" ✗ {str(e)[:50]}", flush=True)
        return False, str(e)


def download_variable(var_name, config):
    """Download all years for a variable"""
    print(f"\n{'=' * 70}", flush=True)
    total_years = config['end_year'] - config['start_year'] + 1
    print(f"{var_name.upper()}: {config['start_year']}-{config['end_year']}", flush=True)
    print(f"Total files: {total_years}", flush=True)
    print(f"{'=' * 70}", flush=True)

    stats = {'success': 0, 'failed': 0, 'skipped': 0, 'total_mb': 0}
    start_time = time.time()

    for idx, year in enumerate(range(config['start_year'], config['end_year'] + 1), 1):
        filename = config['filename_pattern'].format(year=year)
        url = f"{config['base_url']}/{filename}"
        output_path = OUTPUT_DIR / filename

        # Progress indicator
        percent = (idx / total_years) * 100
        print(f"  [{idx:3d}/{total_years}] {percent:5.1f}% | {year}: ", end="", flush=True)

        # Skip if already exists
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            stats['skipped'] += 1
            stats['total_mb'] += size_mb
            print(f"Already exists ({size_mb:.1f} MB)", flush=True)
            continue

        # Download
        success, result = download_file(url, output_path)

        if success:
            stats['success'] += 1
            stats['total_mb'] += result
        else:
            stats['failed'] += 1
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()

        # Time estimate every 10 files
        if idx % 10 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total_years - idx) / rate if rate > 0 else 0
            print(f"      --> Time remaining (est): {remaining / 60:.1f} minutes", flush=True)

    elapsed = time.time() - start_time
    print(f"\n{var_name.upper()} COMPLETE:", flush=True)
    print(f"  ✓ Downloaded: {stats['success']} files", flush=True)
    print(f"  ⊗ Skipped: {stats['skipped']} files", flush=True)
    print(f"  ✗ Failed: {stats['failed']} files", flush=True)
    print(f"  Total size: {stats['total_mb']:.1f} MB ({stats['total_mb'] / 1024:.2f} GB)", flush=True)
    print(f"  Time: {elapsed / 60:.1f} minutes", flush=True)

    return stats


def main():
    print("\n" + "=" * 70, flush=True)
    print("AGCD DATA DOWNLOAD - ALL VARIABLES", flush=True)
    print("=" * 70, flush=True)

    # Show environment info
    print(f"\nPython: {sys.version.split()[0]}", flush=True)
    print(f"Working directory: {Path.cwd()}", flush=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}", flush=True)
    print("SSL certificate verification: DISABLED (for compatibility)", flush=True)

    print("\nVariables to download:", flush=True)
    print("  • Precipitation (1900-2024): 125 files", flush=True)
    print("  • Temperature Max (1910-2024): 115 files", flush=True)
    print("  • Temperature Min (1910-2024): 115 files", flush=True)
    print("  • Vapour Pressure 9am (1971-2024): 54 files", flush=True)
    print("  • Total: 409 files (~29 GB)", flush=True)
    print("=" * 70, flush=True)

    response = input("\nProceed with download? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return 0

    # Download each variable
    overall_stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_mb': 0
    }

    start_time = time.time()

    for var_name, config in VARIABLES.items():
        stats = download_variable(var_name, config)
        overall_stats['success'] += stats['success']
        overall_stats['failed'] += stats['failed']
        overall_stats['skipped'] += stats['skipped']
        overall_stats['total_mb'] += stats['total_mb']

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 70}")
    print(f"Downloaded: {overall_stats['success']} files ({overall_stats['total_mb'] / 1024:.1f} GB)")
    print(f"Skipped (already exists): {overall_stats['skipped']} files")
    print(f"Failed: {overall_stats['failed']} files")
    print(f"Time: {elapsed / 60:.1f} minutes")
    print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")

    if overall_stats['success'] > 0 or overall_stats['skipped'] > 0:
        print("\n" + "=" * 70)
        print("NEXT STEP")
        print("=" * 70)
        print("Run the AGCD importer to process these files:")
        print("  python AGCDImporter.py")
        print("=" * 70)

    return 0 if overall_stats['failed'] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)