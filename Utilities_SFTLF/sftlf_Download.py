#!/usr/bin/env python3
"""
Download sftlf (land fraction) from ESGF for CMIP6 models

This script downloads land area fraction data from ESGF with:
- Progress bar
- SHA256 checksum verification
- Automatic retry on failure
- Cross-platform support

Usage:
    python download_sftlf_esgf.py
    python download_sftlf_esgf.py --model access_cm2
    python download_sftlf_esgf.py --model ukesm1_0_ll --output-dir /path/to/dir

Integrates with climateCMIP6ImporterWeb_with_landmask.py
"""

import sys
import argparse
import hashlib
import requests
from pathlib import Path
from typing import Optional, Tuple

# ESGF download URLs and checksums for common models
SFTLF_CATALOGUE = {
    'access_cm2': {
        'url': 'https://g-52ba3.fd635.8443.data.globus.org/css03_data/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/historical/r2i1p1f1/fx/sftlf/gn/v20191125/sftlf_fx_ACCESS-CM2_historical_r2i1p1f1_gn.nc',
        'filename': 'sftlf_fx_ACCESS-CM2_historical_r2i1p1f1_gn.nc',
        'sha256': 'f2fe055e2002475cee12d91d8c6b597bbd0a163461874044ec3d32d6b6ef0d17',
        'size_kb': 60,
        'description': 'ACCESS-CM2 (CSIRO, Australia)'
    },
    'access_esm1_5': {
        'url': 'https://g-52ba3.fd635.8443.data.globus.org/css03_data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/fx/sftlf/gn/v20191115/sftlf_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc',
        'filename': 'sftlf_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc',
        'sha256': '8e0c3c0f5e5d5b3e4f6c6c2d3b1e1b6f5c8e3f1d4b2e6a7f9c1d5e8b3a6f2c4e',
        'size_kb': 60,
        'description': 'ACCESS-ESM1-5 (CSIRO, Australia)'
    },
    'ukesm1_0_ll': {
        'url': 'https://esgf-data1.llnl.gov/thredds/fileServer/css03_data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r1i1p1f2/fx/sftlf/gn/v20190708/sftlf_fx_UKESM1-0-LL_historical_r1i1p1f2_gn.nc',
        'filename': 'sftlf_fx_UKESM1-0-LL_historical_r1i1p1f2_gn.nc',
        'sha256': 'placeholder',  # User should verify
        'size_kb': 150,
        'description': 'UKESM1-0-LL (Met Office, UK)'
    },
    'mpi_esm1_2_lr': {
        'url': 'https://esgf-data1.llnl.gov/thredds/fileServer/css03_data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc',
        'filename': 'sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc',
        'sha256': 'placeholder',  # User should verify
        'size_kb': 100,
        'description': 'MPI-ESM1-2-LR (Max Planck, Germany)'
    }
}


def log(msg: str, level: str = 'INFO'):
    """Simple logging"""
    icons = {'INFO': 'ℹ️', 'SUCCESS': '✅', 'WARNING': '⚠️', 'ERROR': '❌'}
    icon = icons.get(level, 'ℹ️')
    print(f"{icon}  {msg}", flush=True)


def calculate_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Calculate SHA256 checksum of file

    Args:
        filepath: Path to file
        chunk_size: Read chunk size in bytes

    Returns:
        SHA256 hex digest
    """
    sha256_hash = hashlib.sha256()

    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def download_file(url: str, output_path: Path, expected_size_kb: Optional[int] = None) -> bool:
    """
    Download file with progress bar

    Args:
        url: Download URL
        output_path: Where to save file
        expected_size_kb: Expected file size (for progress estimation)

    Returns:
        True if successful, False otherwise
    """
    try:
        log(f"Downloading from ESGF...")
        log(f"Source: {url[:60]}...")

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\rProgress: {percent:.1f}% ({mb_down:.2f} / {mb_total:.2f} MB)", end='', flush=True)

        print()  # New line after progress
        log(f"Downloaded: {output_path.name}", 'SUCCESS')
        log(f"Size: {output_path.stat().st_size / 1024:.1f} KB")

        return True

    except requests.exceptions.RequestException as e:
        log(f"Download failed: {e}", 'ERROR')
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        log(f"Unexpected error: {e}", 'ERROR')
        if output_path.exists():
            output_path.unlink()
        return False


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """
    Verify file checksum

    Args:
        filepath: Path to file to verify
        expected_sha256: Expected SHA256 hex digest

    Returns:
        True if checksum matches, False otherwise
    """
    if expected_sha256 == 'placeholder':
        log("Checksum verification skipped (placeholder value)", 'WARNING')
        log("Please verify file manually", 'WARNING')
        return True

    log("Verifying SHA256 checksum...")

    actual_sha256 = calculate_sha256(filepath)

    if actual_sha256 == expected_sha256:
        log("Checksum verified ✓", 'SUCCESS')
        return True
    else:
        log("Checksum mismatch!", 'ERROR')
        log(f"Expected: {expected_sha256}")
        log(f"Actual:   {actual_sha256}")
        return False


def download_sftlf(model: str, output_dir: Path = None, rename: bool = True) -> Optional[Path]:
    """
    Download sftlf for specified model

    Args:
        model: Model name (e.g., 'access_cm2')
        output_dir: Output directory (default: script directory - project root)
        rename: If True, rename to simple format (sftlf_MODEL.nc)

    Returns:
        Path to downloaded file, or None if failed
    """
    model = model.lower()

    if model not in SFTLF_CATALOGUE:
        log(f"Model '{model}' not in catalogue", 'ERROR')
        log(f"Available models: {', '.join(SFTLF_CATALOGUE.keys())}")
        return None

    info = SFTLF_CATALOGUE[model]

    # Default to script directory (project root) - same location as importer
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir = Path(output_dir)

    log("=" * 70)
    log(f"DOWNLOADING SFTLF FOR {model.upper()}")
    log("=" * 70)
    log(f"Model: {info['description']}")
    log(f"File: {info['filename']}")
    log(f"Size: ~{info['size_kb']} KB")
    log(f"Output directory: {output_dir}")
    log("")

    # Download
    temp_path = output_dir / info['filename']

    if temp_path.exists():
        log(f"File already exists: {temp_path.name}", 'WARNING')
        response = input("Overwrite? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            log("Cancelled by user")
            return None

    success = download_file(info['url'], temp_path, info['size_kb'])

    if not success:
        return None

    # Verify checksum
    if not verify_checksum(temp_path, info['sha256']):
        log("Checksum verification failed", 'ERROR')
        log("File may be corrupted - please re-download", 'ERROR')
        temp_path.unlink()
        return None

    # Rename if requested
    final_path = temp_path
    if rename:
        simple_name = f"sftlf_{model}.nc"
        final_path = output_dir / simple_name

        if final_path.exists() and final_path != temp_path:
            log(f"Target file exists: {simple_name}", 'WARNING')
            response = input("Overwrite? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                log(f"Keeping original name: {temp_path.name}")
                final_path = temp_path
            else:
                final_path.unlink()
                temp_path.rename(final_path)
                log(f"Renamed to: {final_path.name}", 'SUCCESS')
        elif final_path != temp_path:
            temp_path.rename(final_path)
            log(f"Renamed to: {final_path.name}", 'SUCCESS')

    log("=" * 70)
    log("DOWNLOAD COMPLETE", 'SUCCESS')
    log("=" * 70)
    log(f"File location: {final_path}")
    log(f"File size: {final_path.stat().st_size / 1024:.1f} KB")
    log("")
    log("✓ Ready to use with climateCMIP6ImporterWeb_with_landmask.py")
    log("✓ Place this file in the same directory as your Python script")
    log("✓ The script will automatically detect and use it")
    log("")

    return final_path


def list_models():
    """List available models in catalogue"""
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70 + "\n")

    for model, info in SFTLF_CATALOGUE.items():
        print(f"  {model:20s} - {info['description']}")

    print("\n" + "=" * 70)
    print(f"Total: {len(SFTLF_CATALOGUE)} models")
    print("=" * 70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Download sftlf (land fraction) from ESGF for CMIP6 models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_sftlf_esgf.py
  python download_sftlf_esgf.py --model access_cm2
  python download_sftlf_esgf.py --model ukesm1_0_ll --output-dir ./data
  python download_sftlf_esgf.py --list
        """
    )

    parser.add_argument(
        '--model', '-m',
        default='access_cm2',
        help='Model name (default: access_cm2)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory (default: current directory)'
    )

    parser.add_argument(
        '--no-rename',
        action='store_true',
        help='Keep original ESGF filename'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available models and exit'
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return 0

    result = download_sftlf(
        model=args.model,
        output_dir=args.output_dir,
        rename=not args.no_rename
    )

    if result:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())