#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""
Download GLTF assets from Sketchfab URLs listed in a text file.

Preferred input format:
  - One mesh per line as `unique_mesh_name: https://sketchfab.com/...`
  - Empty lines and lines starting with `#` are ignored.

Legacy input format is still supported:
  - Lines without a URL are treated as section labels for the following URLs.
  - Lines starting with https://sketchfab.com are model URLs.
  - Empty lines are ignored.

For the preferred format, folder names come directly from `unique_mesh_name`.
For the legacy format, folder names are derived from the section label. When
multiple URLs share the same label, they are numbered _0, _1, … (for example
`savoy_cabbage_0`).

Usage
-----
    python scripts/datagen/utils/download_sketchfab.py \\
        --input assets/assets.txt \\
        --output_dir $GLOSS_DATA_DIR/mesh \\
        --api_token YOUR_SKETCHFAB_API_TOKEN \\
        --report assets/downloaded.csv

    # Dry-run: print what would be downloaded without actually downloading
    python scripts/datagen/utils/download_sketchfab.py ... --dry_run

Requirements
------------
    pip install requests
"""

import argparse
import csv
import io
import json
import logging
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

# Local import – kept lazy-friendly (the module lives in the same directory).
try:
    from mesh_status import update_download  # type: ignore
except ImportError:  # pragma: no cover - allows importing this module without sibling
    update_download = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SKETCHFAB_API = "https://api.sketchfab.com/v3"
UID_RE = re.compile(r"[0-9a-f]{32}", re.IGNORECASE)
DIRECT_ENTRY_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*:\s*(?P<url>https?://sketchfab\.com/\S+)\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_uid(url: str) -> str | None:
    """Return the 32-char hex model UID from a Sketchfab URL, or None."""
    m = UID_RE.search(url)
    return m.group(0).lower() if m else None


def label_to_slug(label: str) -> str:
    """Normalise a section label to a filesystem-safe slug."""
    # Strip bracket annotations e.g. "[uv masking needed]"
    label = re.sub(r"\[.*?\]", "", label)
    label = label.strip().lower()
    # Replace spaces and hyphens with underscores; collapse runs
    label = re.sub(r"[\s\-]+", "_", label)
    # Strip any remaining non-alphanumeric/underscore characters
    label = re.sub(r"[^\w]", "", label)
    return label


def parse_input_file(txt_path: Path) -> list[dict]:
    """
    Parse the input file and return a list of entries:
        {url, uid, label, folder_name}

    Preferred folder naming rule:
      - `unique_mesh_name: https://sketchfab.com/...` → "{unique_mesh_name}"

    Legacy folder naming rules:
      - Single URL under a label  → "{slug}"
      - Multiple URLs under a label → "{slug}_0", "{slug}_1", …
    """
    direct_entries: list[dict] = []
    # First pass: collect (label, url) pairs
    raw: list[tuple[str, str]] = []
    current_label = "unknown"
    direct_names: set[str] = set()
    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = DIRECT_ENTRY_RE.match(line)
        if match:
            mesh_name = match.group("name")
            url = match.group("url")
            uid = extract_uid(url)
            if uid is None:
                raise ValueError(f"Could not extract Sketchfab UID from line: {raw_line}")
            if mesh_name in direct_names:
                raise ValueError(f"Duplicate mesh name in {txt_path}: {mesh_name}")
            direct_names.add(mesh_name)
            direct_entries.append(
                {
                    "url": url,
                    "uid": uid,
                    "label": mesh_name,
                    "folder_name": mesh_name,
                }
            )
            continue
        if line.lower().startswith("https://sketchfab.com"):
            url = line.split()[0]  # drop trailing whitespace / annotations
            raw.append((current_label, url))
        else:
            current_label = line

    # Count occurrences per slug so we can decide whether to add an index
    slug_total: dict[str, int] = {}
    for label, _ in raw:
        slug = label_to_slug(label)
        slug_total[slug] = slug_total.get(slug, 0) + 1

    slug_seen: dict[str, int] = {}
    entries = list(direct_entries)
    for label, url in raw:
        slug = label_to_slug(label)
        uid = extract_uid(url)
        if uid is None:
            log.warning("Could not extract UID from URL, skipping: %s", url)
            continue
        if slug_total[slug] == 1:
            folder_name = slug
        else:
            idx = slug_seen.get(slug, 0)
            folder_name = f"{slug}_{idx}"
            slug_seen[slug] = idx + 1
        entries.append({"url": url, "uid": uid, "label": label, "folder_name": folder_name})

    folder_names = [entry["folder_name"] for entry in entries]
    if len(folder_names) != len(set(folder_names)):
        raise ValueError(f"Duplicate output folder names detected in {txt_path}: {folder_names}")

    return entries


# ---------------------------------------------------------------------------
# Sketchfab API
# ---------------------------------------------------------------------------

def get_download_url(uid: str, token: str, max_429_retries: int = 6) -> str:
    """
    Request a temporary download URL from the Sketchfab API.

    Handles transient 429 "Too Many Requests" responses with exponential
    backoff (honouring the ``Retry-After`` header when Sketchfab sends one).

    Returns the URL of the GLTF archive.  Raises RuntimeError on failure.
    """
    backoff = 4.0  # seconds; doubles on each successive 429
    for attempt in range(max_429_retries + 1):
        resp = requests.get(
            f"{SKETCHFAB_API}/models/{uid}/download",
            headers={"Authorization": f"Token {token}"},
            timeout=30,
        )
        if resp.status_code == 401:
            log.error("Sketchfab API: 401 Unauthorized.  Verify your --api_token.")
            sys.exit(1)
        if resp.status_code == 403:
            raise RuntimeError(
                f"Model {uid}: download not permitted (403 Forbidden). "
                "The model may require a Sketchfab PRO account or be non-downloadable."
            )
        if resp.status_code == 404:
            raise RuntimeError(f"Model {uid}: not found (404).")
        if resp.status_code == 429 and attempt < max_429_retries:
            retry_after_hdr = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after_hdr) if retry_after_hdr else backoff
            except ValueError:
                wait = backoff
            log.warning(
                "  429 rate-limited; sleeping %.1fs before retry (%d/%d)",
                wait, attempt + 1, max_429_retries,
            )
            time.sleep(wait)
            backoff *= 2
            continue
        resp.raise_for_status()

        data = resp.json()
        # Prefer explicit GLTF format; fall back to the original source archive
        for key in ("gltf", "source"):
            if key in data and data[key].get("url"):
                return data[key]["url"]
        raise RuntimeError(f"Model {uid}: no downloadable archive in API response: {data}")

    raise RuntimeError(f"Model {uid}: exhausted retries on 429 Too Many Requests.")


# ---------------------------------------------------------------------------
# Download & extraction
# ---------------------------------------------------------------------------

def download_zip(url: str) -> io.BytesIO:
    """Stream-download the ZIP archive and return it as an in-memory buffer."""
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=65536):
        if chunk:
            buf.write(chunk)
    buf.seek(0)
    return buf


def extract_zip(buf: io.BytesIO, dest_dir: Path) -> None:
    """
    Extract the ZIP into dest_dir.

    If the archive has a single top-level directory, its contents are moved
    one level up so that scene.gltf ends up directly in dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buf) as zf:
        members = zf.namelist()
        # Detect a single top-level directory wrapper
        top_dirs = {m.split("/")[0] for m in members if m.split("/")[0]}
        if len(top_dirs) == 1:
            (top,) = top_dirs
            # Extract to a temp subdirectory, then move contents up
            tmp_dir = dest_dir / f"_tmp_{top}"
            zf.extractall(dest_dir)
            unwrap_dir = dest_dir / top
            if unwrap_dir.is_dir() and unwrap_dir != dest_dir:
                for child in unwrap_dir.iterdir():
                    shutil.move(str(child), dest_dir / child.name)
                unwrap_dir.rmdir()
        else:
            zf.extractall(dest_dir)


# ---------------------------------------------------------------------------
# GLTF inspection
# ---------------------------------------------------------------------------

def inspect_gltf(gltf_path: Path) -> dict:
    """
    Parse the GLTF JSON and return mesh statistics:
        vertex_count  – total vertices across all unique POSITION accessors
        has_normal_map – True if any material declares a normalTexture
    """
    data = json.loads(gltf_path.read_text(encoding="utf-8"))
    accessors = data.get("accessors", [])
    meshes = data.get("meshes", [])
    materials = data.get("materials", [])

    # Collect unique POSITION accessor indices across all mesh primitives
    pos_indices: set[int] = set()
    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            idx = prim.get("attributes", {}).get("POSITION")
            if idx is not None:
                pos_indices.add(idx)

    vertex_count = sum(
        accessors[i]["count"]
        for i in pos_indices
        if i < len(accessors)
    )
    has_normal_map = any("normalTexture" in mat for mat in materials)

    return {"vertex_count": vertex_count, "has_normal_map": has_normal_map}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download GLTF assets from Sketchfab and report mesh info.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input", required=True, type=Path,
        help="Path to the text file listing Sketchfab URLs (e.g. assets/assets.txt)",
    )
    ap.add_argument(
        "--output_dir", required=True, type=Path,
        help="Directory to download assets into",
    )
    ap.add_argument(
        "--api_token", required=True,
        help="Sketchfab API token (generate at sketchfab.com/settings#api)",
    )
    ap.add_argument(
        "--report", type=Path, default=None,
        help="Path to write the CSV report of successful downloads. "
             "Defaults to <input_dir>/downloaded.csv",
    )
    ap.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip models whose output folder already contains a scene.gltf",
    )
    ap.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between API requests (be polite to the API)",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be downloaded without actually downloading anything",
    )
    ap.add_argument(
        "--status_manifest", type=Path, default=None,
        help="Optional path to a JSON status manifest (see mesh_status.py). "
             "Each download outcome is merged into this file so the UI can display it.",
    )
    args = ap.parse_args()

    if args.report is None:
        args.report = args.input.parent / "downloaded.csv"

    entries = parse_input_file(args.input)
    if not entries:
        log.error("No valid Sketchfab URLs found in %s", args.input)
        sys.exit(1)

    log.info("Found %d model(s) to process.", len(entries))

    if args.dry_run:
        log.info("--- DRY RUN ---")
        for e in entries:
            log.info("  %-40s  uid=%s  url=%s", e["folder_name"], e["uid"], e["url"])
        return

    report_rows: list[dict] = []

    for i, entry in enumerate(entries):
        folder_name = entry["folder_name"]
        uid = entry["uid"]
        url = entry["url"]
        dest = args.output_dir / folder_name

        log.info("[%d/%d] %s  (uid=%s)", i + 1, len(entries), folder_name, uid)

        # Skip if already downloaded
        if args.skip_existing and (dest / "scene.gltf").exists():
            log.info("  Already exists, skipping.")
            gltf_info = inspect_gltf(dest / "scene.gltf")
            report_rows.append({
                "folder": folder_name,
                "uid": uid,
                "url": url,
                "status": "skipped_existing",
                "vertex_count": gltf_info["vertex_count"],
                "has_normal_map": gltf_info["has_normal_map"],
            })
            if args.status_manifest is not None:
                update_download(
                    args.status_manifest, folder_name,
                    url=url, uid=uid, status="skipped",
                    vertex_count=gltf_info["vertex_count"],
                    has_normal_map=gltf_info["has_normal_map"],
                )
            continue

        try:
            # 1. Get temporary download URL from Sketchfab API
            log.info("  Requesting download URL …")
            dl_url = get_download_url(uid, args.api_token)

            # 2. Download the ZIP archive
            log.info("  Downloading archive …")
            buf = download_zip(dl_url)

            # 3. Extract into output folder
            log.info("  Extracting to %s …", dest)
            extract_zip(buf, dest)

            # 4. Find and parse the GLTF
            gltf_path = dest / "scene.gltf"
            if not gltf_path.exists():
                # Some archives name the file differently; find first .gltf
                found = list(dest.rglob("*.gltf"))
                if found:
                    gltf_path = found[0]
                    # Rename to scene.gltf for consistency
                    target = dest / "scene.gltf"
                    gltf_path.rename(target)
                    gltf_path = target

            if not gltf_path.exists():
                raise RuntimeError(f"No .gltf file found in extracted archive at {dest}")

            gltf_info = inspect_gltf(gltf_path)
            log.info(
                "  OK  vertex_count=%d  has_normal_map=%s",
                gltf_info["vertex_count"],
                gltf_info["has_normal_map"],
            )
            report_rows.append({
                "folder": folder_name,
                "uid": uid,
                "url": url,
                "status": "success",
                "vertex_count": gltf_info["vertex_count"],
                "has_normal_map": gltf_info["has_normal_map"],
            })
            if args.status_manifest is not None:
                update_download(
                    args.status_manifest, folder_name,
                    url=url, uid=uid, status="success",
                    vertex_count=gltf_info["vertex_count"],
                    has_normal_map=gltf_info["has_normal_map"],
                )

        except Exception as exc:
            log.warning("  FAILED: %s", exc)
            report_rows.append({
                "folder": folder_name,
                "uid": uid,
                "url": url,
                "status": f"error: {exc}",
                "vertex_count": "",
                "has_normal_map": "",
            })
            if args.status_manifest is not None:
                update_download(
                    args.status_manifest, folder_name,
                    url=url, uid=uid, status="error", error=str(exc),
                )

        # Polite delay between requests
        if i < len(entries) - 1:
            time.sleep(args.delay)

    # Write report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["folder", "uid", "url", "status", "vertex_count", "has_normal_map"]
    with open(args.report, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    successes = sum(1 for r in report_rows if r["status"] in ("success", "skipped_existing"))
    failures = len(report_rows) - successes
    log.info(
        "\nDone.  %d succeeded, %d failed.  Report written to %s",
        successes, failures, args.report,
    )


if __name__ == "__main__":
    main()
