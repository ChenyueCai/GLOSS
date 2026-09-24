#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser scoreboard that groups scored single views into quality buckets
and lets you reject views (write ``exclude_from_training_indices``) directly
from the bucketed grid.

Consumes the ``view_scores`` written by ``scripts/datagen/steps/score_views.py``
and renders views bucketed into human-readable categories (bad / weak / ok /
good / excellent) per score. The bucket grid is the primary selection
surface: click a card to preview, click its reject toggle (or press ``x``)
to mark it rejected, click a bucket header's "Reject all" to nuke a whole
bucket, then hit Save (or ``s``) to write the dataset's
``exclude_from_training_indices`` back to ``meta.json``. Steps 5/7/8 of the
data-generation pipeline honor that field.

Supports multiple datasets on the same page. Each positional path is a
single-view style directory; view source per path is auto-detected
(``gen_view_masked/view*.png`` first, then ``gen_view_decomposite/view*.basecolor.png``).

Usage
-----
    python scripts/datagen/utils/score_viewer.py \\
        /data/.../cabbage/single_view/civitai2.0 \\
        /data/.../turtle/single_view/civitai2.0 \\
        /data/.../croissant/single_view/civitai2.0 \\
        --port 10014
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps


DEFAULT_VIEW_RE = re.compile(r"^view(\d+)\.png$")
DEFAULT_META_KEY = "exclude_from_training_indices"


def _view_regex(channel: str | None) -> re.Pattern[str]:
    if not channel:
        return DEFAULT_VIEW_RE
    return re.compile(rf"^view(\d+)\.{re.escape(channel)}\.png$")


# (view_subdir, view_channel) auto-detection priority. First source that
# yields at least one matching file wins for a dataset.
_VIEW_SOURCE_CANDIDATES: list[tuple[str, str | None]] = [
    ("gen_view_masked", None),
    ("gen_view_decomposite", "basecolor"),
    ("gen_view_super", "basecolor"),
]


# Bucket definitions per score metric. Boundaries are lower-inclusive;
# "excellent" has no upper bound. Ordered worst → best. Colors are used
# as the bucket swatch in the UI.
BUCKETS: dict[str, list[dict[str, Any]]] = {
    "clip_prompt": [
        {"label": "bad",       "lo": float("-inf"), "hi": 0.15, "color": "#b42318", "desc": "subject likely absent"},
        {"label": "weak",      "lo": 0.15,          "hi": 0.20, "color": "#d97706", "desc": "subject partially wrong"},
        {"label": "ok",        "lo": 0.20,          "hi": 0.25, "color": "#ca8a04", "desc": "right category, weak specifics"},
        {"label": "good",      "lo": 0.25,          "hi": 0.30, "color": "#15803d", "desc": "subject matches"},
        {"label": "excellent", "lo": 0.30,          "hi": float("inf"), "color": "#0c6d62", "desc": "strong match"},
    ],
    "clip_viewpoint": [
        {"label": "bad",       "lo": float("-inf"), "hi": 0.15, "color": "#b42318", "desc": "viewpoint or subject wrong"},
        {"label": "weak",      "lo": 0.15,          "hi": 0.20, "color": "#d97706", "desc": "weak/ambiguous viewpoint"},
        {"label": "ok",        "lo": 0.20,          "hi": 0.25, "color": "#ca8a04", "desc": "viewpoint plausible"},
        {"label": "good",      "lo": 0.25,          "hi": 0.30, "color": "#15803d", "desc": "viewpoint matches"},
        {"label": "excellent", "lo": 0.30,          "hi": float("inf"), "color": "#0c6d62", "desc": "strong viewpoint match"},
    ],
    "aesthetic": [
        {"label": "bad",       "lo": float("-inf"), "hi": 4.5, "color": "#b42318", "desc": "broken / artifacts"},
        {"label": "mediocre",  "lo": 4.5,           "hi": 5.0, "color": "#d97706", "desc": "flat / noisy"},
        {"label": "ok",        "lo": 5.0,           "hi": 5.5, "color": "#ca8a04", "desc": "typical"},
        {"label": "good",      "lo": 5.5,           "hi": 6.5, "color": "#15803d", "desc": "pleasing"},
        {"label": "excellent", "lo": 6.5,           "hi": float("inf"), "color": "#0c6d62", "desc": "professional-looking"},
    ],
    "normal_agreement": [
        {"label": "bad",       "lo": float("-inf"), "hi": 0.25, "color": "#b42318", "desc": "geometry ignored"},
        {"label": "weak",      "lo": 0.25,          "hi": 0.45, "color": "#d97706", "desc": "surface orientations diverge"},
        {"label": "moderate",  "lo": 0.45,          "hi": 0.65, "color": "#ca8a04", "desc": "shape retained, drift"},
        {"label": "strong",    "lo": 0.65,          "hi": 0.80, "color": "#15803d", "desc": "minor drift"},
        {"label": "excellent", "lo": 0.80,          "hi": float("inf"), "color": "#0c6d62", "desc": "geometry respected"},
    ],
}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Score Viewer — multi-dataset</title>
  <style>
    :root {
      --bg: #f3efe4;
      --panel: rgba(255, 252, 245, 0.92);
      --line: #d9cdb5;
      --ink: #1f1a14;
      --muted: #6d6254;
      --accent: #0c6d62;
      --shadow: 0 18px 48px rgba(77, 55, 28, 0.12);
      --radius: 18px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #f8f4ea 0%, #efe7d7 100%);
    }
    button, input, select { font: inherit; color: inherit; }

    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
      gap: 18px;
      padding: 18px;
    }
    .main, .sidebar {
      background: var(--panel);
      border: 1px solid rgba(122, 104, 78, 0.18);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }
    .main { padding: 18px; min-width: 0; }
    .sidebar {
      position: sticky; top: 18px; align-self: start; padding: 18px;
      display: grid; gap: 14px;
      max-height: calc(100vh - 36px); overflow: auto;
    }

    .title { margin: 0; font-size: 1.55rem; font-weight: 650; letter-spacing: -0.02em; }
    .subtle { color: var(--muted); font-size: 0.94rem; }

    .toolbar, .legend, .meta-list {
      display: flex; flex-wrap: wrap; gap: 10px 12px; align-items: center;
    }
    .toolbar label, .legend .chip {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 8px 12px; border-radius: 999px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(122, 104, 78, 0.16);
    }
    .toolbar input, .toolbar select {
      border: 0; background: transparent; outline: none; min-width: 52px;
    }

    .bucket {
      margin: 18px 0 8px 0;
      border-radius: var(--radius);
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.55);
      border: 1px solid rgba(122, 104, 78, 0.14);
    }
    .bucket-header {
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px;
      margin-bottom: 8px;
    }
    .swatch {
      width: 16px; height: 16px; border-radius: 4px; display: inline-block;
    }
    .bucket-label { font-weight: 650; font-size: 1.05rem; }
    .bucket-range { color: var(--muted); font-size: 0.88rem; font-variant-numeric: tabular-nums; }
    .bucket-desc { color: var(--muted); font-size: 0.88rem; font-style: italic; }
    .bucket-count { color: var(--muted); font-size: 0.88rem; font-variant-numeric: tabular-nums; }

    .grid {
      display: grid; gap: 10px;
      grid-template-columns: repeat(auto-fill, minmax(var(--thumb-width, 160px), 1fr));
    }

    .card {
      border-radius: 12px; overflow: hidden; position: relative;
      border: 1px solid rgba(122, 104, 78, 0.16);
      background: rgba(255, 255, 255, 0.82);
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }
    .card:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(82, 60, 31, 0.14); }
    .card.selected { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(12, 109, 98, 0.14); }
    .card.rejected { border-color: #b42318; opacity: 0.55; }
    .card.rejected .thumb-wrap::after {
      content: "rejected"; position: absolute; top: 6px; left: 6px;
      background: rgba(180, 35, 24, 0.92); color: #fff; font-size: 0.7rem;
      padding: 2px 6px; border-radius: 4px; letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .reject-toggle {
      position: absolute; top: 6px; right: 6px; z-index: 2;
      background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(122, 104, 78, 0.32);
      color: #5b3a1e; font-size: 0.7rem; padding: 2px 7px; border-radius: 4px;
      cursor: pointer; line-height: 1.4;
    }
    .reject-toggle:hover { background: #fff; border-color: var(--accent); }
    .card.rejected .reject-toggle { background: #b42318; color: #fff; border-color: #8a1a12; }
    .card.rejected .reject-toggle:hover { background: #962215; }
    .bucket-actions { margin-left: auto; display: flex; gap: 6px; }
    .btn-bucket {
      font-size: 0.78rem; padding: 3px 9px; border-radius: 6px;
      border: 1px solid rgba(122, 104, 78, 0.3); background: rgba(255, 255, 255, 0.7);
      color: #5b3a1e; cursor: pointer;
    }
    .btn-bucket:hover { background: #fff; border-color: var(--accent); }
    .btn-bucket.danger { color: #b42318; border-color: rgba(180, 35, 24, 0.4); }
    .btn-bucket.danger:hover { background: #fff5f4; }
    .save-bar {
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      padding: 10px 14px; margin-bottom: 10px; border-radius: 12px;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid rgba(122, 104, 78, 0.16);
    }
    .save-bar .counts { font-size: 0.92rem; color: var(--muted); font-variant-numeric: tabular-nums; }
    .save-bar .counts strong { color: #5b3a1e; }
    .save-bar .save-status { margin-left: auto; font-size: 0.85rem; color: var(--muted); }
    .save-bar .save-status.dirty { color: #d97706; font-weight: 600; }
    .save-bar .save-status.error { color: #b42318; font-weight: 600; }
    .btn-save {
      padding: 6px 14px; border-radius: 8px; border: 1px solid var(--accent);
      background: var(--accent); color: #fff; font-weight: 600; cursor: pointer;
    }
    .btn-save:hover { background: #0a5c52; }
    .btn-save:disabled { opacity: 0.5; cursor: default; }

    .thumb-wrap {
      aspect-ratio: 1 / 1;
      background:
        linear-gradient(45deg, rgba(0, 0, 0, 0.03) 25%, transparent 25%, transparent 75%, rgba(0, 0, 0, 0.03) 75%),
        linear-gradient(45deg, rgba(0, 0, 0, 0.03) 25%, transparent 25%, transparent 75%, rgba(0, 0, 0, 0.03) 75%);
      background-size: 22px 22px; background-position: 0 0, 11px 11px;
    }
    .thumb-wrap img { width: 100%; height: 100%; display: block; object-fit: cover; }

    .card-body {
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 8px; gap: 6px; font-size: 0.82rem;
    }
    .card-id { font-weight: 600; }
    .card-score { font-variant-numeric: tabular-nums; color: var(--muted); }

    .hero { border-radius: 20px; overflow: hidden; border: 1px solid rgba(122, 104, 78, 0.16); background: #fff; aspect-ratio: 1 / 1; }
    .hero img { width: 100%; height: 100%; object-fit: cover; display: block; }

    .score-table {
      display: grid; grid-template-columns: auto 1fr; gap: 4px 10px;
      font-variant-numeric: tabular-nums; font-size: 0.92rem;
    }
    .score-table .label { color: var(--muted); }
    .score-table .value { font-weight: 600; }
    .score-row { display: contents; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; vertical-align: middle; }

    .help {
      padding: 12px 14px; border-radius: 18px;
      background: rgba(255, 255, 255, 0.66);
      border: 1px solid rgba(122, 104, 78, 0.14);
      font-size: 0.9rem; color: var(--muted);
    }
    .prompt-text {
      font-size: 0.88rem; color: var(--ink);
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(122, 104, 78, 0.12);
      border-radius: 12px; padding: 10px 12px; line-height: 1.4;
      max-height: 160px; overflow: auto;
    }

    @media (max-width: 1080px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar { position: static; max-height: none; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <main class="main">
      <h1 class="title">Score Viewer</h1>
      <div class="subtle" id="header"></div>

      <div class="toolbar" style="margin-top:12px">
        <label>Score
          <select id="scoreSelect"></select>
        </label>
        <label>Sort
          <select id="sortSelect">
            <option value="score-asc" selected>score ascending (worst first)</option>
            <option value="score-desc">score descending (best first)</option>
            <option value="id-asc">view id ascending</option>
          </select>
        </label>
        <label>Thumb
          <select id="thumbSize">
            <option value="120">120 px</option>
            <option value="160" selected>160 px</option>
            <option value="200">200 px</option>
            <option value="240">240 px</option>
          </select>
        </label>
        <label>Max per bucket
          <input id="maxPerBucket" type="number" value="12" min="0" step="6" title="0 = show all">
        </label>
        <button type="button" id="refreshBtn" class="btn-bucket"
                title="Re-scan for newly-completed datasets and reload existing meta.json">
          Refresh
        </button>
        <span class="subtle" id="refreshStatus" style="font-variant-numeric:tabular-nums"></span>
      </div>

      <div class="toolbar" style="margin-top:8px">
        <span class="subtle" style="margin-right:4px">Datasets:</span>
        <span id="datasetFilter"></span>
      </div>

      <div class="legend" id="legend" style="margin-top:10px"></div>

      <div id="datasets"></div>
    </main>

    <aside class="sidebar">
      <div class="hero"><img id="heroImage" alt="selected preview"></div>
      <div>
        <h2 id="heroTitle" style="margin:0">No view selected</h2>
        <div class="subtle" id="heroPath"></div>
      </div>
      <div class="score-table" id="scoreTable"></div>
      <div id="promptBox"></div>
      <div class="help">
        Click a thumbnail to preview here. Use the <b>Score</b> dropdown to
        regroup; views are bucketed by the ranges in the legend. Toggle
        individual datasets via the chips. <b>Reject / Restore</b> on each
        card or the bucket header marks views; <b>Save</b> writes
        <code>exclude_from_training_indices</code> to that dataset's
        <code>meta.json</code> (steps 5/7/8 will skip rejected ids).
        Keyboard: <b>x</b> or <b>space</b> toggles reject on the selected
        view; <b>s</b> saves its dataset.
      </div>
    </aside>
  </div>

  <script>
    const state = {
      datasets: [],
      buckets: {},
      scoreOrder: [],
      previewSize: 960,
    };
    // UI state:
    let selectedKey = null;  // "<ds_idx>:<view_id>"
    const enabledDatasets = new Set();  // integer indices
    const rejected = new Map();          // dsIdx -> Set<viewId>
    const dirtyDatasets = new Set();     // dsIdx with unsaved changes
    const saveStatus = new Map();        // dsIdx -> {ok|error|saving, msg}

    function rejSet(dsIdx) {
      let s = rejected.get(dsIdx);
      if (!s) { s = new Set(); rejected.set(dsIdx, s); }
      return s;
    }
    function isRejected(dsIdx, id) { return rejSet(dsIdx).has(id); }
    function setRejected(dsIdx, id, on) {
      const s = rejSet(dsIdx);
      const before = s.size;
      if (on) s.add(id); else s.delete(id);
      if (s.size !== before) {
        dirtyDatasets.add(dsIdx);
        // mark single card without rebuilding the whole grid
        document.querySelectorAll(`.card[data-key="${keyOf(dsIdx, id)}"]`).forEach(c => {
          c.classList.toggle("rejected", on);
          const btn = c.querySelector(".reject-toggle");
          if (btn) btn.textContent = on ? "Restore" : "Reject";
        });
        updateSaveBar(dsIdx);
      }
    }
    async function saveDataset(dsIdx) {
      saveStatus.set(dsIdx, {kind: "saving", msg: "saving…"});
      updateSaveBar(dsIdx);
      try {
        const ids = Array.from(rejSet(dsIdx)).sort((a, b) => a - b);
        const r = await fetch(`/api/save/${dsIdx}`, {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({rejected: ids}),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
        const out = await r.json();
        dirtyDatasets.delete(dsIdx);
        // Trust server's normalized list (drops invalid ids).
        rejected.set(dsIdx, new Set(out.rejectedIds || []));
        saveStatus.set(dsIdx, {kind: "ok", msg: `saved ${out.count}/${state.datasets[dsIdx].viewIds.length}`});
      } catch (e) {
        saveStatus.set(dsIdx, {kind: "error", msg: String(e)});
      }
      updateSaveBar(dsIdx);
    }
    function updateSaveBar(dsIdx) {
      const bar = document.querySelector(`.save-bar[data-ds-idx="${dsIdx}"]`);
      if (!bar) return;
      const counts = bar.querySelector(".counts");
      const status = bar.querySelector(".save-status");
      const btn = bar.querySelector(".btn-save");
      const total = state.datasets[dsIdx].viewIds.length;
      const rejectedN = rejSet(dsIdx).size;
      const keptN = total - rejectedN;
      counts.innerHTML = `<strong>${total}</strong> views · <strong>${keptN}</strong> kept · <strong>${rejectedN}</strong> rejected`;
      const dirty = dirtyDatasets.has(dsIdx);
      const s = saveStatus.get(dsIdx);
      let text = "";
      let cls = "";
      if (s) {
        text = s.msg;
        if (s.kind === "error") cls = "error";
        else if (s.kind === "saving") cls = "dirty";
      }
      if (dirty) { text = "unsaved changes"; cls = "dirty"; }
      status.textContent = text;
      status.className = `save-status ${cls}`;
      btn.disabled = !dirty;
    }

    const els = {
      header: document.getElementById("header"),
      scoreSelect: document.getElementById("scoreSelect"),
      sortSelect: document.getElementById("sortSelect"),
      refreshBtn: document.getElementById("refreshBtn"),
      refreshStatus: document.getElementById("refreshStatus"),
      thumbSize: document.getElementById("thumbSize"),
      maxPerBucket: document.getElementById("maxPerBucket"),
      datasetFilter: document.getElementById("datasetFilter"),
      legend: document.getElementById("legend"),
      datasets: document.getElementById("datasets"),
      heroImage: document.getElementById("heroImage"),
      heroTitle: document.getElementById("heroTitle"),
      heroPath: document.getElementById("heroPath"),
      scoreTable: document.getElementById("scoreTable"),
      promptBox: document.getElementById("promptBox"),
    };

    const formatViewId = (id) => `view${String(id).padStart(4, "0")}`;
    const fmt = (v) => (v === null || v === undefined || Number.isNaN(v)) ? "—"
                     : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));
    const keyOf = (dsIdx, id) => `${dsIdx}:${id}`;

    function bucketOf(metric, value) {
      if (value === null || value === undefined || Number.isNaN(value)) return null;
      const defs = state.buckets[metric] || [];
      for (const b of defs) {
        if (value >= b.lo && value < b.hi) return b;
      }
      if (defs.length && value >= defs[defs.length - 1].lo) return defs[defs.length - 1];
      return null;
    }

    function colorForMetric(metric, value) {
      const b = bucketOf(metric, value);
      return b ? b.color : "#999";
    }

    function escapeHtml(s) {
      return s.replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }

    function buildLegend() {
      const metric = els.scoreSelect.value;
      const defs = state.buckets[metric] || [];
      const parts = defs.map(b => {
        const hi = isFinite(b.hi) && Math.abs(b.hi) < 1e17 ? fmt(b.hi) : "∞";
        const lo = isFinite(b.lo) && Math.abs(b.lo) < 1e17 ? fmt(b.lo) : "−∞";
        return `<span class="chip"><span class="swatch" style="background:${b.color}"></span>${b.label} [${lo}, ${hi})</span>`;
      });
      els.legend.innerHTML = parts.join("");
    }

    function buildDatasetFilter() {
      const parts = state.datasets.map(ds => {
        const checked = enabledDatasets.has(ds.idx) ? "checked" : "";
        return `
          <label class="chip" style="cursor:pointer">
            <input type="checkbox" data-ds-idx="${ds.idx}" ${checked} style="margin-right:4px">
            ${escapeHtml(ds.name)}
            <span class="subtle" style="margin-left:6px">${ds.viewIds.length}</span>
          </label>
        `;
      }).join("");
      els.datasetFilter.innerHTML = parts;
      for (const cb of els.datasetFilter.querySelectorAll("input[type=checkbox]")) {
        cb.addEventListener("change", (ev) => {
          const idx = Number(ev.target.dataset.dsIdx);
          if (ev.target.checked) enabledDatasets.add(idx);
          else enabledDatasets.delete(idx);
          renderAll();
        });
      }
    }

    function datasetSummary(ds, metric) {
      const vals = ds.viewIds.map(id => ds.scores[id]?.[metric]).filter(v => typeof v === "number");
      if (!vals.length) return "no data for this metric";
      vals.sort((a, b) => a - b);
      const med = vals[Math.floor(vals.length / 2)];
      const p10 = vals[Math.floor(vals.length * 0.1)];
      const p90 = vals[Math.floor(vals.length * 0.9)];
      return `n=${vals.length}, min=${fmt(vals[0])}, p10=${fmt(p10)}, med=${fmt(med)}, p90=${fmt(p90)}, max=${fmt(vals[vals.length - 1])}`;
    }

    function renderDatasets() {
      const metric = els.scoreSelect.value;
      const defs = state.buckets[metric] || [];
      const sortMode = els.sortSelect.value;
      const maxPer = Math.max(0, Number(els.maxPerBucket.value) || 0);
      const thumb = Number(els.thumbSize.value);
      document.documentElement.style.setProperty("--thumb-width", `${thumb}px`);

      const fragment = document.createDocumentFragment();

      for (const ds of state.datasets) {
        if (!enabledDatasets.has(ds.idx)) continue;

        const dsWrap = document.createElement("section");
        dsWrap.className = "dataset";
        dsWrap.style.margin = "18px 0";
        dsWrap.style.padding = "14px";
        dsWrap.style.background = "rgba(255, 255, 255, 0.45)";
        dsWrap.style.borderRadius = "18px";
        dsWrap.style.border = "1px solid rgba(122, 104, 78, 0.14)";

        const header = document.createElement("div");
        header.innerHTML = `
          <div style="display:flex; flex-wrap:wrap; align-items:baseline; gap:10px 14px">
            <div style="font-size:1.15rem; font-weight:650">${escapeHtml(ds.name)}</div>
            <div class="subtle">${escapeHtml(ds.viewSource)} · ${ds.viewIds.length} views</div>
            <div class="subtle">${escapeHtml(ds.datasetPath)}</div>
          </div>
          <div class="subtle" style="font-variant-numeric:tabular-nums; margin-top:4px">
            ${metric}: ${datasetSummary(ds, metric)}
          </div>
        `;
        dsWrap.appendChild(header);

        // Save bar (per dataset). Save button enabled when dirty.
        const saveBar = document.createElement("div");
        saveBar.className = "save-bar";
        saveBar.dataset.dsIdx = ds.idx;
        saveBar.innerHTML = `
          <span class="counts"></span>
          <button type="button" class="btn-save" disabled>Save</button>
          <span class="save-status"></span>
        `;
        saveBar.querySelector(".btn-save").addEventListener("click", () => saveDataset(ds.idx));
        dsWrap.appendChild(saveBar);

        // Bucket each view within THIS dataset
        const grouped = new Map(defs.map(b => [b.label, []]));
        const unscored = [];
        for (const id of ds.viewIds) {
          const v = ds.scores[id]?.[metric];
          const b = bucketOf(metric, v);
          if (b) grouped.get(b.label).push({ id, value: v });
          else unscored.push({ id, value: null });
        }

        function sortArr(arr) {
          if (sortMode === "score-asc") arr.sort((a, b) => (a.value ?? Infinity) - (b.value ?? Infinity));
          else if (sortMode === "score-desc") arr.sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
          else arr.sort((a, b) => a.id - b.id);
        }

        for (const b of defs) {
          const items = grouped.get(b.label);
          sortArr(items);
          const shown = maxPer > 0 ? items.slice(0, maxPer) : items;
          if (!items.length && !shown.length) continue;

          const section = document.createElement("section");
          section.className = "bucket";
          const hiStr = isFinite(b.hi) && Math.abs(b.hi) < 1e17 ? fmt(b.hi) : "∞";
          const loStr = isFinite(b.lo) && Math.abs(b.lo) < 1e17 ? fmt(b.lo) : "−∞";

          const bh = document.createElement("div");
          bh.className = "bucket-header";
          bh.innerHTML = `
            <span class="swatch" style="background:${b.color}"></span>
            <span class="bucket-label">${b.label}</span>
            <span class="bucket-range">[${loStr}, ${hiStr})</span>
            <span class="bucket-desc">— ${b.desc}</span>
            <span class="bucket-count">${items.length} views${shown.length < items.length ? ` (showing ${shown.length})` : ""}</span>
            <span class="bucket-actions">
              <button type="button" class="btn-bucket danger" data-act="reject-all">Reject all</button>
              <button type="button" class="btn-bucket"        data-act="restore-all">Restore all</button>
            </span>
          `;
          const allIds = items.map(it => it.id);
          bh.querySelector('[data-act="reject-all"]').addEventListener("click", () => {
            for (const id of allIds) setRejected(ds.idx, id, true);
          });
          bh.querySelector('[data-act="restore-all"]').addEventListener("click", () => {
            for (const id of allIds) setRejected(ds.idx, id, false);
          });
          section.appendChild(bh);

          if (shown.length) {
            const grid = document.createElement("div");
            grid.className = "grid";
            for (const item of shown) grid.appendChild(makeCard(ds, item.id, item.value, b.color));
            section.appendChild(grid);
          }
          dsWrap.appendChild(section);
        }

        if (unscored.length) {
          const section = document.createElement("section");
          section.className = "bucket";
          section.innerHTML = `
            <div class="bucket-header">
              <span class="swatch" style="background:#999"></span>
              <span class="bucket-label">unscored</span>
              <span class="bucket-desc">— no ${metric} value in meta.json</span>
              <span class="bucket-count">${unscored.length} views</span>
            </div>
          `;
          const grid = document.createElement("div");
          grid.className = "grid";
          const shown = maxPer > 0 ? unscored.slice(0, maxPer) : unscored;
          for (const item of shown) grid.appendChild(makeCard(ds, item.id, null, "#999"));
          section.appendChild(grid);
          dsWrap.appendChild(section);
        }

        fragment.appendChild(dsWrap);
      }

      els.datasets.replaceChildren(fragment);
    }

    function makeCard(ds, id, value, borderColor) {
      const card = document.createElement("article");
      card.className = "card";
      const key = keyOf(ds.idx, id);
      card.dataset.key = key;
      if (key === selectedKey) card.classList.add("selected");
      const rejNow = isRejected(ds.idx, id);
      if (rejNow) card.classList.add("rejected");
      card.style.borderLeft = `4px solid ${borderColor}`;
      const thumb = Number(els.thumbSize.value);
      card.innerHTML = `
        <div class="thumb-wrap">
          <button type="button" class="reject-toggle">${rejNow ? "Restore" : "Reject"}</button>
          <img loading="lazy" alt="${formatViewId(id)}" src="/api/image/${ds.idx}/${id}?size=${thumb}&v=${encodeURIComponent(ds.cacheToken)}">
        </div>
        <div class="card-body">
          <span class="card-id">${formatViewId(id)}</span>
          <span class="card-score">${fmt(value)}</span>
        </div>
      `;
      card.addEventListener("click", () => selectView(ds.idx, id));
      const btn = card.querySelector(".reject-toggle");
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        setRejected(ds.idx, id, !isRejected(ds.idx, id));
      });
      return card;
    }

    function selectView(dsIdx, id) {
      selectedKey = keyOf(dsIdx, id);
      const ds = state.datasets[dsIdx];
      els.heroImage.src = `/api/image/${dsIdx}/${id}?size=${state.previewSize}&v=${encodeURIComponent(ds.cacheToken)}`;
      els.heroTitle.textContent = `${ds.name} · ${formatViewId(id)}`;
      els.heroPath.textContent = `dataset #${dsIdx} · view ${id}`;

      const scores = ds.scores[id] || {};
      const rows = state.scoreOrder.map(k => {
        const v = scores[k];
        const color = colorForMetric(k, v);
        const b = bucketOf(k, v);
        return `
          <div class="score-row">
            <div class="label"><span class="dot" style="background:${color}"></span>${k}</div>
            <div class="value">${fmt(v)} ${b ? `<span class="subtle">(${b.label})</span>` : ""}</div>
          </div>
        `;
      }).join("");
      els.scoreTable.innerHTML = rows;

      const prompt = ds.prompts[id];
      els.promptBox.innerHTML = prompt ? `<div class="prompt-text">${escapeHtml(prompt)}</div>` : "";

      // Highlight only the newly-selected card.
      document.querySelectorAll(".card.selected").forEach(c => c.classList.remove("selected"));
      document.querySelectorAll(`.card[data-key="${selectedKey}"]`).forEach(c => c.classList.add("selected"));
    }

    function renderAll() {
      buildLegend();
      renderDatasets();
      // The grid is rebuilt; re-sync per-dataset save bars from current state.
      for (const ds of state.datasets) {
        if (rejected.has(ds.idx)) updateSaveBar(ds.idx);
      }
      if (selectedKey) {
        const [dsIdx, id] = selectedKey.split(":").map(Number);
        if (enabledDatasets.has(dsIdx)) selectView(dsIdx, id);
      }
    }

    function attachEvents() {
      for (const el of [els.scoreSelect, els.sortSelect, els.thumbSize, els.maxPerBucket]) {
        el.addEventListener("change", renderAll);
      }
      if (els.refreshBtn) els.refreshBtn.addEventListener("click", refreshDatasets);
    }

    async function refreshDatasets() {
      if (dirtyDatasets.size > 0) {
        if (!confirm(`${dirtyDatasets.size} dataset(s) have unsaved changes. Refresh anyway and discard them?`)) return;
      }
      els.refreshStatus.textContent = "scanning…";
      els.refreshBtn.disabled = true;
      try {
        const r = await fetch("/api/refresh", {method: "POST"});
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
        const out = await r.json();
        // Re-fetch full state and rebuild UI from scratch (cheap).
        const sresp = await fetch("/api/state");
        const payload = await sresp.json();
        // Reset client state.
        state.datasets = payload.datasets;
        state.buckets = payload.buckets;
        state.scoreOrder = payload.scoreOrder;
        state.previewSize = payload.previewSize;
        state.thumbSize = payload.thumbSize;
        rejected.clear(); dirtyDatasets.clear(); saveStatus.clear();
        for (const ds of state.datasets) {
          rejected.set(ds.idx, new Set(ds.rejectedIds || []));
        }
        // Rebuild dataset filter chips (set of datasets may have changed).
        enabledDatasets.clear();
        for (const ds of state.datasets) enabledDatasets.add(ds.idx);
        const totalViews = state.datasets.reduce((acc, ds) => acc + ds.viewIds.length, 0);
        els.header.textContent =
          `${state.datasets.length} dataset${state.datasets.length === 1 ? "" : "s"} · ${totalViews} views total`;
        buildDatasetFilter();
        renderAll();
        for (const ds of state.datasets) updateSaveBar(ds.idx);

        const parts = [];
        if (out.added.length)   parts.push(`+${out.added.length} added (${out.added.slice(0, 3).join(", ")}${out.added.length > 3 ? "…" : ""})`);
        if (out.removed.length) parts.push(`-${out.removed.length} removed`);
        if (!parts.length)      parts.push(`no change · ${out.kept.length} kept`);
        const ts = new Date().toLocaleTimeString();
        els.refreshStatus.textContent = `${parts.join(" · ")} · ${ts}`;
      } catch (e) {
        els.refreshStatus.textContent = `error: ${e.message || e}`;
      } finally {
        els.refreshBtn.disabled = false;
      }
    }

    async function boot() {
      const response = await fetch("/api/state");
      const payload = await response.json();
      Object.assign(state, payload);

      const totalViews = state.datasets.reduce((acc, ds) => acc + ds.viewIds.length, 0);
      els.header.textContent =
        `${state.datasets.length} dataset${state.datasets.length === 1 ? "" : "s"} · ${totalViews} views total`;

      // Score dropdown
      for (const k of state.scoreOrder) {
        const opt = document.createElement("option");
        opt.value = k; opt.textContent = k;
        els.scoreSelect.appendChild(opt);
      }
      const firstWithData = state.scoreOrder.find(k =>
        state.datasets.some(ds => ds.viewIds.some(id => typeof ds.scores[id]?.[k] === "number"))
      );
      if (firstWithData) els.scoreSelect.value = firstWithData;

      // Default: all datasets enabled
      for (const ds of state.datasets) enabledDatasets.add(ds.idx);

      // Initialize rejected sets from each dataset's exclude_from_training_indices
      for (const ds of state.datasets) {
        rejected.set(ds.idx, new Set(ds.rejectedIds || []));
      }

      attachEvents();
      buildDatasetFilter();
      renderAll();
      // Sync save bars after the grid first renders.
      for (const ds of state.datasets) updateSaveBar(ds.idx);

      // Keyboard: 'x' or space toggle reject on selected; 's' saves its dataset.
      document.addEventListener("keydown", (ev) => {
        const tag = (ev.target && ev.target.tagName || "").toUpperCase();
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (!selectedKey) return;
        const [dsIdxStr, idStr] = selectedKey.split(":");
        const dsIdx = Number(dsIdxStr); const id = Number(idStr);
        if (ev.key === "x" || ev.key === " ") {
          ev.preventDefault();
          setRejected(dsIdx, id, !isRejected(dsIdx, id));
        } else if (ev.key === "s") {
          ev.preventDefault();
          if (dirtyDatasets.has(dsIdx)) saveDataset(dsIdx);
        }
      });

      // Warn on close if anything is unsaved.
      window.addEventListener("beforeunload", (ev) => {
        if (dirtyDatasets.size > 0) { ev.preventDefault(); ev.returnValue = ""; }
      });

      // Auto-select the first view of the first dataset that has one
      const first = state.datasets.find(ds => ds.viewIds.length);
      if (first) selectView(first.idx, first.viewIds[0]);
    }

    boot().catch(e => {
      els.datasets.innerHTML = `<div class="help" style="color:#b42318">${escapeHtml(e.message || String(e))}</div>`;
    });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server app
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewerConfig:
    thumb_size: int
    preview_size: int


class Dataset:
    """A single mesh's scored views, identified by its single_view style dir."""

    def __init__(self, single_view_dir: Path, meta_subdir: str, meta_filename: str,
                 view_subdir: str | None = None, view_channel: str | None = None):
        self.single_view_dir = single_view_dir
        self.meta_path = single_view_dir / meta_filename
        self.meta_subdir_path = single_view_dir / meta_subdir

        # Resolve view source (view_dir, channel). If an explicit view_subdir
        # is given, use that. Otherwise probe the candidate list.
        if view_subdir is not None:
            vd = single_view_dir / view_subdir
            if not vd.is_dir():
                raise FileNotFoundError(f"View directory not found: {vd}")
            self.view_dir = vd
            self.view_channel = view_channel
        else:
            chosen: tuple[Path, str | None] | None = None
            for sub, chan in _VIEW_SOURCE_CANDIDATES:
                vd = single_view_dir / sub
                if not vd.is_dir():
                    continue
                regex = _view_regex(chan)
                if any(regex.match(p.name) for p in vd.iterdir() if p.is_file()):
                    chosen = (vd, chan)
                    break
            if chosen is None:
                tried = [
                    f"{s}/view*.{c}.png" if c else f"{s}/view*.png"
                    for s, c in _VIEW_SOURCE_CANDIDATES
                ]
                raise FileNotFoundError(
                    f"No view source found under {single_view_dir}; tried {tried}"
                )
            self.view_dir, self.view_channel = chosen

        self.view_paths = self._discover_views()
        self.meta = self._load_meta()
        self._prompts_cache: dict[int, str] | None = None

    @property
    def name(self) -> str:
        """Display label.

        Path layout: ``<mesh_name>/single_view/<sv_subdir>``. We surface
        ``<mesh_name>/<sv_subdir>`` so e.g. ``barrel_2/origin`` and
        ``barrel_2/anchors`` are distinguishable in the UI. Falls back to
        the directory name if the path doesn't match this layout."""
        try:
            mesh = self.single_view_dir.parent.parent.name
            sv = self.single_view_dir.name
            return f"{mesh}/{sv}"
        except Exception:
            return self.single_view_dir.name

    @property
    def prompts(self) -> dict[int, str]:
        if self._prompts_cache is None:
            self._prompts_cache = self._load_prompts()
        return self._prompts_cache

    def _discover_views(self) -> dict[int, Path]:
        regex = _view_regex(self.view_channel)
        paths: dict[int, Path] = {}
        for p in sorted(self.view_dir.iterdir()):
            m = regex.match(p.name)
            if not m:
                continue
            paths[int(m.group(1))] = p
        if not paths:
            raise FileNotFoundError(
                f"No matching view files under {self.view_dir} "
                f"(pattern={regex.pattern})"
            )
        return paths

    def _load_meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        with self.meta_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Expected object in {self.meta_path}")
        return data

    def _load_prompts(self) -> dict[int, str]:
        """Load per-view prompts via plain yaml.safe_load (much faster than
        OmegaConf for a few thousand small files). Absence is fine — the UI
        just won't show a prompt line."""
        out: dict[int, str] = {}
        if not self.meta_subdir_path.is_dir():
            return out
        try:
            import yaml
            try:
                from yaml import CSafeLoader as SafeLoader  # libyaml-backed, fastest
            except ImportError:
                from yaml import SafeLoader  # type: ignore
        except Exception:
            return out
        for vid in self.view_paths:
            yml = self.meta_subdir_path / f"view{vid:04d}.yml"
            if not yml.exists():
                yml = self.meta_subdir_path / f"view{vid:04d}.yaml"
                if not yml.exists():
                    continue
            try:
                with yml.open("r", encoding="utf-8") as fh:
                    cfg = yaml.load(fh, Loader=SafeLoader)
            except Exception:
                continue
            if isinstance(cfg, dict):
                p = cfg.get("prompt", None)
                if p:
                    out[vid] = str(p)
        return out

    def as_state(self) -> dict[str, Any]:
        scores_in = self.meta.get("view_scores", {}) or {}
        scores_out: dict[int, dict[str, float]] = {}
        for vid_str, rec in scores_in.items():
            try:
                vid = int(vid_str)
            except ValueError:
                continue
            if vid not in self.view_paths:
                continue
            if not isinstance(rec, dict):
                continue
            scores_out[vid] = {k: float(v) for k, v in rec.items() if isinstance(v, (int, float))}

        cache_token = hashlib.sha1(
            f"{self.view_dir}|{self.meta_path}|{len(self.view_paths)}".encode("utf-8")
        ).hexdigest()[:12]

        view_source = self.view_dir.name
        if self.view_channel:
            view_source += f" (channel={self.view_channel})"

        rejected_in = self.meta.get(DEFAULT_META_KEY) or []
        rejected_ids = sorted({int(i) for i in rejected_in
                               if isinstance(i, (int, str)) and str(i).lstrip("-").isdigit()})

        return {
            "name": self.name,
            "datasetPath": str(self.view_dir),
            "metaPath": str(self.meta_path),
            "viewSource": view_source,
            "viewIds": sorted(self.view_paths),
            "scores": {str(k): v for k, v in scores_out.items()},
            "prompts": {str(k): v for k, v in self.prompts.items()},
            "rejectedIds": rejected_ids,
            "cacheToken": cache_token,
        }

    def save_rejected(self, ids: list[int]) -> dict[str, Any]:
        """Atomically write the new ``exclude_from_training_indices`` list to
        meta.json. Preserves all other fields (including ``view_scores``,
        ``auto_excluded_indices``, etc.). Updates ``self.meta`` in-memory.

        Returns a small status dict for the HTTP response."""
        clean = sorted({int(i) for i in ids
                        if isinstance(i, (int, str)) and str(i).lstrip("-").isdigit()
                        and int(i) in self.view_paths})
        # Reload from disk to capture concurrent score-writes, then patch.
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        else:
            meta = {}
        meta[DEFAULT_META_KEY] = clean

        # Atomic replace: write to a temp sibling then rename.
        tmp = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
            fh.write("\n")
        tmp.replace(self.meta_path)
        self.meta = meta
        return {
            "saved": True,
            "count": len(clean),
            "metaPath": str(self.meta_path),
            "rejectedIds": clean,
        }

    def image_bytes(self, view_id: int, size: int) -> bytes:
        if view_id not in self.view_paths:
            raise KeyError(view_id)
        path = self.view_paths[view_id]
        return render_image(path, size=size, mtime_ns=path.stat().st_mtime_ns)


class ViewerApp:
    def __init__(self, config: ViewerConfig, datasets: list[Dataset]):
        self.config = config
        self.datasets = datasets
        # Ensure unique display labels across datasets (disambiguate duplicates
        # by appending the style dir name).
        seen: dict[str, int] = {}
        for ds in datasets:
            if datasets.count(ds) > 1 or [d.name for d in datasets].count(ds.name) > 1:
                seen[ds.name] = seen.get(ds.name, 0) + 1
        # Set by main() so /api/refresh can run discovery again.
        self.collect_paths = lambda: [ds.single_view_dir for ds in self.datasets]
        self.build_dataset = lambda p: None  # populated by main()

    def rescan(self) -> dict[str, Any]:
        """Re-run discovery and reload datasets in place. Existing datasets
        keep their loaded prompt caches; new ones get freshly built. Returns
        a small status dict for the HTTP response."""
        paths = self.collect_paths()
        existing = {ds.single_view_dir: ds for ds in self.datasets}
        new_datasets: list[Dataset] = []
        added: list[str] = []
        kept: list[str] = []
        for p in paths:
            if p in existing:
                ds = existing[p]
                # Reload meta in case scores or rejected list changed on disk.
                ds.meta = ds._load_meta()  # noqa: SLF001
                new_datasets.append(ds)
                kept.append(f"{p.parent.parent.name}/{p.name}")
            else:
                ds = self.build_dataset(p)
                if ds is None:
                    continue
                # Pre-warm prompts (cheap once warmed).
                _ = ds.prompts
                new_datasets.append(ds)
                added.append(f"{p.parent.parent.name}/{p.name}")
        # Drop datasets whose paths are no longer in the discovery set.
        removed = [
            f"{p.parent.parent.name}/{p.name}"
            for p in existing if p not in set(paths)
        ]
        self.datasets = new_datasets
        return {
            "datasetCount": len(self.datasets),
            "added": added,
            "removed": removed,
            "kept": kept,
        }

    def build_state(self) -> dict[str, Any]:
        # Serialize buckets once, shared across datasets.
        buckets_payload: dict[str, Any] = {}
        for metric, defs in BUCKETS.items():
            clean = []
            for b in defs:
                lo = b["lo"]
                hi = b["hi"]
                clean.append({
                    "label": b["label"],
                    "lo": -1e18 if lo == float("-inf") else lo,
                    "hi": 1e18 if hi == float("inf") else hi,
                    "color": b["color"],
                    "desc": b["desc"],
                })
            buckets_payload[metric] = clean

        score_order = list(BUCKETS.keys())
        seen = set(score_order)
        datasets_state = []
        for idx, ds in enumerate(self.datasets):
            ds_state = ds.as_state()
            ds_state["idx"] = idx
            for rec in ds_state["scores"].values():
                for k in rec:
                    if k not in seen:
                        score_order.append(k)
                        seen.add(k)
            datasets_state.append(ds_state)

        return {
            "datasets": datasets_state,
            "buckets": buckets_payload,
            "scoreOrder": score_order,
            "previewSize": self.config.preview_size,
            "thumbSize": self.config.thumb_size,
        }

    def image_bytes(self, ds_idx: int, view_id: int, size: int) -> bytes:
        if ds_idx < 0 or ds_idx >= len(self.datasets):
            raise KeyError(f"dataset index out of range: {ds_idx}")
        return self.datasets[ds_idx].image_bytes(view_id, size)


@lru_cache(maxsize=8192)
def render_image(path: Path, *, size: int, mtime_ns: int) -> bytes:
    del mtime_ns
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((size, size), resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "ScoreViewer/1.0"

    @property
    def app(self) -> ViewerApp:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_json(self.app.build_state())
            return
        if parsed.path.startswith("/api/image/"):
            self._serve_image(parsed)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/refresh":
            try:
                result = self.app.rescan()
            except Exception as e:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Refresh failed: {e}")
                return
            self._send_json(result)
            return
        # /api/save/<ds_idx> body: {"rejected": [int, ...]}
        if parsed.path.startswith("/api/save/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 3:
                self.send_error(HTTPStatus.BAD_REQUEST, "Use /api/save/<ds_idx>"); return
            try:
                ds_idx = int(parts[2])
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Bad ds_idx"); return
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > 8 * 1024 * 1024:
                self.send_error(HTTPStatus.BAD_REQUEST, "Empty or oversized body"); return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as e:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Bad JSON: {e}"); return
            ids = payload.get("rejected") if isinstance(payload, dict) else None
            if not isinstance(ids, list):
                self.send_error(HTTPStatus.BAD_REQUEST, "rejected must be a list"); return
            try:
                ds = self.app.datasets[ds_idx]
            except IndexError:
                self.send_error(HTTPStatus.NOT_FOUND, f"No dataset {ds_idx}"); return
            try:
                result = ds.save_rejected(ids)
            except Exception as e:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Save failed: {e}"); return
            self._send_json(result)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def _serve_image(self, parsed) -> None:
        # Accepts /api/image/<ds_idx>/<view_id> (new) or /api/image/<view_id>
        # (single-dataset fallback for backwards compatibility).
        parts = parsed.path.strip("/").split("/")
        try:
            if len(parts) == 4:
                ds_idx = int(parts[2])
                vid = int(parts[3])
            elif len(parts) == 3:
                ds_idx = 0
                vid = int(parts[2])
            else:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid image URL"); return
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid ids"); return
        qs = parse_qs(parsed.query)
        try:
            size = int(qs.get("size", [str(self.app.config.thumb_size)])[0])
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid size"); return
        size = max(32, min(size, max(self.app.config.preview_size, self.app.config.thumb_size)))
        try:
            payload = self.app.image_bytes(ds_idx, vid, size)
        except KeyError as e:
            self.send_error(HTTPStatus.NOT_FOUND, str(e)); return
        self._send(payload, "image/jpeg")

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[score_viewer] {self.address_string()} - {fmt % args}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_single_view_dir(base: Path, view_subdir_hint: str | None) -> Path:
    base = base.expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(base)
    if not base.is_dir():
        raise ValueError(f"Expected directory, got: {base}")
    # If the user pointed at the view subdir directly (e.g. .../gen_view_masked),
    # climb up one level.
    if view_subdir_hint and base.name == view_subdir_hint:
        return base.parent
    if base.name in {s for s, _ in _VIEW_SOURCE_CANDIDATES}:
        return base.parent
    return base


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Browser viewer for view_scores grouped into buckets.")
    ap.add_argument("paths", type=Path, nargs="*",
                    help="Zero or more single-view style directories (e.g. .../single_view/civitai2.0). "
                         "Each dataset is shown stacked on the same page. "
                         "Combine with --scan-root to auto-discover.")
    ap.add_argument("--scan-root", type=Path, action="append", default=[],
                    help="Auto-discover scored datasets under this root. The "
                         "viewer scans <root>/*/single_view/*/meta.json and "
                         "includes every dataset whose JSON has at least one "
                         "view_scores entry. Repeat to scan multiple roots. "
                         "Hit the in-page Refresh button to pick up newly-"
                         "scored datasets without restarting.")
    ap.add_argument("--scan-root-deep", type=Path, action="append", default=[],
                    help="Like --scan-root but recurses to any depth, matching "
                         "<root>/**/single_view/*/meta.json. Use for nested "
                         "layouts like mesh/transfer_mesh/<category>/<mesh>/"
                         "single_view/<style>/.")
    ap.add_argument("--view-subdir", default=None,
                    help="Force a specific view subdir (e.g. gen_view_masked). "
                         "Default: auto-detect gen_view_masked → gen_view_decomposite(basecolor) → gen_view_super(basecolor).")
    ap.add_argument("--view-channel", default=None,
                    help="Channel suffix (e.g. basecolor) to use with --view-subdir.")
    ap.add_argument("--meta-subdir", default="meta")
    ap.add_argument("--meta-filename", default="meta.json")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=10014)
    ap.add_argument("--thumb-size", type=int, default=192)
    ap.add_argument("--preview-size", type=int, default=960)
    return ap.parse_args()


def discover_scored_datasets(root: Path, meta_filename: str = "meta.json",
                              deep: bool = False) -> list[Path]:
    """Return single_view directories under *root* whose meta.json has at
    least one ``view_scores`` entry. Sorted by mesh-name then sv-subdir.

    With ``deep=True`` the scan recurses to any depth so nested layouts
    (e.g. ``transfer_mesh/<category>/<mesh>/single_view/<style>/``) are
    discovered."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    if deep:
        # rglob anchors at the meta_filename and walks back up to verify
        # that the parent path looks like .../single_view/<style>/.
        meta_iter = root.rglob(meta_filename)
    else:
        # Glob: <root>/<mesh>/single_view/<sv_subdir>/<meta_filename>
        meta_iter = root.glob(f"*/single_view/*/{meta_filename}")
    for meta_fp in meta_iter:
        if deep and (len(meta_fp.parts) < 3 or meta_fp.parent.parent.name != "single_view"):
            continue
        try:
            with meta_fp.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        scores = meta.get("view_scores") if isinstance(meta, dict) else None
        if isinstance(scores, dict) and scores:
            found.append(meta_fp.parent)
    # mesh_dir.name then sv_subdir (parent.name)
    return sorted(found, key=lambda p: (p.parent.parent.name, p.name))


def main() -> None:
    args = parse_args()
    import sys as _sys
    import time as _time
    # Line-buffer stdout so startup messages appear in real time when redirected.
    try:
        _sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    def collect_paths() -> list[Path]:
        """Combine positional --paths and --scan-root discoveries, dedup."""
        out: list[Path] = []
        seen: set[Path] = set()
        for raw in args.paths:
            p = resolve_single_view_dir(raw, args.view_subdir)
            if p not in seen:
                seen.add(p); out.append(p)
        for root in args.scan_root:
            for p in discover_scored_datasets(Path(root), args.meta_filename):
                if p not in seen:
                    seen.add(p); out.append(p)
        for root in args.scan_root_deep:
            for p in discover_scored_datasets(Path(root), args.meta_filename, deep=True):
                if p not in seen:
                    seen.add(p); out.append(p)
        return out

    def build_dataset(svd: Path) -> Dataset | None:
        try:
            ds = Dataset(
                single_view_dir=svd,
                meta_subdir=args.meta_subdir,
                meta_filename=args.meta_filename,
                view_subdir=args.view_subdir,
                view_channel=args.view_channel,
            )
            return ds
        except FileNotFoundError as e:
            print(f"[skip] {svd}: {e}")
            return None

    datasets: list[Dataset] = []
    for svd in collect_paths():
        ds = build_dataset(svd)
        if ds is None:
            continue
        datasets.append(ds)
        print(f"[{ds.name}/{ds.single_view_dir.name}] view source: {ds.view_dir.name}"
              f"{f' (channel={ds.view_channel})' if ds.view_channel else ''} "
              f"| {len(ds.view_paths)} views | meta: {ds.meta_path}")
    if not datasets:
        raise SystemExit(
            "No datasets found. Provide positional paths or --scan-root with "
            "scored datasets present.")

    # Pre-warm prompt cache so the first /api/state call doesn't stall the
    # browser for 30+ seconds.
    for ds in datasets:
        t0 = _time.time()
        n = len(ds.prompts)
        print(f"[{ds.name}] loaded {n} prompts in {_time.time() - t0:.1f}s")

    config = ViewerConfig(
        thumb_size=max(64, args.thumb_size),
        preview_size=max(256, args.preview_size),
    )
    app = ViewerApp(config, datasets)
    # Make discovery functions accessible to /api/refresh.
    app.collect_paths = collect_paths   # type: ignore[attr-defined]
    app.build_dataset = build_dataset   # type: ignore[attr-defined]
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.app = app  # type: ignore[attr-defined]
    host, port = server.server_address[:2]
    print(f"Open http://{host}:{port}  ({len(datasets)} dataset{'s' if len(datasets) != 1 else ''})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
