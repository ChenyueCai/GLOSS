#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local browser tool for reviewing single-view renders and rejecting training views."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps


DEFAULT_CHANNEL = "basecolor"
DEFAULT_META_KEY = "exclude_from_training_indices"
DEFAULT_THUMB_SIZE = 192
DEFAULT_PREVIEW_SIZE = 960
DEFAULT_PAGE_SIZE = 120
VIEW_NAME_RE = re.compile(r"^view(\d+)\.(?P<channel>[^.]+)\.png$")

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training View Reviewer</title>
  <style>
    :root {
      --bg: #f3efe4;
      --panel: rgba(255, 252, 245, 0.92);
      --panel-strong: #fff9ef;
      --line: #d9cdb5;
      --ink: #1f1a14;
      --muted: #6d6254;
      --accent: #0c6d62;
      --accent-soft: rgba(12, 109, 98, 0.12);
      --danger: #b42318;
      --danger-soft: rgba(180, 35, 24, 0.12);
      --shadow: 0 18px 48px rgba(77, 55, 28, 0.12);
      --thumb-width: 176px;
      --radius: 18px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 255, 255, 0.8), transparent 28%),
        radial-gradient(circle at bottom right, rgba(196, 152, 62, 0.12), transparent 24%),
        linear-gradient(180deg, #f8f4ea 0%, #efe7d7 100%);
    }

    button, input, select {
      font: inherit;
      color: inherit;
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
      gap: 18px;
      padding: 18px;
    }

    .main,
    .sidebar {
      background: var(--panel);
      border: 1px solid rgba(122, 104, 78, 0.18);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .main {
      padding: 18px;
      min-width: 0;
    }

    .sidebar {
      position: sticky;
      top: 18px;
      align-self: start;
      padding: 18px;
      display: grid;
      gap: 14px;
      max-height: calc(100vh - 36px);
      overflow: auto;
    }

    .header {
      display: grid;
      gap: 14px;
      margin-bottom: 18px;
    }

    .title-row,
    .toolbar,
    .status-row,
    .pager {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 12px;
      align-items: center;
    }

    .title {
      margin: 0;
      font-size: 1.55rem;
      font-weight: 650;
      letter-spacing: -0.02em;
    }

    .subtle {
      color: var(--muted);
      font-size: 0.94rem;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(122, 104, 78, 0.16);
    }

    .toolbar label,
    .pager label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(122, 104, 78, 0.16);
    }

    .toolbar input,
    .toolbar select,
    .pager input {
      border: 0;
      background: transparent;
      min-width: 52px;
      outline: none;
    }

    .btn {
      border: 1px solid transparent;
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
      background: rgba(255, 255, 255, 0.72);
    }

    .btn:hover {
      transform: translateY(-1px);
    }

    .btn-primary {
      background: var(--accent);
      color: #f6fbfa;
    }

    .btn-danger {
      background: var(--danger);
      color: #fff8f6;
    }

    .btn-outline {
      border-color: rgba(122, 104, 78, 0.22);
    }

    .btn:disabled {
      cursor: default;
      transform: none;
      opacity: 0.5;
    }

    .grid {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fill, minmax(var(--thumb-width), 1fr));
    }

    .card {
      border-radius: var(--radius);
      overflow: hidden;
      border: 1px solid rgba(122, 104, 78, 0.16);
      background: rgba(255, 255, 255, 0.84);
      display: grid;
      gap: 0;
      position: relative;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }

    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 24px rgba(82, 60, 31, 0.14);
    }

    .card.selected {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }

    .card.rejected {
      border-color: rgba(180, 35, 24, 0.45);
      box-shadow: 0 0 0 3px var(--danger-soft);
    }

    .preview-trigger {
      border: 0;
      background: transparent;
      padding: 0;
      cursor: pointer;
      text-align: left;
    }

    .thumb-wrap {
      aspect-ratio: 1 / 1;
      background:
        linear-gradient(45deg, rgba(0, 0, 0, 0.03) 25%, transparent 25%, transparent 75%, rgba(0, 0, 0, 0.03) 75%),
        linear-gradient(45deg, rgba(0, 0, 0, 0.03) 25%, transparent 25%, transparent 75%, rgba(0, 0, 0, 0.03) 75%);
      background-size: 22px 22px;
      background-position: 0 0, 11px 11px;
    }

    .thumb-wrap img,
    .hero img {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
    }

    .card-body {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px 12px;
    }

    .card-name {
      font-weight: 600;
      font-size: 0.93rem;
    }

    .card-state {
      color: var(--muted);
      font-size: 0.84rem;
    }

    .toggle {
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 1;
      border-radius: 999px;
      border: 1px solid rgba(122, 104, 78, 0.2);
      background: rgba(255, 250, 241, 0.92);
      padding: 7px 11px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
    }

    .toggle.rejected {
      border-color: rgba(180, 35, 24, 0.32);
      background: rgba(180, 35, 24, 0.96);
      color: #fff7f4;
    }

    .hero {
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid rgba(122, 104, 78, 0.16);
      background: #fff;
      aspect-ratio: 1 / 1;
    }

    .sidebar h2,
    .sidebar h3 {
      margin: 0;
      font-size: 1rem;
    }

    .meta-list {
      display: grid;
      gap: 8px;
    }

    .meta-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .meta-item strong {
      color: var(--ink);
    }

    .help {
      display: grid;
      gap: 6px;
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.66);
      border: 1px solid rgba(122, 104, 78, 0.14);
      font-size: 0.9rem;
      color: var(--muted);
    }

    .save-status {
      min-height: 1.25rem;
      font-size: 0.9rem;
      color: var(--muted);
    }

    .save-status.error {
      color: var(--danger);
    }

    .warning {
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(180, 35, 24, 0.08);
      color: #8a2018;
      border: 1px solid rgba(180, 35, 24, 0.16);
      font-size: 0.9rem;
    }

    @media (max-width: 1080px) {
      .shell {
        grid-template-columns: 1fr;
      }

      .sidebar {
        position: static;
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <main class="main">
      <section class="header">
        <div class="title-row">
          <div>
            <h1 class="title">Training View Reviewer</h1>
            <div class="subtle" id="datasetLabel"></div>
          </div>
        </div>

        <div class="toolbar">
          <div class="pill" id="countsPill"></div>
          <label>Filter
            <select id="filterMode">
              <option value="all">All views</option>
              <option value="kept">Kept only</option>
              <option value="rejected">Rejected only</option>
            </select>
          </label>
          <label>Thumb
            <select id="thumbSize">
              <option value="144">144 px</option>
              <option value="176">176 px</option>
              <option value="192" selected>192 px</option>
              <option value="224">224 px</option>
            </select>
          </label>
          <label>Page size
            <select id="pageSize">
              <option value="60">60</option>
              <option value="120" selected>120</option>
              <option value="180">180</option>
              <option value="240">240</option>
            </select>
          </label>
          <label>Jump to view
            <input id="jumpInput" type="number" min="0" step="1" placeholder="e.g. 217">
          </label>
          <button class="btn btn-outline" id="clearDraftBtn" type="button">Reset Unsaved</button>
          <button class="btn btn-primary" id="saveBtn" type="button">Save To Meta</button>
        </div>

        <div class="status-row">
          <div class="subtle" id="rangeLabel"></div>
          <div class="save-status" id="saveStatus"></div>
        </div>

        <div class="pager">
          <button class="btn btn-outline" id="prevPageBtn" type="button">Prev Page</button>
          <label>Page
            <input id="pageInput" type="number" min="1" step="1">
          </label>
          <div class="pill" id="pageCountLabel"></div>
          <button class="btn btn-outline" id="nextPageBtn" type="button">Next Page</button>
        </div>
      </section>

      <section class="grid" id="grid"></section>
    </main>

    <aside class="sidebar">
      <div class="hero"><img id="heroImage" alt="Selected view preview"></div>
      <div>
        <h2 id="heroTitle">No view selected</h2>
        <div class="subtle" id="heroPath"></div>
      </div>
      <div class="toolbar">
        <button class="btn btn-outline" id="prevViewBtn" type="button">Prev</button>
        <button class="btn btn-outline" id="nextViewBtn" type="button">Next</button>
        <button class="btn btn-danger" id="toggleBtn" type="button">Reject</button>
      </div>
      <div class="meta-list" id="metaList"></div>
      <div class="warning" id="orphanWarning" hidden></div>
      <div class="help">
        <div><strong>Mouse:</strong> click a card to preview, use the card button or sidebar button to toggle reject.</div>
        <div><strong>Keyboard:</strong> arrows move, <strong>x</strong> or <strong>space</strong> toggles reject, <strong>s</strong> saves.</div>
        <div><strong>Storage:</strong> unsaved changes are kept in browser local storage until you save or reset them.</div>
      </div>
    </aside>
  </div>

  <script>
    const state = {
      metaPath: null,
      metaKey: null,
      datasetPath: null,
      channel: null,
      pageSize: 120,
      previewSize: 960,
      thumbSize: 192,
      viewIds: [],
      rejectedIds: [],
      orphanedRejectedIds: [],
      imagePaths: {},
    };

    let filteredIds = [];
    let selectedId = null;
    let currentPage = 0;
    let currentRejected = new Set();
    let savedRejected = new Set();

    const els = {
      countsPill: document.getElementById("countsPill"),
      datasetLabel: document.getElementById("datasetLabel"),
      filterMode: document.getElementById("filterMode"),
      thumbSize: document.getElementById("thumbSize"),
      pageSize: document.getElementById("pageSize"),
      jumpInput: document.getElementById("jumpInput"),
      clearDraftBtn: document.getElementById("clearDraftBtn"),
      saveBtn: document.getElementById("saveBtn"),
      saveStatus: document.getElementById("saveStatus"),
      rangeLabel: document.getElementById("rangeLabel"),
      pageInput: document.getElementById("pageInput"),
      pageCountLabel: document.getElementById("pageCountLabel"),
      prevPageBtn: document.getElementById("prevPageBtn"),
      nextPageBtn: document.getElementById("nextPageBtn"),
      grid: document.getElementById("grid"),
      heroImage: document.getElementById("heroImage"),
      heroTitle: document.getElementById("heroTitle"),
      heroPath: document.getElementById("heroPath"),
      prevViewBtn: document.getElementById("prevViewBtn"),
      nextViewBtn: document.getElementById("nextViewBtn"),
      toggleBtn: document.getElementById("toggleBtn"),
      metaList: document.getElementById("metaList"),
      orphanWarning: document.getElementById("orphanWarning"),
    };

    const setToSortedArray = (input) => Array.from(input).sort((a, b) => a - b);
    const formatViewId = (id) => `view${String(id).padStart(4, "0")}`;
    const draftKey = () => `training-view-review:${state.metaPath}:${state.metaKey}`;

    function loadDraft() {
      try {
        const raw = window.localStorage.getItem(draftKey());
        if (!raw) {
          return null;
        }
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) {
          return null;
        }
        return parsed
          .map((value) => Number(value))
          .filter((value) => Number.isInteger(value));
      } catch (error) {
        return null;
      }
    }

    function persistDraft() {
      window.localStorage.setItem(draftKey(), JSON.stringify(setToSortedArray(currentRejected)));
    }

    function clearDraftStorage() {
      window.localStorage.removeItem(draftKey());
    }

    function sameSets(left, right) {
      if (left.size !== right.size) {
        return false;
      }
      for (const value of left) {
        if (!right.has(value)) {
          return false;
        }
      }
      return true;
    }

    function currentFilterMode() {
      return els.filterMode.value;
    }

    function updateCounts() {
      const rejectedCount = currentRejected.size;
      const keptCount = state.viewIds.length - rejectedCount;
      const dirty = sameSets(currentRejected, savedRejected) ? "saved" : "unsaved";
      els.countsPill.textContent =
        `${state.viewIds.length} views | ${keptCount} kept | ${rejectedCount} rejected | ${dirty}`;
      els.saveBtn.disabled = sameSets(currentRejected, savedRejected);
    }

    function updateSaveStatus(message, isError = false) {
      els.saveStatus.textContent = message || "";
      els.saveStatus.classList.toggle("error", Boolean(isError));
    }

    function recomputeFilteredIds() {
      const mode = currentFilterMode();
      filteredIds = state.viewIds.filter((id) => {
        const rejected = currentRejected.has(id);
        if (mode === "kept") {
          return !rejected;
        }
        if (mode === "rejected") {
          return rejected;
        }
        return true;
      });
      if (!filteredIds.length) {
        selectedId = null;
        currentPage = 0;
        return;
      }
      if (selectedId === null || !filteredIds.includes(selectedId)) {
        selectedId = filteredIds[0];
      }
      const selectedIndex = filteredIds.indexOf(selectedId);
      currentPage = Math.floor(selectedIndex / Number(els.pageSize.value));
    }

    function visibleIds() {
      const pageSize = Number(els.pageSize.value);
      const start = currentPage * pageSize;
      return filteredIds.slice(start, start + pageSize);
    }

    function updatePageWidgets() {
      const pageSize = Number(els.pageSize.value);
      const totalPages = Math.max(1, Math.ceil(filteredIds.length / pageSize));
      currentPage = Math.min(Math.max(currentPage, 0), totalPages - 1);
      const start = filteredIds.length ? currentPage * pageSize + 1 : 0;
      const end = Math.min(filteredIds.length, (currentPage + 1) * pageSize);
      els.rangeLabel.textContent = `${start}-${end} of ${filteredIds.length} visible views`;
      els.pageInput.value = totalPages ? String(currentPage + 1) : "1";
      els.pageInput.max = String(totalPages);
      els.pageCountLabel.textContent = `${totalPages} page${totalPages === 1 ? "" : "s"}`;
      els.prevPageBtn.disabled = currentPage <= 0;
      els.nextPageBtn.disabled = currentPage >= totalPages - 1;
    }

    function updateSidebar() {
      if (selectedId === null) {
        els.heroTitle.textContent = "No view selected";
        els.heroPath.textContent = "";
        els.heroImage.removeAttribute("src");
        els.toggleBtn.disabled = true;
        els.metaList.innerHTML = "";
        return;
      }

      const rejected = currentRejected.has(selectedId);
      els.heroTitle.textContent = `${formatViewId(selectedId)}.${state.channel}.png`;
      els.heroPath.textContent = state.imagePaths[String(selectedId)] || "";
      els.heroImage.src = `/api/image/${selectedId}?size=${state.previewSize}&v=${encodeURIComponent(state.cacheToken)}`;
      els.toggleBtn.disabled = false;
      els.toggleBtn.textContent = rejected ? "Keep This View" : "Reject This View";
      els.toggleBtn.classList.toggle("btn-danger", !rejected);
      els.toggleBtn.classList.toggle("btn-outline", rejected);
      els.metaList.innerHTML = `
        <div class="meta-item"><span>Status</span><strong>${rejected ? "Rejected for train" : "Kept for train"}</strong></div>
        <div class="meta-item"><span>View id</span><strong>${selectedId}</strong></div>
        <div class="meta-item"><span>Meta key</span><strong>${state.metaKey}</strong></div>
        <div class="meta-item"><span>Page</span><strong>${currentPage + 1}</strong></div>
      `;
    }

    function createCard(id) {
      const rejected = currentRejected.has(id);
      const card = document.createElement("article");
      card.className = "card";
      if (id === selectedId) {
        card.classList.add("selected");
      }
      if (rejected) {
        card.classList.add("rejected");
      }

      const previewTrigger = document.createElement("button");
      previewTrigger.type = "button";
      previewTrigger.className = "preview-trigger";
      previewTrigger.innerHTML = `
        <div class="thumb-wrap">
          <img alt="${formatViewId(id)}" loading="lazy" src="/api/image/${id}?size=${Number(els.thumbSize.value)}&v=${encodeURIComponent(state.cacheToken)}">
        </div>
        <div class="card-body">
          <div>
            <div class="card-name">${formatViewId(id)}</div>
            <div class="card-state">${rejected ? "Rejected" : "Kept"}</div>
          </div>
        </div>
      `;
      previewTrigger.addEventListener("click", () => selectView(id));

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = `toggle${rejected ? " rejected" : ""}`;
      toggle.textContent = rejected ? "Rejected" : "Reject";
      toggle.addEventListener("click", (event) => {
        event.stopPropagation();
        selectView(id);
        toggleSelected();
      });

      card.appendChild(previewTrigger);
      card.appendChild(toggle);
      return card;
    }

    function renderGrid() {
      updatePageWidgets();
      const fragment = document.createDocumentFragment();
      for (const id of visibleIds()) {
        fragment.appendChild(createCard(id));
      }
      els.grid.replaceChildren(fragment);
      updateSidebar();
      updateCounts();
    }

    function selectView(id) {
      selectedId = id;
      const pageSize = Number(els.pageSize.value);
      const selectedIndex = filteredIds.indexOf(id);
      if (selectedIndex >= 0) {
        currentPage = Math.floor(selectedIndex / pageSize);
      }
      renderGrid();
    }

    function toggleSelected() {
      if (selectedId === null) {
        return;
      }
      if (currentRejected.has(selectedId)) {
        currentRejected.delete(selectedId);
      } else {
        currentRejected.add(selectedId);
      }
      persistDraft();
      updateSaveStatus("");
      renderGrid();
    }

    function moveSelection(delta) {
      if (!filteredIds.length) {
        return;
      }
      if (selectedId === null) {
        selectView(filteredIds[0]);
        return;
      }
      const currentIndex = filteredIds.indexOf(selectedId);
      const nextIndex = Math.min(filteredIds.length - 1, Math.max(0, currentIndex + delta));
      selectView(filteredIds[nextIndex]);
    }

    function moveSelectionByGrid(rowsDelta) {
      const width = els.grid.clientWidth || window.innerWidth;
      const columns = Math.max(1, Math.floor(width / (Number(els.thumbSize.value) + 22)));
      moveSelection(rowsDelta * columns);
    }

    async function saveState() {
      const payload = { rejected_ids: setToSortedArray(currentRejected) };
      updateSaveStatus("Saving...");
      try {
        const response = await fetch("/api/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.error || "Save failed");
        }
        const visibleRejectedIds = result.visible_rejected_ids || result.rejected_ids;
        savedRejected = new Set(visibleRejectedIds);
        currentRejected = new Set(visibleRejectedIds);
        clearDraftStorage();
        updateSaveStatus(`Saved ${result.rejected_ids.length} rejected view ids at ${result.saved_at}.`);
        renderGrid();
      } catch (error) {
        updateSaveStatus(error.message || String(error), true);
      }
    }

    function resetUnsaved() {
      currentRejected = new Set(savedRejected);
      clearDraftStorage();
      updateSaveStatus("Reset to last saved state.");
      recomputeFilteredIds();
      renderGrid();
    }

    function jumpToView(rawValue) {
      if (rawValue === "") {
        return;
      }
      const target = Number(rawValue);
      if (!Number.isInteger(target) || !state.viewIds.includes(target)) {
        updateSaveStatus(`View ${rawValue} is not available in this review set.`, true);
        return;
      }
      if (!filteredIds.includes(target)) {
        els.filterMode.value = "all";
        recomputeFilteredIds();
      }
      updateSaveStatus("");
      selectView(target);
    }

    function attachEvents() {
      els.filterMode.addEventListener("change", () => {
        recomputeFilteredIds();
        renderGrid();
      });

      els.thumbSize.addEventListener("change", () => {
        document.documentElement.style.setProperty("--thumb-width", `${Number(els.thumbSize.value) - 16}px`);
        renderGrid();
      });

      els.pageSize.addEventListener("change", () => {
        recomputeFilteredIds();
        renderGrid();
      });

      els.pageInput.addEventListener("change", () => {
        const totalPages = Math.max(1, Math.ceil(filteredIds.length / Number(els.pageSize.value)));
        const nextPage = Math.min(totalPages, Math.max(1, Number(els.pageInput.value || "1")));
        currentPage = nextPage - 1;
        renderGrid();
      });

      els.prevPageBtn.addEventListener("click", () => {
        currentPage -= 1;
        renderGrid();
      });

      els.nextPageBtn.addEventListener("click", () => {
        currentPage += 1;
        renderGrid();
      });

      els.prevViewBtn.addEventListener("click", () => moveSelection(-1));
      els.nextViewBtn.addEventListener("click", () => moveSelection(1));
      els.toggleBtn.addEventListener("click", () => toggleSelected());
      els.saveBtn.addEventListener("click", () => saveState());
      els.clearDraftBtn.addEventListener("click", () => resetUnsaved());

      els.jumpInput.addEventListener("change", () => jumpToView(els.jumpInput.value));
      els.jumpInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          jumpToView(els.jumpInput.value);
        }
      });

      window.addEventListener("keydown", (event) => {
        const target = event.target;
        if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) {
          return;
        }
        if (event.key === "ArrowRight") {
          event.preventDefault();
          moveSelection(1);
        } else if (event.key === "ArrowLeft") {
          event.preventDefault();
          moveSelection(-1);
        } else if (event.key === "ArrowDown") {
          event.preventDefault();
          moveSelectionByGrid(1);
        } else if (event.key === "ArrowUp") {
          event.preventDefault();
          moveSelectionByGrid(-1);
        } else if (event.key === "x" || event.key === "X" || event.key === " ") {
          event.preventDefault();
          toggleSelected();
        } else if (event.key === "s" || event.key === "S") {
          event.preventDefault();
          saveState();
        }
      });
    }

    async function boot() {
      const response = await fetch("/api/state");
      const payload = await response.json();
      Object.assign(state, payload);
      els.datasetLabel.textContent = `${state.datasetPath} | meta: ${state.metaPath}`;
      els.pageSize.value = String(state.pageSize);
      els.thumbSize.value = String(state.thumbSize);
      document.documentElement.style.setProperty("--thumb-width", `${Number(els.thumbSize.value) - 16}px`);

      savedRejected = new Set(payload.rejected_ids);
      const draft = loadDraft();
      currentRejected = new Set(draft && draft.length ? draft : payload.rejected_ids);
      if (draft && !sameSets(currentRejected, savedRejected)) {
        updateSaveStatus("Restored unsaved draft from local storage.");
      }

      if (payload.orphaned_rejected_ids.length) {
        els.orphanWarning.hidden = false;
        els.orphanWarning.textContent =
          `Preserving ${payload.orphaned_rejected_ids.length} rejected ids already stored in meta.json that are not present in this review folder: ${payload.orphaned_rejected_ids.join(", ")}`;
      }

      recomputeFilteredIds();
      attachEvents();
      renderGrid();
    }

    boot().catch((error) => {
      updateSaveStatus(error.message || String(error), true);
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a local browser tool for marking single-view images that should "
            "be excluded from training."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Either the single-view style directory (for example .../single_view/civitai2.0) "
            "or the image directory itself (for example .../gen_view_decomposite)."
        ),
    )
    parser.add_argument("--host", default="localhost", help="Host interface to bind. Default: localhost")
    parser.add_argument("--port", type=int, default=10013, help="Port to bind. Default: 10013")
    parser.add_argument(
        "--view-subdir",
        default="gen_view_decomposite",
        help="View subdirectory when passing a single-view root. Default: gen_view_decomposite",
    )
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        help="Image channel suffix to review, such as basecolor or roughness. Default: basecolor",
    )
    parser.add_argument(
        "--meta-key",
        default=DEFAULT_META_KEY,
        help=f"Top-level JSON key to save rejected ids under. Default: {DEFAULT_META_KEY}",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Optional explicit meta.json path. Default: infer from the input directory.",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=DEFAULT_THUMB_SIZE,
        help=f"Default thumbnail size in the UI. Default: {DEFAULT_THUMB_SIZE}",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=DEFAULT_PREVIEW_SIZE,
        help=f"Preview image max size in pixels. Default: {DEFAULT_PREVIEW_SIZE}",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Default page size in the UI. Default: {DEFAULT_PAGE_SIZE}",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, found {type(data).__name__}")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def normalize_index_list(values: Any, *, key_name: str) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"Expected {key_name} to be a JSON list, found {type(values).__name__}")

    result: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Boolean value {value!r} is not a valid view id in {key_name}")
        result.add(int(value))
    return sorted(result)


def infer_paths(base_path: Path, view_subdir: str, channel: str, meta_path: Path | None) -> tuple[Path, Path, Path]:
    base_path = base_path.expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path}")

    if base_path.is_dir() and base_path.name == view_subdir:
        single_view_dir = base_path.parent
        view_dir = base_path
    elif base_path.is_dir():
        single_view_dir = base_path
        view_dir = single_view_dir / view_subdir
    else:
        raise ValueError(f"Expected a directory path, got file: {base_path}")

    if not view_dir.is_dir():
        raise FileNotFoundError(f"View directory not found: {view_dir}")

    resolved_meta_path = meta_path.expanduser().resolve() if meta_path is not None else single_view_dir / "meta.json"
    if resolved_meta_path.name != "meta.json":
        raise ValueError(f"Expected a meta.json path, got: {resolved_meta_path}")

    matching = sorted(view_dir.glob(f"view*.{channel}.png"))
    if not matching:
        raise FileNotFoundError(
            f"No files matching view*.{channel}.png were found under {view_dir}"
        )

    return single_view_dir, view_dir, resolved_meta_path


@dataclass(frozen=True)
class ReviewConfig:
    single_view_dir: Path
    view_dir: Path
    meta_path: Path
    channel: str
    meta_key: str
    thumb_size: int
    preview_size: int
    page_size: int


class ReviewApp:
    def __init__(self, config: ReviewConfig):
        self.config = config
        self.view_paths = self._discover_views()
        self.saved_rejected_ids, self.orphaned_rejected_ids = self._load_saved_ids()

    def _discover_views(self) -> dict[int, Path]:
        view_paths: dict[int, Path] = {}
        for path in sorted(self.config.view_dir.glob(f"view*.{self.config.channel}.png")):
            match = VIEW_NAME_RE.match(path.name)
            if match is None or match.group("channel") != self.config.channel:
                continue
            view_paths[int(match.group(1))] = path
        if not view_paths:
            raise FileNotFoundError(
                f"No files matching view*.{self.config.channel}.png were found under {self.config.view_dir}"
            )
        return view_paths

    def _load_saved_ids(self) -> tuple[list[int], list[int]]:
        meta = load_json(self.config.meta_path)
        saved_ids = normalize_index_list(meta.get(self.config.meta_key, []), key_name=self.config.meta_key)
        present_ids = [view_id for view_id in saved_ids if view_id in self.view_paths]
        orphaned_ids = [view_id for view_id in saved_ids if view_id not in self.view_paths]
        return present_ids, orphaned_ids

    def build_state(self) -> dict[str, Any]:
        cache_token = hashlib.sha1(
            f"{self.config.view_dir}|{self.config.meta_path}|{self.config.channel}|{len(self.view_paths)}".encode("utf-8")
        ).hexdigest()[:12]
        return {
            "datasetPath": str(self.config.view_dir),
            "metaPath": str(self.config.meta_path),
            "metaKey": self.config.meta_key,
            "channel": self.config.channel,
            "cacheToken": cache_token,
            "pageSize": self.config.page_size,
            "thumbSize": self.config.thumb_size,
            "previewSize": self.config.preview_size,
            "viewIds": sorted(self.view_paths),
            "rejected_ids": self.saved_rejected_ids,
            "orphaned_rejected_ids": self.orphaned_rejected_ids,
            "imagePaths": {str(view_id): str(path) for view_id, path in self.view_paths.items()},
        }

    def save_rejected_ids(self, rejected_ids: list[int]) -> tuple[list[int], str]:
        normalized = normalize_index_list(rejected_ids, key_name="rejected_ids")
        invalid = [view_id for view_id in normalized if view_id not in self.view_paths]
        if invalid:
            raise ValueError(f"Rejected ids are not present in this view folder: {invalid}")

        final_ids = sorted(set(normalized) | set(self.orphaned_rejected_ids))
        meta = load_json(self.config.meta_path)
        meta[self.config.meta_key] = final_ids
        atomic_write_json(self.config.meta_path, meta)

        self.saved_rejected_ids = normalized
        saved_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return final_ids, saved_at

    def image_bytes(self, view_id: int, size: int) -> bytes:
        if view_id not in self.view_paths:
            raise KeyError(f"Unknown view id: {view_id}")
        path = self.view_paths[view_id]
        stat = path.stat()
        return render_image(path, size=size, mtime_ns=stat.st_mtime_ns)


@lru_cache(maxsize=4096)
def render_image(path: Path, *, size: int, mtime_ns: int) -> bytes:
    del mtime_ns
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((size, size), resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=86, optimize=True)
    return output.getvalue()


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "TrainingViewReviewer/1.0"

    @property
    def app(self) -> ReviewApp:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/state":
            self._send_json(self.app.build_state())
            return
        if parsed.path.startswith("/api/image/"):
            self._send_image(parsed)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND, "Route not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            rejected_ids = payload.get("rejected_ids", [])
            final_ids, saved_at = self.app.save_rejected_ids(rejected_ids)
        except Exception as exc:  # pragma: no cover - handled in browser
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(
            {
                "ok": True,
                "rejected_ids": final_ids,
                "visible_rejected_ids": self.app.saved_rejected_ids,
                "saved_at": saved_at,
            }
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[review_training_views] {self.address_string()} - {fmt % args}")

    def _send_image(self, parsed) -> None:
        view_id_str = parsed.path.rsplit("/", 1)[-1]
        try:
            view_id = int(view_id_str)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid view id: {view_id_str}")
            return

        query = parse_qs(parsed.query)
        try:
            size = int(query.get("size", [str(self.app.config.thumb_size)])[0])
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid size query parameter")
            return
        size = max(32, min(size, max(self.app.config.preview_size, self.app.config.thumb_size)))

        try:
            payload = self.app.image_bytes(view_id, size)
        except KeyError:
            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown view id: {view_id}")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, payload: str) -> None:
        raw = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    args = parse_args()
    single_view_dir, view_dir, meta_path = infer_paths(args.path, args.view_subdir, args.channel, args.meta_path)
    config = ReviewConfig(
        single_view_dir=single_view_dir,
        view_dir=view_dir,
        meta_path=meta_path,
        channel=args.channel,
        meta_key=args.meta_key,
        thumb_size=max(64, args.thumb_size),
        preview_size=max(256, args.preview_size),
        page_size=max(1, args.page_size),
    )
    app = ReviewApp(config)
    server = ThreadingHTTPServer((args.host, args.port), ReviewRequestHandler)
    server.app = app  # type: ignore[attr-defined]

    host, port = server.server_address[:2]
    print(f"Reviewing {len(app.view_paths)} views under {config.view_dir}")
    print(f"Writing rejected ids to {config.meta_path} under key {config.meta_key}")
    print(f"Open http://{host}:{port} in a browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down reviewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
