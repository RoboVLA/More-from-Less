# Release validation

## Method and data validation — 2026-09-30

- All 24 engineering tests passed with CPU PyTorch 2.14.0+cpu, including actual gradients, optimizer updates, FK, local SO(3) mapping, valid-step masks, complete-success acceptance, real-group weighting, asynchronous references, DLS and shared QP filtering. Tests used analytic fixtures and no robot execution.
- Appendix recomputation matched every saved pose component for 72 pouring and 75 wiping points. The internal distance reductions round to 61.2% and 62.0%.
- The website's 97 mean/SD pairs across Tables 1, 2, 4 and 6 matched the manuscript. Table 2/6 switching and layout were checked in a browser.
- Manuscript v164 remains 76 pages. Relative to v163, LaTeX changed only the availability statement; extracted PDF text changed only on page 46, which was rendered and inspected.

## Website/documentation revision — 2026-09-30

All 14 Markdown documents are in English. The revision passed 275 local link/asset checks and all 24 engineering tests; 97 table mean/SD pairs and 114 preserved source/data/media files were unchanged. The source-browser file selection, GitHub target and page header were checked in a browser. The website omits the Paper button, links explicitly to GitHub, and uses rendered GitHub documentation links. Public trajectory metadata replaces workstation-only paths; all CSV bytes, JSON trajectory values and processing parameters remain unchanged. Source snapshots retain their original content and hashes, including any original-language comments or asset names.

Method tests establish engineering behavior only. They do not demonstrate reproduction of Tables 1–8, recovery of the historical system, or hardware safety certification. See [method scope](METHOD_REFERENCE.md) and [reproducibility](REPRODUCIBILITY.md).
