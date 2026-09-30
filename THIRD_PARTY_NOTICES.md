# Sources and license boundaries

## Academic website references

The information architecture was informed by [Nerfies](https://github.com/nerfies/nerfies.github.io) and [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), consulted on 2026-09-21 and 2026-09-22. Their websites/templates state CC BY-SA 4.0. This release implements its own HTML/CSS/JS and does not bundle template author photos, analytics, papers, or research claims. If template code is incorporated later, retain its attribution and applicable share-alike license. Layout inspiration is credited on the project page.

## Research code snapshots

- `reference_code/gaussian_grouping/`: local `code/GaussianModel`, derived from Gaussian Grouping / Gaussian Splatting; upstream copyright headers and the local LICENSE are preserved.
- `reference_code/3dgs_proxy/`: local 3DGRUT geometry helpers plus two leisaac USD assembly helpers. Their exact origins are listed per file in `docs/source_manifest.json`.
- `reference_code/leisaac/`: local leisaac integration snapshots, with its original LICENSE.
- Full upstream checkouts, CUDA extensions, checkpoints, datasets and vendor SDK binaries are not bundled. Their own licenses continue to apply when installed separately.

## Project material

The root MIT license applies to project-authored Python utilities, release scripts, tests and the newly authored website code. It does not supersede any third-party file notice. Manuscript PDFs, paper figures, videos and example data are accompanying research material and are not automatically relicensed under MIT. Their reuse rights must be assessed separately; this local preparation is not a claim that every upstream dependency is permissively licensed.
