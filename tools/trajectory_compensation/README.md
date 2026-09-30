# Historical offline video and trajectory tools

This directory retains existing video-processing, depth-diagnostic, retargeting and weighted-fusion scripts. Per-file provenance is recorded in the [source manifest](../../docs/source_manifest.json).

These are offline tools, not the online image-reference/alignment/masked-IK/shared-QP pipeline of Section 3.3. Generated depth, manual image processing and fused trajectories are not real-robot measurements or new evaluation data.

Original scripts contain scene-specific filenames, Data directories, and compiler/video-tool defaults, including original-language strings. Inspect each script's arguments and dependencies and supply your own inputs in a separate working directory. The default example runner does not start a camera or robot.

For an example that does not need source videos, run `python paper_pipeline/run_morefromless_pipeline.py` from the repository root. Optional video dependencies are in [requirements-vision.txt](../../requirements-vision.txt). Historical algorithms are retained without modifying them to fit the current manuscript narrative.
