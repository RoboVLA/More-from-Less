# Retained research-code snapshots

These snapshots preserve existing implementations and copyright headers. Per-file origins and SHA-256 hashes are in the [source manifest](../docs/source_manifest.json). Original line endings and whitespace are retained through `.gitattributes`; original-language comments and scene filenames may remain. English release documentation describes how to use the snapshots without rewriting their provenance.

## Gaussian Grouping

[gaussian_grouping](gaussian_grouping/) includes object-level training, rendering/mask entries and supporting modules. Upstream: [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping). The manifest records the local baseline commit and exact modified snapshot hashes.

Configure upstream Gaussian Splatting CUDA extensions, PyTorch, COLMAP cameras and instance-ID masks, then use this directory as a matching-structure overlay. Submodules, weights, DEVA/SAM/LAMA and training data are not bundled. The saved training JSON identifies an existing configuration, not a verified configuration for Tables 1–8. This module trains object-level Gaussians, not bimanual VLA policies.

## Physical proxies

[3dgs_proxy](3dgs_proxy/) includes cropping/splitting, mesh conversion, USD hinge assembly and associated entry points. CPU/Open3D dependencies are listed in [requirements-geometry.txt](../requirements-geometry.txt). Gaussian USDZ export requires [NVIDIA 3DGRUT](https://github.com/nv-tlabs/3dgrut); USD/PhysX assembly requires Isaac Sim Python/`pxr`.

```bash
python reference_code/3dgs_proxy/split_gaussian_ply_by_plane.py --help
python reference_code/3dgs_proxy/crop_gaussian_ply.py --help
```

Full pipelines retain compiler, CUDA and scene defaults. Supply actual paths, axes, scales and output locations. Mass, friction and joints come from measurements or priors; geometric conversion does not validate physics automatically.

## Simulator interfaces

[leisaac](leisaac/) retains policy inference, action processing, format conversion and hinge interaction in upstream-relative locations. A complete [leisaac](https://github.com/LightwheelAI/leisaac) and Isaac Lab environment is required. Original demonstrations and weights are not included. These generic interfaces are separate from the new [method reference](../morefromless/method/).

## Licensing

Source directories preserve their original licenses and file-level notices. External submodules can have additional terms. The root MIT license does not replace third-party licenses.
