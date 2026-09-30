# Paper-to-code mapping

Method descriptions follow anonymous manuscript v164. Figures and results are retained from v162; v164 updated resource availability. The [asset manifest](current_manuscript.json) records hashes.

| Paper component | Implementation | Scope |
|---|---|---|
| Eqs. (4)–(7): structured supervision | [structured.py](../morefromless/method/structured.py), [kinematics.py](../morefromless/method/kinematics.py), [train.py](../morefromless/method/train.py) | New reference implementation; actual model/data/geometry adapters required |
| Algorithm 1: failure-region expansion | [expansion.py](../morefromless/method/expansion.py) | New reference implementation; simulator and acceptance callbacks required |
| Eqs. (24), (26)–(31): mapping, gate and filter | [control.py](../morefromless/method/control.py) | New reference implementation; no robot connection |
| Appendix A.4: offline curves | [reproduce_appendix.py](../morefromless/trajectory/reproduce_appendix.py) | Recomputes G/R inputs and compares all saved F components |
| Reconstruction and physical assets | [gaussian_grouping](../reference_code/gaussian_grouping/), [3dgs_proxy](../reference_code/3dgs_proxy/) | Retained source snapshots |
| Simulator and camera interfaces | [leisaac](../reference_code/leisaac/), [camera tools](../tools/gemini_camera/) | Retained source snapshots |
| Original Tables 1–8 experiments | Original checkpoints, splits and episode logs are unavailable | No complete reproduction claim |

The [pipeline](../paper_pipeline/) runs appendix reproduction by default. Explicit method stages run engineering checks and do not start original experiments with guessed settings. See [method interfaces](METHOD_REFERENCE.md).
