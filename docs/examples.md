# Examples

These notebooks demonstrate workflows using your own imagery, lidar rasters, and annotations.
Replace the paths in each notebook before running it. The source data and base checkpoint are supplied separately; see [Data Preparation](data_preparation.md) and [Training](training.md#base-checkpoint).

Open a notebook with a Python environment containing `samstars` and Jupyter.
The code uses the installed package and can run outside a repository checkout.

| Notebook | Purpose |
| --- | --- |
| [Train and Segment One Area](train_and_segment_one_area.ipynb) | Train on a labeled area and segment another area. |
| [Train and Segment a Tiled Area](train_and_segment_tiled_area.ipynb) | Train a bundle, prepare larger rasters as tiles, and segment them. |
| [Retrain Refinement for One Area](retrain_stardist_one_area.ipynb) | Reuse saved proposals to train and run a replacement refinement model. |

The first two notebooks show the essential settings first, with optional tuning at the end.
For evaluation, use areas held out from training; the notebooks demonstrate execution of the workflow.

See [Quick Start](quickstart.md) for a compact example and [Advanced Workflows](advanced_workflows.md) for intermediate files and partial reruns.
