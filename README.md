# MIL CONCH

Core training and evaluation code for **Cross-dataset predictions of breast cancer subtypes by H&E image-based multiple instance learning**.

This is a code-only repository. It assumes that feature arrays, labels, and split definitions have already been prepared. It intentionally excludes data preparation/conversion, image preprocessing, feature extraction, Slurm launchers, rerun-management scripts, patient data, fitted models, and experiment outputs.

## Main code

| File or directory | Purpose |
| --- | --- |
| `run_mi_classify.py` | Within-dataset training/evaluation, including cross-validation and repeated stratified holdout |
| `run_cross_dataset.py` | Source-dataset training and external-dataset evaluation |
| `sil.py` | Supervised instance learning and quantile-based score aggregation |
| `linear_classifier.py` | Linear SVM, logistic-regression, and DWD classifiers and parameter selection |
| `util.py` | Input loading required by the training/evaluation entry points |
| `CLAM/main.py`, `CLAM/eval.py` | CLAM training and evaluation |
| `MambaMIL/main.py`, `MambaMIL/eval_mamba_mil.py` | MambaMIL training and evaluation |

The model directories retain their necessary model classes, training utilities, and runtime dataset loaders. These loaders consume prepared inputs; they are not data-preparation pipelines. `MambaMIL/mamba` contains the Mamba runtime used by the server model, not preprocessing code.

## Dependencies

The classical MIL entry points require Python, NumPy, SciPy, scikit-learn, joblib, scikit-survival, and [weighted-dwd](https://github.com/taebinkim7/weighted-dwd). `scikit-survival` and `wdwd` are imported even when running an SVM-only task.

CLAM and MambaMIL additionally require a compatible PyTorch/torchvision environment, pandas, h5py, matplotlib, tqdm, tensorboardX/tensorboard, and model-specific dependencies such as timm and einops. CLAM's optional top-k instance loss requires [smooth-topk](https://github.com/oval-group/smooth-topk); the cross-entropy instance-loss setting does not. Mamba's optional accelerated paths require compatible causal-conv1d/Mamba CUDA extensions. Preserve the bundled runtime for the project's reference-scan behavior instead of substituting a new model implementation.

The source is a snapshot of an existing research environment, not a newly validated dependency lockfile. Full model training was not rerun as part of this upload. CONCH/VGG embeddings and any restricted encoder weights must be obtained separately under their original access terms.

## Use with prepared inputs

The classical entry points read `labels.csv`, `sample_images.csv`, and feature `.npy` files from the supplied feature directory. Feature file names must match the model, layer, instance-size, and instance-stride arguments. Labels and feature identifiers must align. Do not place patient inputs or generated models inside this checkout.

For example, using existing 400-pixel CONCH features and externally prepared labels:

```bash
python run_mi_classify.py \
  --out_dir /path/to/prepared/conch_feat/he/ \
  --model conch_ViT-B-16 --layer end --cat er \
  --instance-size 400 --instance-stride 400 \
  --mi quantile --quantiles 16 --cv-folds 5 --random-state 111

python run_cross_dataset.py \
  --train-dir /path/to/prepared/source/conch_feat/he/ \
  --test-dir /path/to/prepared/target/conch_feat/he/ \
  --model conch_ViT-B-16 --layer end --cat er \
  --train-instance-size 400 --train-instance-stride 400 \
  --test-instance-size 400 --test-instance-stride 400 \
  --mi quantile --quantiles 16 --cv-folds 5 --random-state 111
```

`--mi none` selects the feature-mean baseline. Other pooling variants remain in `sil.py`. For PAM50 endpoints, supply the corresponding column names from your prepared labels. Use each entry point's `--help` in its configured environment for the complete interface. The example commands illustrate the API; they are not a claim that default settings reproduce every manuscript table.

CLAM/MambaMIL consume prepared `output_clam.csv`, feature tensors, and split CSVs. Their training/evaluation options specify input paths, feature dimensions, model selection, and split definitions. Run each entry point in its own model directory so that its local `utils`/`models` imports resolve correctly. The project Mamba trainer uses the legacy task name `LUAD_LUSC` to select its prepared binary-input interface, including for breast endpoints; this identifier does not describe the breast dataset itself. Set `--model_type mamba_mil` for Mamba training and `--model_type mamba` for its evaluation entry point.

## Source and interpretation

Copied from `bioinf.unc.edu` on 2026-10-23. The five top-level Python files come from `/home/ykwang/tma-mil`. CLAM/Mamba training, evaluation, and Python runtime modules come from the isolated `/home/ykwang/san0911_rerun/source` snapshot, which preserves string identifiers when reading prepared split inputs. Original source files on the server were not changed. No training or aggregation algorithms were changed for this upload.

Keep protocols distinct: the classical code supports stratified fivefold CV and repeated holdout, while the project's CLAM/Mamba split generator used five separate splits with 10% validation and 10% test each. Those five splits are **not exhaustive fivefold CV**. For SAN comparisons, use source-only fitting and appropriately matched unadapted/SAN target inputs; this repository does not include the stain-adaptation pipeline or a reconciliation of historical versus rerun numerical results.

## Attribution and licenses

The top-level implementation derives from [Heather Couture's tma-mil](https://github.com/hdcouture/tma-mil); retain the original MIT notice in `LICENSE`. Cite Couture et al., *Image analysis with deep learning to predict breast cancer grade, ER status, histologic subtype, and intrinsic subtype*, npj Breast Cancer, 2018.

[CLAM](https://github.com/mahmoodlab/CLAM) has its own GPL license in `CLAM/LICENSE.md`; the top-level MIT notice does not relicense it. Cite Lu et al., *Data-efficient and weakly supervised computational pathology on whole-slide images*, Nature Biomedical Engineering, 2021.

[MambaMIL](https://github.com/isyangshu/MambaMIL) is attributed to Yang et al., *MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology*, MICCAI, 2024. Its upstream README advertises MIT, but the server snapshot did not contain a separate MambaMIL license file; no replacement copyright notice is invented here. The bundled Mamba runtime retains its own Apache 2.0 license and author notice under `MambaMIL/mamba/`. Review upstream terms before any public redistribution.

CONCH is described by Lu et al., *A visual-language foundation model for computational pathology*, Nature Medicine, 2024; encoder source and weights are not redistributed here.
