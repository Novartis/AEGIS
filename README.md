# AEGIS - Predicting peptide presentation by MHCII using attention models.

## Getting started

Install dependencies using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or similar. We recommend [mamba](https://mamba.readthedocs.io/en/latest/installation.html), a much faster drop-in replacement for conda rewritten in C++. 

To recreate the environment, in a terminal on a computer with a GPU supporting CUDA 11.8, run:
    
```bash
mamba create -n aegis python=3.10
mamba activate aegis
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # Pytorch with CUDA 11.8
pip install tensorboard black gputil isort black joblib jedi mypy networkx pyprojroot rich torchmetrics hydra-core tqdm torchmetrics pydantic python-dotenv biopython pandas matplotlib seaborn scikit-learn tqdm ipython install lightning # Pip dependencies
pip install -e . # Install package in editable mode
```

To train all model variants, we recommend access to and familiarity with a [SLURM](https://slurm.schedmd.com) computing cluster with GPUs. Each model was trained on 1 H100 GPU and the batch size was matched to maximize the utilization of the GPU VRAM (80GB), but doing an inference using the trained models requires much less VRAM, and if you want to retrain the models with fewer ressources, it's possible and would lead to similar results, albeit slower.

## Training

To train all models, we first generate a series of bash scripts which contain all the necessary commands to train the models. This is done by running the following command (make sure to change the SLURM parameters as needed in the `script_cotents` according to your cluster's configuration):

```bash
# Train all models
python experiments/generate_model_variants.py
```

Then, submit all those generated scripts to the cluster using the following command:

```bash
# Submit all scripts to the cluster
python experiments/submit_variants.py
```

If you are only interested in running one variant, or a few, you can run something akin to the following

```bash
# Train a single model
python experiments/train.py dataset.data_source=iedb_nod model.feature_set=seq_mhc seed.seed=0 model.aegis.n_layers=4 hydra.run.dir=./path/to/custom/logs/directory
```
where `dataset.data_source` can be any of `{iedb_nod, nod, iedb}`, `model.feature_set` can be any of `{seq_mhc, seq_only}`, `seed.seed` $\in \mathbb{N}$, `model.aegis.n_layers` $\in 2\mathbb{N}$ and `hydra.run.dir` can be any directory.

## Inference

To run inference on the trained models, a similar procedure can be followed.

```bash
# Run inference on all models
python experiments/generate_inference_scripts.py
```

```bash
# Submit all inference scripts to the cluster
python experiments/submit_inference.py
```

```bash
# Run inference on a single model
python experiments/evaluation/hold_out.py dataset.data_source=iedb_nod model.feature_set=seq_mhc model.aegis.n_layers=8 paths.checkpoint=outputs/variants/seq_mhc-iedb_nod-8-1/checkpoints/last.ckpt hydra.run.dir=./path/to/custom/logs/directory
```

where the script can be any of `{hold_out.py, cd4.py, maria.py, nod.py, strattmann.py, you.py, xu.py}`, `dataset.data_source` can be any of `{iedb_nod, nod, iedb}`, `model.feature_set` can be any of `{seq_mhc, seq_only}`, `model.aegis.n_layers` $\in 2\mathbb{N}$, `paths.checkpoint` is the path to the checkpoint of the model you want to run inference on (make sure it has been trained with the same number of layers as the specified `model.aegis.n_layers` parameter) and `hydra.run.dir` can be any directory where you want to log the results. You can check the generated scripts to see the exact commands that are run to get the paper's results.

Ablation studies can be reproduced similarly using:

```bash
# Run ablation study on all models
python experiments/generate_ablation_scripts.py
```

```bash
# Submit all ablation study scripts to the cluster
python experiments/submit_ablation.py
```

```bash
# Run a single ablation study
python experiments/train.py dataset.data_source=iedb model.feature_set=seq_mhc seed.seed=0 model.aegis.embedding.dummy_embedding=true model.aegis.embedding.all_ones=false hydra.run.dir=./path/to/custom/logs/directory
```
