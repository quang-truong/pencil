# Plain Transformers are Surprisingly Powerful Link Predictors
Official code for **Plain Transformers are Surprisingly Powerful Link Predictors**.


## Abstract

Link prediction is a core challenge in graph machine learning, demanding models that capture rich and complex topological dependencies. While Graph Neural Networks (GNNs) are the standard solution, state-of-the-art pipelines often rely on explicit structural heuristics or memory-intensive node embeddings---approaches that struggle to generalize or scale to massive graphs. Emerging Graph Transformers (GTs) offer a potential alternative but often incur significant overhead due to complex structural encodings, hindering their applications to large-scale link prediction. We challenge these sophisticated paradigms with PENCIL, an encoder-only plain Transformer that replaces hand-crafted priors with attention over sampled local subgraphs, retaining the scalability and hardware efficiency of standard Transformers. Through experimental and theoretical analysis, we show that PENCIL extracts richer structural signals than GNNs, implicitly generalizing a broad class of heuristics and subgraph-based expressivity. Empirically, PENCIL outperforms heuristic-informed GNNs and is far more parameter-efficient than ID-embedding--based alternatives, while remaining competitive across diverse benchmarks---even without node features. Our results challenge the prevailing reliance on complex engineering techniques, demonstrating that simple design choices are potentially sufficient to achieve the same capabilities. 

## Quick Start

First, clone the repository. Then, check [this link](https://drive.google.com/file/d/1xcrrgvGGaQ5SLuItTrfgQinQQrqpoyUd/view?usp=sharing) to download the data, and execute the following commands:
```bash
unzip data.zip
```

Then, set up the environment by executing the following commands:

```bash
# Set up environment
conda create -n pencil python=3.12
conda activate pencil
pip install -r requirements.txt
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

## Training the Model Yourself

```bash
# Training
python run_with_best_gpus.py args/official/<config_file>.yaml
python run_with_best_gpus.py args/official/<config_file>.yaml --use_features
```

Every run automatically saves the best model checkpoint and the metrics.jsonl file in the `ckpts` directory. You can use the `extract_results.py` script to aggregate the results from multiple runs. For example, to aggregate the results from 5 runs, you can run the following command:

```bash
python extract_results.py --n 5 -p <prefix>
```
