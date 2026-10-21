# enhanced-bertopic-patents
Code for fine-grained technology topic discovery from patents using an enhanced BERTopic framework.
# Enhanced BERTopic for Patent Topic Discovery

This repository contains Python modules for fine-grained technology topic discovery from patent documents. The framework combines patent-specific PaECTER embeddings with a two-stage BERTopic strategy: the main model discovers topics in the full input corpus, and the sub-model explores documents left unassigned by the main model. Keyword enhancement and topic labeling support the interpretation of the resulting topics.

The associated study applies the framework to digital health patents from 2015 to 2024. The resulting topics are candidates for subsequent technology monitoring and assessment of technological emergence; documents being unassigned in the first stage does not, by itself, establish that they represent emerging technologies.

## Source files

| File | Module purpose |
| --- | --- |
| `config.py` | Central configuration: analysis period, embedding model, clustering parameters, and file paths. |
| `data_loader.py` | Patent data loading and preparation. |
| `embedding_generator.py` | Document embedding generation. |
| `topic_modeling.py` | Topic modeling. |
| `keyword_extraction.py` | Keyword extraction and topic representation. |
| `topic_labeling.py` | Topic labeling. |
| `evaluation.py` | Model and topic evaluation. |
| `visualization.py` | Visualization. |
| `result_saving.py` | Result export. |

Module interfaces and any additional requirements are defined in the corresponding source files. Python-generated `__pycache__` directories are not required for distribution.

## Data availability and example records

The full patent dataset was obtained from the Derwent patent database and is not redistributed in this repository because of third-party licensing restrictions. Users wishing to reproduce the full study must obtain access to the source database through an appropriate subscription and follow the data selection and preprocessing procedures described in the manuscript.

A 10-record patent example is provided to illustrate the input layout, column names, and record contents. Use this file as the reference for the expected spreadsheet structure, and preserve its column names when preparing a compatible input file. The example is not the full experimental dataset.

**The 10-record example is a format reference, not a dataset for reproducing the reported clustering results.** The default `min_cluster_size` is 40 for the main model and 20 for the sub-model, both larger than the entire example dataset. The example should therefore not be used to validate topic discovery under the published parameter settings. Changing these settings for a small demonstration would constitute a different experiment.

The configured input filename is `all_patents.xlsx`. Point `DATA_PATH` to the intended input spreadsheet before running the workflow. The stopword file is configured separately through `STOPWORDS_PATH`.

## Configuration

All central settings are defined in `config.py`.

### Analysis period

The default is:

```python
stage = "all"
```

| `stage` value | Start year | End year |
| --- | --- | --- |
| `1` | 2015 | 2017 |
| `2` | 2018 | 2020 |
| `3` | 2021 | 2022 |
| `4` | 2023 | 2024 |
| `"all"` | 2015 | 2024 |

These periods are defined in `stage_year_mapping`. The main controller or notebook must apply the selected period when preparing the analysis input.

### Embedding model

```python
embedding_model_name = "mpi-inno-comp/paecter"
```

Both model configurations reference this embedding model. Access to the model files is required when generating embeddings.

### UMAP parameters

| Parameter | Main model | Sub-model |
| --- | --- | --- |
| `n_neighbors` | 50 | 30 |
| `n_components` | 10 | 10 |
| `min_dist` | 0.07 | 0.07 |
| `metric` | `"cosine"` | `"cosine"` |
| `random_state` | 42 | 42 |

### HDBSCAN parameters

| Parameter | Main model | Sub-model |
| --- | --- | --- |
| `min_cluster_size` | 40 | 20 |
| `min_samples` | 15 | 10 |
| `cluster_selection_epsilon` | 0.1 | 0.1 |
| `cluster_selection_method` | `"leaf"` | `"leaf"` |
| `metric` | `"euclidean"` | `"euclidean"` |
| `prediction_data` | `True` | `True` |

The configurations are collected in `MAIN_MODEL_PARAMS` and `SUB_MODEL_PARAMS`, respectively.

### File paths

The current configuration selects a root directory according to whether `google.colab` is present in `sys.modules`:

| Environment | `DRIVE_PREFIX` |
| --- | --- |
| Google Colab | `/content/drive/My Drive/Colab Notebooks/Bertopic` |
| Local Python environment | `~/local_results/Bertopic` |

The local path is expanded using `os.path.expanduser`. The input and output paths are:

```python
DATA_PATH = f"{DRIVE_PREFIX}/Data/all_patents.xlsx"
SAVE_DIR = f"{DRIVE_PREFIX}/Result"
STOPWORDS_PATH = f"{DRIVE_PREFIX}/Data/US_stopwords.txt"
```

Update these paths to match your own environment. Ensure that the input spreadsheet and stopword file exist, and that the output directory is writable. In Colab, mount Google Drive before accessing the configured Drive paths; `config.py` selects paths but does not mount Drive.

## Environment and execution

The core workflow uses Python, BERTopic, Sentence Transformers, UMAP, and HDBSCAN. Reading the configured Excel input also requires a compatible spreadsheet reader. Install any additional packages imported by the keyword extraction, labeling, evaluation, and visualization modules.

For reproducing the published experiments, use the package versions and model files from the original experimental environment. A fixed UMAP seed alone is not a complete specification of that environment.

Prepare the workflow in the following order:

1. Install the dependencies required by the source modules.
2. Obtain a suitable patent dataset and match its spreadsheet structure to the provided example.
3. Provide the stopword file and update the paths in `config.py`.
4. Select the analysis period and model settings.
5. Execute the main controller or notebook that invokes the pipeline modules: data preparation, embedding generation, main-model topic discovery, sub-model analysis of unassigned documents, keyword enhancement, labeling, evaluation, visualization, and export.

These files form a modular workflow. For end-to-end reproduction, the main controller or notebook that calls these modules must be supplied alongside them.

If the labeling implementation uses an external language-model service, configure its credentials locally according to that module's implementation and do not commit credentials to the repository.

## Reproducibility

Preserve the corpus selection, preprocessing, stopword list, model files, package versions, and configuration used for each analysis. Distinguish results obtained with the published settings from results obtained after modifying parameters or input data. The 10-record example cannot reproduce the topic counts or evaluation values reported in the manuscript.
