# Third-party components

BotanicAI's own source code is licensed under the Apache License 2.0.
The components listed below are **not** covered by that license. Model
weights, datasets, photographs, and any generated embeddings or indexes
derived from them remain subject to their original terms. Verify the
current license on each source before redistribution or commercial use.

## Models

### BioCLIP 2

- Role: image encoder for plant identification.
- Source: https://huggingface.co/imageomics/bioclip-2
- License: MIT

### Qwen3.5-2B

- Role: language model for generating source-backed facts (run locally via MLX).
- Source: https://huggingface.co/Qwen/Qwen3.5-2B
- License: Apache License 2.0
- License text: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE

## Datasets

### GBIF Backbone Taxonomy

- Role: species / genus / family taxonomic hierarchy.
- Source: https://www.gbif.org/
- Download: https://hosted-datasets.gbif.org/datasets/backbone/
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Note: confirm the license field on the specific GBIF dataset record you use.

### Pl@ntNet-300K

- Role: reference plant images used to build and evaluate embeddings.
- Source: https://zenodo.org/record/4726653
- Paper: C. Garcin et al., "Pl@ntNet-300K: a plant image dataset with high
  label ambiguity and a long-tailed distribution," NeurIPS 2021
  (Datasets and Benchmarks Track).
- License: **per-image licenses vary.** Images are distributed under a mix
  of CC BY-SA, CC BY-NC, and CC BY-NC-SA licenses; an original CC BY 4.0
  release is also referenced. Check the per-image license metadata on the
  Zenodo record before reuse or redistribution. The dataset code is
  BSD-2-Clause.
- Attribution: cite the NeurIPS 2021 paper when publishing results.

## Text sources

### Wikipedia

- Role: botanical fact text stored in the local knowledge base and quoted,
  with a link back to the source article, next to each generated fact.
- Source: https://www.wikipedia.org/ (via the `wikipedia-api` client)
- License: text is available under Creative Commons Attribution-ShareAlike
  4.0 (CC BY-SA 4.0); some text may additionally be available under the
  GNU Free Documentation License.

## Key libraries

| Library    | Source                                        | License    |
| ---------- | --------------------------------------------- | ---------- |
| Gradio     | https://github.com/gradio-app/gradio          | Apache 2.0 |
| OpenCLIP   | https://github.com/mlfoundations/open_clip    | MIT        |
| FAISS      | https://github.com/facebookresearch/faiss     | MIT        |
| MLX / mlx-lm | https://github.com/ml-explore/mlx           | MIT        |
| PyTorch    | https://github.com/pytorch/pytorch            | BSD-3      |

This list covers the components most relevant to licensing. See each
package's own metadata for the authoritative, current terms.
