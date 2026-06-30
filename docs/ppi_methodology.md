# PPI Network Embeddings: A Third Modality for IRD Gene Classification

## Why protein-protein interaction data

The classifier's first two towers - normalized phylogenetic profiling (NPP) and ESM2 protein language model embeddings - capture two distinct but related signals: cross-species co-conservation patterns, and sequence/structure-derived representations. Both are, in a sense, properties of a gene considered in isolation. Neither directly encodes whether two genes' protein products physically interact, or sit in the same functional neighborhood of the cell.

Protein-protein interaction (PPI) networks fill that gap. Genes belonging to the same inherited retinal dystrophy (IRD) phenotypic module often do so because their products work together in a shared complex or pathway - the ciliary transport machinery being a clear example. That kind of relationship is structurally invisible to a model that only ever sees one gene at a time. Adding a PPI-derived tower was therefore the natural extension once the available variation in NPP/ESM2 architecture and training configuration had been explored without a comparable performance jump.

## Building the network

The interactome is built from STRING v12.0 (human, taxon 9606), restricted to interactions with a combined confidence score of 400 or higher - STRING's standard medium-confidence cutoff, chosen to exclude low-confidence predicted associations while retaining well-supported physical and functional links. After deduplication and identifier harmonization against the project's master gene list (HGNC symbols, with alias-table rescue for legacy or alternate naming), the resulting network contains on the order of 880,000 weighted edges connecting roughly 93% of the gene set directly.

The remaining genes are absent from the network either because STRING has no data for them, or because all of their candidate edges fall below the confidence threshold. These genes do not simply get dropped: each receives an imputed embedding (the mean of all covered genes' vectors) plus an explicit binary coverage flag. The flag is the important part - it gives the model a way to recognize "this is not a real network signal" and down-weight it accordingly during training, rather than silently treating an imputed placeholder as genuine topology.

## From graph to vector: node2vec

Per-gene embeddings are generated with node2vec, using biased random walks over the weighted STRING graph (edge weight = confidence score, walk parameters p = q = 1 as an unbiased baseline) and a skip-gram objective, producing 128-dimensional vectors per gene. Combined with the coverage flag, each gene enters the model as a 129-dimensional input.

This baseline parameterization treats the graph symmetrically - walks are not biased toward staying within tight local neighborhoods (which would emphasize community/cluster structure) or toward exploring broadly (which would emphasize global role similarity). Whether a more targeted walk strategy - biased toward local community structure, for instance - better separates functionally tight clusters like the ciliary genes remains an open question for future iterations; the current results, discussed below, suggest the unbiased baseline already captures a substantial and useful amount of structure.

## Tower architecture

The PPI tower is a single linear projection - `Linear(129 → 64) → ReLU → Dropout(0.3)` - intentionally matched in output dimensionality to the NPP tower, so that no single modality dominates the fused representation by sheer width.

## Does the embedding space actually mean anything biologically?

Before trusting a 128-dimensional vector as a meaningful representation of a gene's network context, it's worth checking it against biology we already understand. A small cosine-similarity spot-check among five well-characterized IRD genes provides exactly that check, independent of anything the classifier itself was trained to predict.

Three of the five genes - CEP290, RPGR, and IFT88 - are core components of ciliary transport. The other two - ABCA4 and RHO - are central, non-ciliary photoreceptor genes. In the learned embedding space, the three ciliary genes cluster together with substantially higher mutual similarity than either has to the two photoreceptor genes, which in turn cluster together with each other. Cross-group similarity is consistently the lowest of the three groupings. None of this was supervised; it falls directly out of network topology, and it lines up with established biology.

A more specific detail makes the case more interesting than a clean cluster diagram would: RPGR is *more* similar to ABCA4 than it is to IFT88, its fellow ciliary gene. This is not noise - RPGR has a well-documented dual role, contributing to ciliary structure while also being required for photoreceptor outer-segment maintenance. In network terms, it functions as a hub bridging two otherwise distinct functional neighborhoods. That kind of cross-module bridging is, almost by construction, something evolutionary conservation profiles and sequence-based embeddings cannot see - it only exists in interaction topology. It is the clearest concrete illustration of what this modality contributes that the other two structurally cannot.

## Contribution to overall model performance

Across a full ablation comparing every combination of the three modalities (seven variants total - see the training methodology document for the complete breakdown), every configuration that included the PPI tower outperformed every configuration without it by a wide margin, well beyond what could be explained by ordinary run-to-run variance. The non-PPI two-tower baseline (NPP + ESM2) plateaued at a validation macro-F1 of roughly 0.29 ± 0.05, a ceiling that held stable across nine separate architecture and training variations explored before this modality was introduced. Adding PPI broke that ceiling decisively: the full three-tower fusion model reached 0.40 ± 0.02, and - more strikingly - PPI features alone, with no NPP or ESM2 input at all, reached 0.44 ± 0.05, the highest result obtained by any single configuration in the project. With 17 target classes and a near-uniform random baseline around 0.06, a macro-F1 of 0.44 from only 434 labeled training genes represents roughly a 7.5-fold improvement over chance.

The PPI signal alone was, on its own, the single strongest predictor of IRD module membership among all three modalities tested - a result that was not anticipated at the outset of the project and motivates treating network topology as a first-class feature for this problem, not a secondary enrichment. That the PPI-only model edges out the full three-tower fusion is itself a finding worth dwelling on, addressed further in the open questions below.

## Open questions

- Whether biasing node2vec's random walks toward local community structure further improves separation of tightly-knit functional modules (the ciliary genes among them) compared to the unbiased baseline used here.
- Why the PPI-only model outperforms the full three-tower fusion on this dataset, and what that implies for genome-wide candidates versus the well-characterized training genes - addressed in detail in the training methodology document, since it bears directly on how PPI-only predictions should be interpreted, not only on fusion architecture.
