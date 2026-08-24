# Citation Verification Index

Every bibliography entry used in the papers under `.docs/`. Each external entry was checked
against Crossref / DBLP / arXiv / the publisher's record on **2026-08-17**; the identifier in
**Verify at** is the one that matched. DOI/arXiv links open in any browser (SSRN and IEEE pages
may briefly error on automated clients, but open normally in a browser).

Covered documents:

- **whitepaper** — `.docs/entity_resolution_whitepaper.tex`
- **paradox** — `.docs/calibration_paradox.tex`
- **batch** — `.docs/vector_dedup_batch.tex`
- **decay-litreview** — `.docs/decay-in-fs-litreview.tex` (the technical survey cited as `decaylitreview`)

Status: ✅ = authoritative record matched; ⚠️ = no DOI exists (search link provided);
📄 = the author's own technical report, verifiable in this repository.

Where a paper copy exists locally it is linked from the row as a `local:` link pointing
into `source_papers/` (index lives in `.docs/`, mirrors in `source_papers/`). Works with
available copies are flagged with the local link in their row.

---

## A. Fellegi–Sunter foundations and decision rules

| Key | Used in | Reference | Verify at |
|---|---|---|---|
| `fellegi1969` | whitepaper, paradox | Fellegi & Sunter, "A theory for record linkage", *JASA* 64(328):1183–1210, 1969 | https://doi.org/10.1080/01621459.1969.10501049 ✅ · local: [Fellegi-Sunter.pdf](../source_papers/Fellegi-Sunter.pdf) |
| `winkler1993` | whitepaper, paradox, litreview | Winkler, "Improved decision rules in the Fellegi–Sunter model of record linkage", *Proc. ASA Section on Survey Research Methods*, 274–279, 1993 | https://scholar.google.com/scholar?q=%22Improved+decision+rules+in+the+Fellegi-Sunter+model+of+record+linkage%22 ⚠️ · local: [Winkler - ImprovedDecisionRules.pdf](../source_papers/Winkler%20-%20ImprovedDecisionRules.pdf) |
| `winkler1993` (litreview variant) | litreview only | Winkler, "String comparator metrics and enhanced decision rules in the Fellegi–Sunter model of record linkage", *Proc. ASA Section on Survey Research Methods*, 1993 (companion title, same author/meeting) | https://scholar.google.com/scholar?q=%22String+comparator+metrics+and+enhanced+decision+rules%22 ⚠️ · local: [Winkler - StringComparatorMetrics.pdf](../source_papers/Winkler%20-%20StringComparatorMetrics.pdf) |
| `winkler2000` | whitepaper, paradox | Winkler, "Frequency-based matching in the Fellegi–Sunter model of record linkage", U.S. Census Bureau SDR Research Report RR2000/06, 2000 | https://www.census.gov/library/working-papers/2000/adrm/rr2000-06.html ✅ · local: [Winkler - FrequencyBasedMatching.pdf](../source_papers/Winkler%20-%20FrequencyBasedMatching.pdf) |
| `winkler2006` | whitepaper | Winkler, "Overview of record linkage and current research directions", U.S. Census Bureau SDR Report RRS2006/02, 2006 | https://www.census.gov/library/working-papers/2006/adrm/rrs2006-02.html ✅ · local: [OverviewOfRecordLinkage.pdf](../source_papers/OverviewOfRecordLinkage.pdf) · [Winkler - OverviewOfRecordLinkage.pdf](../source_papers/Winkler%20-%20OverviewOfRecordLinkage.pdf) |
| `christen2012` | whitepaper, paradox | Christen, *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*, Springer, 2012 | https://doi.org/10.1007/978-3-642-31164-2 ✅ — physical copy held by author |
| `davis2006` | whitepaper, paradox | Davis & Goadrich, "The relationship between precision-recall and ROC curves", ICML 2006, 233–240 | https://doi.org/10.1145/1143844.1143874 ✅ · local: [PrecisionRecallAndROCCurves.pdf](../source_papers/PrecisionRecallAndROCCurves.pdf) |
| `linacre2022` | whitepaper, paradox (key `splink` in that file), batch | Linacre, Lindsay, Manassis, Slade, Hepworth, Kennedy, Bond, "Splink: Free software for probabilistic record linkage at scale", *Int. J. Popul. Data Sci.* 7(3):1794, 2022 | https://doi.org/10.23889/ijpds.v7i3.1794 ✅ • code: https://github.com/moj-analytical-services/splink · local: [Splink-IntJPopulDataSci.pdf](../source_papers/Splink-IntJPopulDataSci.pdf) (open access PDF) |
| `brier` | paradox | Brier, "Verification of forecasts expressed in terms of probability", *Monthly Weather Review* 78(1):1–3, 1950 | https://doi.org/10.1175/1520-0493(1950)078%3C0001%3AVOFEIT%3E2.0.CO%3B2 ✅ · local: [Brier - VerificationOfForecasts.pdf](../source_papers/Brier%20-%20VerificationOfForecasts.pdf) |

## B. Blocking, learned matchers, end-to-end resolution

| Key | Used in | Reference | Verify at |
|---|---|---|---|
| `mccallum2000` | whitepaper, batch | McCallum, Nigam, Ungar, "Efficient clustering of high-dimensional data sets with application to reference matching", KDD 2000, 169–178 | https://doi.org/10.1145/347090.347123 ✅ · local: [EfficientClusteringOfHighDimensionalDataSets.pdf](../source_papers/EfficientClusteringOfHighDimensionalDataSets.pdf) · [CanopyClustering.pdf](../source_papers/CanopyClustering.pdf) |
| `thirumuruganathan2021` | whitepaper, batch | Thirumuruganathan, Li, Tang, Ouzzani, Govind, Paulsen, Fung, Doan, "Deep learning for blocking in entity matching", *PVLDB* 14(11):2459–2472, 2021 | https://doi.org/10.14778/3476249.3476294 ✅ · local: [DeepLearningForBlockingInEntityMatching.pdf](../source_papers/DeepLearningForBlockingInEntityMatching.pdf) |
| `zhang2020autoblock` | whitepaper, batch | Zhang, Wei, Sisman, Dong, Faloutsos, Page, "AutoBlock: a hands-off blocking framework for entity matching", WSDM 2020, 744–752 | https://doi.org/10.1145/3336191.3371813 ✅ · local: [AutoBlock.pdf](../source_papers/AutoBlock.pdf) |
| `ebreheem2018` | whitepaper, batch | Ebraheem, Thirumuruganathan, Joty, Ouzzani, Tang, "Distributed representations of tuples for entity resolution", *PVLDB* 11(11):1454–1467, 2018 | https://doi.org/10.14778/3236187.3236198 ✅ · local: [DeepER-DistributedRepresentationsOfTuples.pdf](../source_papers/DeepER-DistributedRepresentationsOfTuples.pdf) (arXiv version) |
| `zeakis2023` | whitepaper, batch | Zeakis, Papadakis, Skoutas, Koubarakis, "Pre-trained embeddings for entity resolution: an experimental analysis", *PVLDB* 16(9):2225–2238, 2023 | https://doi.org/10.14778/3598581.3598594 ✅ · local: [PreTrainedEmbeddingsForEntityResolution.pdf](../source_papers/PreTrainedEmbeddingsForEntityResolution.pdf) (author copy, Zenodo) |
| `wang2024` | whitepaper | Wang, Lin, Han, Chen, Cao, Sun, "Towards universal dense blocking for entity resolution", arXiv:2404.14831, 2024 | https://arxiv.org/abs/2404.14831 ✅ · code: https://github.com/tshu-w/uniblocker · local: [UniBlocker.pdf](../source_papers/UniBlock%20%5BREAD%20THIS%5D.pdf) |
| `mudgal2018` | whitepaper | Mudgal, Li, Rekatsinas, Doan, Park, Krishnan, Deep, Arcaute, Raghavendra, "Deep learning for entity matching: a design space exploration", SIGMOD 2018, 19–34 | https://doi.org/10.1145/3183713.3196926 ✅ · local: [DeepLearningForEntityMatching.pdf](../source_papers/DeepLearningForEntityMatching.pdf) |
| `li2020ditto` | whitepaper | Li, Li, Suhara, Doan, Tan, "Deep entity matching with pre-trained language models", *PVLDB* 14(1):50–60, 2020 | https://doi.org/10.14778/3421424.3421431 ✅ · local: [Ditto-DeepEntityMatching.pdf](../source_papers/Ditto-DeepEntityMatching.pdf) (arXiv version) |
| `peeters2023` | whitepaper | Peeters, Steiner, Bizer, "Entity matching using large language models", arXiv:2310.11244, 2023 | https://arxiv.org/abs/2310.11244 ✅ · local: [EntityMatchingUsingLLMs.pdf](../source_papers/EntityMatchingUsingLLMs.pdf) |
| `christophides2021` | whitepaper | Christophides, Efthymiou, Palpanas, Papadakis, Stefanidis, "An overview of end-to-end entity resolution for big data", *ACM Computing Surveys* 53(6):127, 2021 | https://doi.org/10.1145/3418896 ✅ · local: [OverviewOfEndToEndERForBigData.pdf](../source_papers/OverviewOfEndToEndERForBigData.pdf) |
| `sadinle2013` | whitepaper | Sadinle & Fienberg, "A generalized Fellegi–Sunter framework for multiple record linkage with application to homicide record systems", *JASA* 108(502):651–660, 2013 | https://arxiv.org/abs/1205.3217 ✅ · local: [GeneralizedFellegiSunterFramework.pdf](../source_papers/GeneralizedFellegiSunterFramework.pdf) |

## C. Embeddings, similarity search, tabular representations

| Key | Used in | Reference | Verify at |
|---|---|---|---|
| `johnson2019` | whitepaper | Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs", *IEEE Trans. Big Data* 7(3):535–547, 2021 | https://doi.org/10.1109/TBDATA.2019.2921572 ✅ • preprint: https://arxiv.org/abs/1702.08734 ✅ · local: [BillionScaleSimilaritySearch.pdf](../source_papers/BillionScaleSimilaritySearch.pdf) |
| `franz2025` | whitepaper | Franz, Hoppe, Michaelis, Göbel, "Universal embeddings of tabular data", arXiv:2507.05904, 2025 | https://arxiv.org/abs/2507.05904 ✅ · local: [UniversalEmbeddingsOfTabularData.pdf](../source_papers/UniversalEmbeddingsOfTabularData.pdf) |
| `vogel2026` | whitepaper | Vogel, Srinivas, D'Souza, Shirai, Hassanzadeh, Samulowitz, "Towards universal tabular embeddings: A benchmark across data tasks", arXiv:2604.21696, 2026 | https://arxiv.org/abs/2604.21696 ✅ · local: [TowardsUniversalTabularEmbeddings.pdf](../source_papers/TowardsUniversalTabularEmbeddings.pdf) |

## D. Temporal record linkage

| Key | Used in | Reference | Verify at |
|---|---|---|---|
| `lidong2011` | litreview | Li, Dong, Maurino, Srivastava, "Linking temporal records", *PVLDB* 4(11):956–967, 2011 | https://doi.org/10.14778/3402707.3402733 ✅ · local: [LinkingTemporalRecords.pdf](../source_papers/LinkingTemporalRecords.pdf) |
| `christengayler2013` | litreview (key `cg2013`) | Christen, Gayler, "Adaptive temporal entity resolution on dynamic databases", PAKDD 2013, 558–569 | https://doi.org/10.1007/978-3-642-37456-2_47 ✅ · local: [AdaptiveTemporalEntityResolutionOnDynamicDatabases.pdf](../source_papers/AdaptiveTemporalEntityResolutionOnDynamicDatabases.pdf) (author copy, ANU repo) |
| `hu2017regression` | litreview (key `hu2017`) | Hu, Wang, Vatsalan, Christen, "Improving temporal record linkage using regression classification", PAKDD 2017, 561–573 | https://doi.org/10.1007/978-3-319-57454-7_44 ✅ · local: [ImprovingTemporalRecordLinkageUsingRegressionClassification.pdf](../source_papers/ImprovingTemporalRecordLinkageUsingRegressionClassification.pdf) (author copy, ANU repo) |
| `chiang2014` | litreview (key `cd14`) | Chiang, Doan, Naughton, "Modeling entity evolution for temporal record matching", SIGMOD 2014, 1175–1186 | https://doi.org/10.1145/2588555.2588560 ✅ · local: [ModelingEntityEvolution.pdf](../source_papers/ModelingEntityEvolution.pdf) |
| `shim2026` | litreview | Shim, "TimeLink: time-aware record linkage with semi-Markov dynamics", SSRN preprint 6468518, 2026 | https://doi.org/10.2139/ssrn.6468518 ✅ · local: [TimeLink.pdf](../source_papers/TimeLink.pdf) |
| `linacrepc2026` | whitepaper | Linacre, "Comment on Splink issue #3240 concerning piecewise temporal decay via comparison levels", GitHub, moj-analytical-services/splink, 2026 | https://github.com/moj-analytical-services/splink/issues/3240#issuecomment-5309218620 ✅ |

## E. Data sources and external resources

| Key | Verify at |
|---|---|
| `ncvoter` | https://www.ncsbe.gov/results-data/voter-registration-data ✅ |
| ER benchmarks (Leipzig) | https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution/ ✅ |
| `mccallum2000` Cora data | https://hpi.de/naumann/projects/repeatability/datasets/cora-dataset.html ✅ |
| FEBRL data distribution | https://sourceforge.net/projects/febrl/ ✅ |

## F. Author's own technical reports (verifiable in this repository)

| Key | Used in | Location |
|---|---|---|
| `pipeline` | whitepaper, paradox, batch | `.docs/entity_resolution_whitepaper.tex` |
| `paradox` | whitepaper, batch | `.docs/calibration_paradox.tex` |
| `vectordedupbatch` | whitepaper | `.docs/vector_dedup_batch.tex` |
| `decaynote` | whitepaper | “Why the decaying address weight cannot be absorbed into m/u” (technical note, 2026) — source file: `.docs/design_note_mu_cannot_capture_decay.md`|
| `decaylitreview` | whitepaper | “Time decay in Fellegi–Sunter record linkage: a literature survey” (technical note, 2026) — source file: `.docs/decay-in-fs-litreview.tex` |



