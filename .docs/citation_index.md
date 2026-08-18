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

---

## A. Fellegi–Sunter foundations and decision rules

| Key | Used in | Reference | Verify at |
|---|---|---|---|
| `fellegi1969` | whitepaper, paradox | Fellegi & Sunter, "A theory for record linkage", *JASA* 64(328):1183–1210, 1969 | https://doi.org/10.1080/01621459.1969.10501049 ✅ |
| `winkler1993` | whitepaper, paradox, litreview | Winkler, "Improved decision rules in the Fellegi–Sunter model of record linkage", *Proc. ASA Section on Survey Research Methods*, 274–279, 1993 | https://scholar.google.com/scholar?q=%22Improved+decision+rules+in+the+Fellegi-Sunter+model+of+record+linkage%22 ⚠️ |
| `winkler1993` (litreview variant) | litreview only | Winkler, "String comparator metrics and enhanced decision rules in the Fellegi–Sunter model of record linkage", *Proc. ASA Section on Survey Research Methods*, 1993 (companion title, same author/meeting) | https://scholar.google.com/scholar?q=%22String+comparator+metrics+and+enhanced+decision+rules%22 ⚠️ |
| `winkler2000` | whitepaper, paradox | Winkler, "Frequency-based matching in the Fellegi–Sunter model of record linkage", U.S. Census Bureau SDR Research Report RR2000/06, 2000 | https://www.census.gov/library/working-papers/2000/adrm/rr2000-06.html ✅ |
| `winkler2006` | whitepaper | Winkler, "Overview of record linkage and current research directions", U.S. Census Bureau SDR Report RRS2006/02, 2006 | https://www.census.gov/library/working-papers/2006/adrm/rrs2006-02.html ✅ |
| `belin1995` | whitepaper | Belin & Rubin, "A method for calibrating false-match rates in record linkage", *JASA* 90(430):694–707, 1995 | https://doi.org/10.1080/01621459.1995.10476563 ✅ |
| `copashilton1990` | whitepaper | Copas & Hilton, "Record linkage: statistical models for matching computer records", *JRSS A* 153(3):287–320, 1990 | https://doi.org/10.2307/2982975 ✅ |
| `mclachlan2000` | whitepaper, paradox | McLachlan & Peel, *Finite Mixture Models*, Wiley, 2000 | https://doi.org/10.1002/0471721182 ✅ |
| `christen2012` | whitepaper, paradox | Christen, *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*, Springer, 2012 | https://doi.org/10.1007/978-3-642-31164-2 ✅ |
| `davis2006` | whitepaper, paradox | Davis & Goadrich, "The relationship between precision-recall and ROC curves", ICML 2006, 233–240 | https://doi.org/10.1145/1143844.1143874 ✅ |
| `linacre2022` | whitepaper, paradox (key `splink` in that file), batch | Linacre, Lindsay, Manassis, Slade, Hepworth, Kennedy, Bond, "Splink: Free software for probabilistic record linkage at scale", *Int. J. Popul. Data Sci.* 7(3):1794, 2022 | https://doi.org/10.23889/ijpds.v7i3.1794 ✅ • code: https://github.com/moj-analytical-services/splink |
| `brier` | paradox | Brier, "Verification of forecasts expressed in terms of probability", *Monthly Weather Review* 78(1):1–3, 1950 | https://doi.org/10.1175/1520-0493(1950)078%3C0001%3AVOFEIT%3E2.0.CO%3B2 ✅ |

## B. Blocking, learned matchers, end-to-end resolution

| Key | Used in | Reference | Verify at |
|---|---|---|---|
| `mccallum2000` | whitepaper, batch | McCallum, Nigam, Ungar, "Efficient clustering of high-dimensional data sets with application to reference matching", KDD 2000, 169–178 | https://doi.org/10.1145/347090.347123 ✅ |
| `thirumuruganathan2021` | whitepaper, batch | Thirumuruganathan, Li, Tang, Ouzzani, Govind, Paulsen, Fung, Doan, "Deep learning for blocking in entity matching", *PVLDB* 14(11):2459–2472, 2021 | https://doi.org/10.14778/3476249.3476294 ✅ |
| `zhang2020autoblock` | whitepaper, batch | Zhang, Wei, Sisman, Dong, Faloutsos, Page, "AutoBlock: a hands-off blocking framework for entity matching", WSDM 2020, 744–752 | https://doi.org/10.1145/3336191.3371813 ✅ |
| `ebreheem2018` | whitepaper, batch | Ebraheem, Thirumuruganathan, Joty, Ouzzani, Tang, "Distributed representations of tuples for entity resolution", *PVLDB* 11(11):1454–1467, 2018 | https://doi.org/10.14778/3236187.3236198 ✅ |
| `zeakis2023` | whitepaper, batch | Zeakis, Papadakis, Skoutas, Koubarakis, "Pre-trained embeddings for entity resolution: an experimental analysis", *PVLDB* 16(9):2225–2238, 2023 | https://doi.org/10.14778/3598581.3598594 ✅ |
| `wang2024` | whitepaper | Wang, Lin, Han, Chen, Cao, Sun, "Towards universal dense blocking for entity resolution", arXiv:2404.14831, 2024 | https://arxiv.org/abs/2404.14831 ✅ • code: https://github.com/tshu-w/uniblocker |
| `strojny2026` | whitepaper | Strojny & Beręsewicz, "BlockingPy: approximate nearest neighbours for blocking of records for entity resolution", *SoftwareX* 34:102583, 2026 | https://doi.org/10.1016/j.softx.2026.102583 ✅ • code: https://github.com/ncn-foreigners/BlockingPy |
| `mudgal2018` | whitepaper | Mudgal, Li, Rekatsinas, Doan, Park, Krishnan, Deep, Arcaute, Raghavendra, "Deep learning for entity matching: a design space exploration", SIGMOD 2018, 19–34 | https://doi.org/10.1145/3183713.3196926 ✅ |
| `li2020ditto` | whitepaper | Li, Li, Suhara, Doan, Tan, "Deep entity matching with pre-trained language models", *PVLDB* 14(1):50–60, 2020 | https://doi.org/10.14778/3421424.3421431 ✅ |
| `peeters2023` | whitepaper | Peeters, Steiner, Bizer, "Entity matching using large language models", arXiv:2310.11244, 2023 | https://arxiv.org/abs/2310.11244 ✅ |
| `christophides2021` | whitepaper | Christophides, Efthymiou, Palpanas, Papadakis, Stefanidis, "An overview of end-to-end entity resolution for big data", *ACM Computing Surveys* 53(6):127, 2021 | https://doi.org/10.1145/3418896 ✅ |
| `sadinle2013` | whitepaper | Sadinle & Fienberg, "A generalized Fellegi–Sunter framework for multiple record linkage with application to homicide record systems", *JASA* 108(502):651–660, 2013 | https://arxiv.org/abs/1205.3217 ✅ |
| `kopcke2010` | whitepaper | Köpcke, Thor, Rahm, "Evaluation of entity resolution approaches on real-world match problems", *PVLDB* 3(1–2):484–493, 2010 | https://doi.org/10.14778/1920841.1920904 ✅ |
| `christen2004` | whitepaper | Christen, Churches, Hegland, "Febrl – a parallel open source data linkage system", PAKDD 2004, LNCS 3056, 638–647 | https://doi.org/10.1007/978-3-540-24775-3_75 ✅ |

## C. Embeddings, similarity search, tabular representations

| Key | Used in | Reference | Verify at |
|---|---|---|---|
| `johnson2019` | whitepaper | Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs", *IEEE Trans. Big Data* 7(3):535–547, 2021 | https://doi.org/10.1109/TBDATA.2019.2921572 ✅ • preprint: https://arxiv.org/abs/1702.08734 ✅ |
| `franz2025` | whitepaper | Franz, Hoppe, Michaelis, Göbel, "Universal embeddings of tabular data", arXiv:2507.05904, 2025 | https://arxiv.org/abs/2507.05904 ✅ |
| `vogel2026` | whitepaper | Vogel, Srinivas, D'Souza, Shirai, Hassanzadeh, Samulowitz, "Towards universal tabular embeddings: A benchmark across data tasks", arXiv:2604.21696, 2026 | https://arxiv.org/abs/2604.21696 ✅ |

## D. Temporal record linkage

| Key | Used in | Reference | Verify at |
|---|---|---|---|
| `lidong2011` | whitepaper, litreview (keys `lidong2011` / `li2011`) | Li, Dong, Maurino, Srivastava, "Linking temporal records", *PVLDB* 4(11):956–967, 2011 | https://doi.org/10.14778/3402707.3402733 ✅ |
| `christengayler2013` | whitepaper, litreview (key `cg2013`) | Christen, Gayler, "Adaptive temporal entity resolution on dynamic databases", PAKDD 2013, 558–569 | https://doi.org/10.1007/978-3-642-37456-2_47 ✅ |
| `hu2017regression` | whitepaper, litreview (key `hu2017`) | Hu, Wang, Vatsalan, Christen, "Improving temporal record linkage using regression classification", PAKDD 2017, 561–573 | https://doi.org/10.1007/978-3-319-57454-7_44 ✅ |
| `li2015transition` | whitepaper, litreview (key `ll15`) | Li, Lee, Hsu, Tan, "Linking temporal records for profiling entities", SIGMOD 2015, 593–605 | https://doi.org/10.1145/2723372.2737789 ✅ |
| `chiang2014` | whitepaper, litreview (key `cd14`) | Chiang, Doan, Naughton, "Modeling entity evolution for temporal record matching", SIGMOD 2014, 1175–1186 | https://doi.org/10.1145/2588555.2588560 ✅ |
| `ranbaduge2020` | whitepaper, litreview (keys `ranbaduge2020` / `rc2020`) | Ranbaduge, Christen, "A scalable privacy-preserving framework for temporal record linkage", *Knowledge and Inf. Sys.* 62(1):45–78, 2020 | https://doi.org/10.1007/s10115-019-01370-1 ✅ |
| `rc2018` | litreview only | Ranbaduge, Christen, "Privacy-preserving temporal record linkage", IEEE ICDM 2018, 377–386 | https://doi.org/10.1109/ICDM.2018.00053 ✅ |
| `litoux2026` | whitepaper, litreview (keys `litoux2026` / `lr2026`) | Litoux, Ray, "Temporal record linkage using time decay models applied to vessel data", IEEE MDM 2026, 313–318 | https://doi.org/10.1109/MDM71479.2026.00044 ✅ |
| `shim2026` | whitepaper, litreview | Shim, "TimeLink: time-aware record linkage with semi-Markov dynamics", SSRN preprint 6468518, 2026 | https://doi.org/10.2139/ssrn.6468518 ✅ |

## E. Data sources and external resources

| Key | Verify at |
|---|---|
| `ncvoter` | https://www.ncsbe.gov/results-data/voter-registration-data ✅ |
| `kopcke2010` benchmark data | https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution/ ✅ |
| `mccallum2000` Cora data | https://hpi.de/naumann/projects/repeatability/datasets/cora-dataset.html ✅ |
| `christen2004` FEBRL data | https://sourceforge.net/projects/febrl/ ✅ |

## F. Author's own technical reports (verifiable in this repository)

| Key | Used in | Location |
|---|---|---|
| `pipeline` | whitepaper, paradox, batch | `.docs/entity_resolution_whitepaper.tex` |
| `paradox` | whitepaper, batch | `.docs/calibration_paradox.tex` |
| `vectordedupbatch` | whitepaper | `.docs/vector_dedup_batch.tex` |
| `decaynote` | whitepaper | “Why the decaying address weight cannot be absorbed into m/u” (technical note, 2026) |
| `decaylitreview` | whitepaper | “Time decay in Fellegi–Sunter record linkage: a literature survey” (technical note, 2026) — source file: `.docs/decay-in-fs-litreview.tex` |

---

## A note on entries that were removed on 2026-08-17

| Removed entry | Reason | Replaced by |
|---|---|---|
| `das2020deepblocker` — “The case for blocking…” | Fabricated. arXiv 2004.02588 is not this paper (it is a Navier–Stokes paper). | `thirumuruganathan2021` (real DeepBlocker paper) |
| `efthymiou2020` — “Entity matching meets record linkage…” | Fabricated. No such work in DBLP/Crossref. | `christophides2021` (real survey including Efthymiou as a genuine co-author) |
| `sadinle2014` — “…continuous covariates…” | Fabricated. Not in Sadinle’s record; its arXiv id 1404.0969 is a cyclic-codes paper. | `sadinle2013` (real JASA paper, arXiv:1205.3217) |
| `wang2024` author list | Garbled (all six author names incorrect). | Corrected to arXiv:2404.14831 author list |

See the corresponding `thebibliography` blocks in the four `.tex` files for the current,
canonical entries.
<!--
Do not edit DOIs in this file without re-checking them against Crossref/arXiv/DBLP.
-->