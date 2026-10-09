# Evidence for datasets, protocols and reference numbers (ads)

Every dataset, split, hyperparameter and target number used in `examples/ads`
is taken from one of three kinds of source: (1) the original paper, (2) the
authors' official code, (3) an authoritative framework or benchmark
implementing the model.  Numbers below were read from the linked pages;
"verified" marks facts re-checked on the downloaded data or code.

## Datasets

| Dataset | Used by | Source | Verification |
| --- | --- | --- | --- |
| AmazonElectronics_x1 | DIN | reczoo/Datasets `Amazon/AmazonElectronics_x1` (DIN authors' Electronics protocol: leave-last-out, one random negative per positive); download https://huggingface.co/datasets/reczoo/AmazonElectronics_x1 | verified: md5 train e9bf80b9..., test 57a20e82... as published; train 2,608,764 / test 384,806 rows; vocab item 63,003 / cate 803 (= BARS log) |
| Criteo_x1 | DeepFM, DCN-Mix | reczoo/Datasets `Criteo/Criteo_x1` (random 7:2:1 split following AFN, AAAI 2020); download https://huggingface.co/datasets/reczoo/Criteo_x1 | verified: md5 train 30b89c1c..., valid f73c71fb..., test 2c48b26e... as published; 33,003,326 / 8,250,124 / 4,587,167 rows |

The raw DIN files (`reviews_Electronics_5.json.gz`, `meta_Electronics.json.gz`)
are those downloaded by the DIN authors' `utils/0_download_raw.sh`
(https://github.com/zhougr1993/DeepInterestNetwork).  AmazonElectronics_x1 is
the output of the authors' `din/build_dataset.py`: 2,608,764 = 2 x
(1,689,188 - 2 x 192,403) and 384,806 = 2 x 192,403.

## DIN

| Kind | Source | What it gives |
| --- | --- | --- |
| Paper | Zhou et al., KDD 2018, arXiv 1706.06978 | Amazon(Electro) 192,403 users / 63,001 goods / 801 categories / 1,689,188 samples (Table 2); split: train k = 1..n-2, test predicts the last item; metric is the user-weighted AUC of Eq. 10 ("we still refer it as AUC"); Table 3: DIN 0.8818, DIN with Dice 0.8871 |
| Official code | github.com/zhougr1993/DeepInterestNetwork | category = `categories[-1][-1]`; one negative per position (seed 1234); SGD lr 1.0 -> 0.1, batch 32; README GAUC: DIN 0.8698, DIN with Dice 0.8711 |
| Framework / benchmark | BARS `ranking/ctr/DIN/DIN_amazonelectronics_x1` with FuxiCTR v2.0.1 `model_zoo/DIN` | full config (embedding 64, attention [512, 256] Dice + dropout 0.1, softmax attention, DNN [1024, 512, 256] Dice + BN + dropout 0.1, Adam 5e-4, batch 1024, embedding_regularizer 5e-3, max_len 100, patience 2); **test gAUC 0.883526, AUC 0.886028, logloss 0.430190**; 5,403,522 parameters |

Reproduction target: BARS (gAUC 0.8835 / AUC 0.8860).  The implementation
matches the FuxiCTR code line by line and has exactly 5,403,522 parameters.
Caveat carried over from BARS: its config sets `valid_data` to test.csv, so
early stopping and model selection see the test split (`valid_split: test`).

## DeepFM

| Kind | Source | What it gives |
| --- | --- | --- |
| Paper | Guo et al., IJCAI 2017 (https://www.ijcai.org/proceedings/2017/0239.pdf) | Criteo 45M records, random 90/10 split without published seed; Table 2 DeepFM AUC 0.8016 / logloss 0.44985 (arXiv v1: 0.8007 / 0.45083) |
| Official code | none published by the authors | - |
| Framework / benchmark | BARS `ranking/ctr/DeepFM/DeepFM_criteo_x1` with FuxiCTR v1.1.0 `DeepFM` | embedding 10, DNN [400, 400, 400] ReLU + BN + dropout 0.1, Adam 1e-3 with reduce-LR-on-plateau, batch 4096, embedding_regularizer 1e-5; **test AUC 0.813666, logloss 0.438147** (5-run mean AUC 0.813812); 23,429,477 parameters |
| Framework | TorchRec `torchrec/modules/deepfm.py` (`DeepFM`, `FactorizationMachine`) | module implementation, no benchmark number |

Reproduction target: BARS (AUC 0.8137 / logloss 0.4381).  The paper's split
cannot be recreated, so its numbers are not used as a target.

## DCN-Mix (DCN-V2 with mixture of low-rank experts)

| Kind | Source | What it gives |
| --- | --- | --- |
| Paper | Wang et al., WWW 2021, arXiv 2008.13535 | Criteo: days 1-6 train, day 7 split into valid/test (not published); Table 6: DCN-Mix AUC 0.8112 / logloss 0.4408, DCN-V2 0.8115 / 0.4406 |
| Official code | TensorFlow Recommenders `tfrs.layers.dcn.Cross(projection_dim=...)` | low-rank cross layer; the mixture of experts is not in TFRS core |
| Framework | TorchRec `torchrec/modules/crossnet.py` `LowRankMixtureCrossNet` | reference implementation of the mixture cross network |
| Framework / benchmark | FuxiCTR v1.1.0 `DCNv2` with `use_low_rank_mixture=True` (`CrossNetMix`); BARS `ranking/ctr/DCNv2/DCNv2_criteo_x1` | BARS config (embedding 10, 3 cross layers, low_rank 32, num_experts 4, parallel DNN [500, 500, 500], dropout 0.1, Adam 1e-3, batch 4096, embedding_regularizer 1e-5) published with `use_low_rank_mixture: False`: **test AUC 0.814195, logloss 0.437788** |

No benchmark publishes DCN-Mix on a downloadable split.  The run uses the BARS
DCNv2_criteo_x1 configuration with the mixture switched on.  The reference is
the BARS full-rank DCN-V2 number together with the paper's finding that the
mixture costs about 0.0003 AUC (0.8112 vs 0.8115).

## Training procedure (all three)

FuxiCTR `BaseModel` (v1.1.0 `fit_generator` / v2.0.1 `fit`): Adam; loss = BCE
+ embedding_regularizer / 2 * ||W_emb||^2 over parameters under
`embedding_layer`; gradient norm clipped to 10; embeddings N(0, 1e-4), Linear
Xavier-normal; after each epoch the monitored validation metric is checked,
a non-improvement (< 1e-6) multiplies the learning rate by 0.1 and counts
towards `patience` = 2; the best checkpoint is restored for the test split.
