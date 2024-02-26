# Changelog

## 2.1.0
**26 February 2024**
- Addition of RotatE to the available scoring functions.
- Addition of a module to load models (TransE, DistMult, ComplEx, RotatE, HolE) pre-trained on benchmark datasets (fb15k-237,
wn18rr, yago3-10, fb15k, wn18).
- Improved efficiency of the validation.
- Other minor efficiency improvements, fixes and code clean-up.


## 2.0.1
**12 July 2023**
- Fixed bug preventing the saving of calibrated models.
- Extended type support for the predict method to list of triples.
- Updated experiments performance.
- Minor fixes.


## 2.0.0
**7 March 2023**
- Switched to TensorFlow 2 back-end.
- Keras style APIs.
- Unique model class ScoringBasedEmbeddingModel for all scoring functions that can be specified as a parameter when initializing the class.
- Change of the data input/output pipeline.
- Extension of supported optimizers, regularizers and initializer.
- Different data storage support: no-backend (in memory) and SQLite-based backend.
- Codex-M Knowledge Graph included in the APIs for automatic download.
- ConvKB, ConvE, ConvE(1-N) not supported anymore as they are computationally expensive and thus not commonly used.
- Support AmpliGraph 1.4 API within ampligraph.compat module.


## 1.4.0
**26 May 2021**

- Added support for numerical attributes on edges (FocusE) (#235)
- Added loaders for benchmark datasets with numeric values on edges (O*NET20K, PPI5K, NL27K, CN15K)
- Added discovery API to find nearest neighbors in embedding space (#240)
- Change of optimizer (from BFSG to Adam) to calibrate models with ground truth negatives (#239)
- 10x speed improvement on train_test_split_unseen API (#242)
- Added support to visualize training progression via tensorboard (#230)
- Bug fix in large graph mode (when evaluate_performance with entities_subset is used) (#231)
- Updated save model api to save embedding matrix > 6GB (#233)
- Doc updates (#247, #221)
- Fixed ntriples loader spurious trailing dot.
- Add tensorboard_logs_path to model.fit() for tracking training loss and early stopping criteria.


## 1.3.2
**25 Aug 2020**

- Added constant initializer (#205)
- Ranking strategies for breaking ties (#212)
- ConvE Bug Fixes (#210, #194)
- Efficient batch sampling (#202)
- Added pointer to documentation for large graph mode and Docs for Optimizer (#216)


## 1.3.1 
**18 Mar 2020**

- Minor bug fix in ConvE (#189)


## 1.3.0 
**9 Mar 2020**

- ConvE model Implementation (#178)
- Changes to evaluate_performance API (#183)
- Option to add reciprocal relations (#181)
- Minor fixes to ConvKB (#168, #167)
- Minor fixes in large graph mode (#174, #172, #169)
- Option to skip unseen entities checks when train_test_split is used (#163)
- Stability of NLL losses (#170)
- ICLR-20 calibration paper experiments added in branch paper/ICLR-20 


## 1.2.0 
**22 Oct 2019**

- Probability calibration using Platt scaling, both with provided negatives or synthetic negative statements (#131)
- Added ConvKB model
- Added WN11, FB13 loaders (datasets with ground truth positive and negative triples) (#138)
- Continuous integration with CircleCI, integrated on GitHub (#127)
- Refactoring of models into separate files (#104)
- Fixed bug where the number of epochs did not exactly match the provided number by the user (#135)
- Fixed some bugs on RandomBaseline model (#133, #134)
- Fixed some bugs on discover_facts with strategies "exhaustive" and "graph_degree"
- Fixed bug on subsequent calls of model.predict on the GPU (#137)

## 1.1.0 
**16 Aug 2019**

- Support for large number of entities (#61, #113)
- Faster evaluation protocol (#74)
- New Knowledge discovery APIs: discover facts, clustering, near-duplicates detection, topn query (#118)
- API change: model.predict() does not return ranks anymore
- API change: friendlier ranking API output (#101)
- Implemented nuclear-3 norm (#23)
- Jupyter notebook tutorials: AmpliGraph basics (#67) and Link-based clustering 
- Random search for hyper-parameter tuning (#106)
- Additional initializers (#112)
- Experiment campaign with multiclass-loss
- System-wide bugfixes and minor improvements


## 1.0.3 
**7 Jun 2019**

- Fixed regression in RandomBaseline (#94)
- Added TensorBoard Embedding Projector support (#86)
- Minor bugfixing (#87, #47)


## 1.0.2
**19 Apr 2019**

- Added multiclass loss (#24 and #22)
- Updated the negative generation to speed up evaluation for default protocol.(#74)
- Support for visualization of embeddings using Tensorboard (#16)
- Save models with custom names. (#71)
- Quick fix for the overflow issue for datasets with a million entities. (#61)
- Fixed issues in train_test_split_no_unseen API and updated api (#68)
- Added unit test cases for better coverage of the code(#75)
- Corrupt_sides : can now generate corruptions for training on both sides, or only on subject or object
- Better error messages
- Reduced logging verbosity
- Added YAGO3-10 experiments
- Added MD5 checksum for datasets (#47)
- Addressed issue of ambiguous dataset loaders (#59)
- Renamed ‘type’ parameter in models.get_embeddings  to fix masking built-in function
- Updated String comparison to use equality instead of identity.
- Moved save_model and restore_model to ampligraph.utils (but existing API will remain for several releases).
- Other minor issues (#63, #64, #65, #66)


## 1.0.1 
**22 Mar 2019**

- evaluation protocol now ranks object and subjects corruptions separately
- Corruption generation can now use entities from current batch only
- FB15k-237, WN18RR loaders filter out unseen triples by default
- Removed some unused arguments
- Improved documentation
- Minor bugfixing

## 1.0.0
**16 Mar 2019**

- TransE
- DistMult
- ComplEx
- FB15k, WN18, FB15k-237, WN18RR, YAGO3-10 loaders
- generic loader for csv files
- RDF, ntriples loaders
- Learning to rank evaluation protocol
- Tensorflow-based negatives generation
- save/restore capabilities for models
- pairwise loss
- nll loss
- self-adversarial loss
- absolute margin loss
- Model selection routine
- LCWA corruption strategy for training and eval
- rank, Hits@N, MRR scores functions
