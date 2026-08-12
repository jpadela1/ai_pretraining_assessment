# AI Pre-Training Safety Assessment: An Automated, Modality-Aware Rubric for Scoring Dataset Safety Risk Before Training

*Anonymous Author(s) — Anonymous Institution*

> **Assembly note (delete before submission).** This is the consolidated clean draft.
> Your Intro / Related Work / Methodology spine is preserved and corrected; the stale
> Results, the drafting-note sections (Library, Discussion, Conclusion), and the old
> Abstract are rewritten; everything now reflects the two confirmed arms (text toxicity
> on Pythia-410M, tabular label-integrity on Diabetes 130) and the model-scale finding.
> Figures referenced: Fig. 1 (framework flow — **you must redraw, see punch-list**),
> Fig. 2 = `fig_calibration.png`, Fig. 3 = `fig_h3_410m.png`, Fig. 4 = `fig_tabular_dose.png`.

---

## Abstract

Existing AI risk governance frameworks—the NIST AI Risk Management Framework and OMB Memorandum M-25-21 among them—classify and control risk at the level of the deployed system. Yet the dominant determinants of downstream harm—training-time poisoning, harmful content the model may reproduce, and corrupted labels—originate in the training data, before any model exists. What is missing is an instrument: an automated, pre-training mechanism that scores a dataset's safety risk at the data layer. We propose a decomposed safety rubric whose sub-dimensions are each operationalized through a reproducible automated proxy, released as an importable library, and we validate it through controlled injection across two modalities. Injecting a known dose of harmful content into text fine-tuning data, the harm-content proxy recovers the dose almost exactly (r = 0.9996) and predicts the downstream model's generated toxicity (r = 0.92, 95% CI excluding zero) once the model is of sufficient scale; injecting label noise into a clinical tabular dataset, an integrity proxy recovers the dose (r = 0.98) and predicts downstream AUC degradation and miscalibration. The instrument is therefore feasible, reproducible through automated proxies, modality-extensible, and complementary to existing system-level governance.

**Index Terms—** data-centric AI, AI risk management, safety-impacting AI, training-data assessment, dataset documentation, large language models.

---

## I. Introduction

Existing AI risk governance treats the deployed system as the unit of analysis. The NIST AI Risk Management Framework (RMF) [1], OMB Memorandum M-25-21 [2], and the EU AI Act [3] all classify systems by risk tier and prescribe controls accordingly. This view inverts the causal order: the dominant sources of downstream risk—poisoned or corrupted records, harmful content a model may reproduce, and outdated or mislabeled data—originate in the training data, not in the model architecture or the deployment context alone. Governance measures safety at the deployed-system level, after the determinants of that safety have already been baked in.

What is missing is an instrument: an automated, pre-training, data-layer mechanism that scores a dataset's safety risk before it trains a model. Our contributions are: (1) a decomposed safety rubric whose sub-dimensions are each operationalized through a documented automated proxy; (2) a working, importable library that computes the score; and (3) a feasibility validation across two modalities—text and clinical tabular—demonstrating that the pre-training score predicts downstream harmful behavior.

The motivation is operational as much as conceptual. AI systems are typically well tested at deployment, with data validated and monitored at that stage. Once data is ingested, however, it is rarely re-examined for the properties that matter for safety, largely due to limited funding and shifting priorities. A safety assessment is therefore most effective at the point of ingestion or when data is repurposed for a new model, rather than after a harm has surfaced downstream. Our instrument is designed for exactly that point in the lifecycle.

## II. Related Work and Limitations

**a) Data quality frameworks.** Pipino et al. [4] provide the most widely cited operational decomposition of data quality, with sixteen dimensions spanning objective and subjective measurement. This framework is anchored in relational and warehouse settings and was not designed to capture the failure modes that dominate generative AI—training-time poisoning, large-scale representation gaps in scraped corpora, and harmful-content reproduction.

**b) Dataset documentation standards.** Datasheets for Datasets [5], Data Statements for NLP [6], and Model Cards [7] each prescribe what information should accompany a released dataset or model. These standards have substantially raised the floor of transparency, but they are documentation instruments rather than scoring instruments: a datasheet records what is true about a dataset without producing a quantitative measure of fitness. Our framework is complementary—a completed datasheet supplies much of the input our proxies consume, and our score can be reported alongside it.

**c) AI risk governance.** The NIST AI RMF [1] organizes risk management into Govern, Map, Measure, and Manage. OMB M-25-21 [2] directs federal agencies to evaluate the data used to develop AI applications and articulates "safety-impacting" and "rights-impacting" AI as governance categories. The EU AI Act [3] establishes tiered obligations by system risk class. Across these documents the data layer is treated as an input to be documented and monitored; none provides a quantitative instrument for scoring the data's safety before it enters a model.

**d) Data-centric AI and fairness auditing.** Data-centric work argues that systematic data-quality improvement often yields greater returns than further model engineering. Fairness toolkits—AIF360 [8], Fairlearn [9]—provide post-hoc auditing of *trained models*; they inform a practitioner whether a deployed model fails a criterion, not whether a candidate dataset is likely to produce such a failure. Across these works, each treats the data layer as an input to document, but none produces a pre-training safety score to inform a go/no-go threshold. Our contribution is the first quantitative safety instrument operating at the data layer before model training.

## III. AI Safety Assessment Framework

### A. Design principles

Three principles guide the framework. **Measurability:** every sub-dimension has a defined proxy with a documented computation, so scoring is reproducible across datasets and organizations. **Modality awareness:** the proxy used for a sub-dimension depends on the data type, because harmful content in text, in images, and in tabular labels manifests differently and demands different detectors. **Transparency:** the value judgments a proxy encodes (which classifier, which threshold, which vocabulary) are named, versioned, and swappable rather than hidden, so the score is challengeable rather than oracular.

### B. Safety-impact rubric

Both NIST AI RMF [1] and OMB M-25-21 [2] introduce "safety-impacting AI" as a governance category but do not specify how to measure safety impact at the data layer. We decompose the concept into six sub-dimensions in two groups (Table I). The **content-origin (CO)** group characterizes the data as received: how susceptible it is to malicious modification (poisoning), how likely it contains undetected adversarial contributions, how quickly its facts decay, and how densely it contains harmful content a model might reproduce. The **physical-safety (PS)** group characterizes what the data could enable downstream: content providing meaningful uplift toward physical harm, and coverage of safety-critical edge cases for systems that control physical processes.

We treat safety as a dataset's propensity to induce harmful downstream behavior, with two faces: harmful-content reproduction (content-origin) and integrity or coverage failure in a consequential setting (physical-safety). Both reduce to one underlying property—the *trustworthiness of the data entering training*—which is why the content-origin sub-dimensions form the conceptual core of the instrument and the focus of our validation.

Two clarifications. First, harm-content density (CO) and physical-harm enablement (PS) are distinct: the former concerns content a model might reproduce as output; the latter concerns content that would furnish technical capability if queried. They have different proxies and different threshold semantics. Second, a high physical-harm-enablement score does not by itself recommend exclusion—legitimate scientific corpora can score high—so the rubric flags such datasets for elevated review and access control, not automatic removal.

### C. Composite scoring

The safety composite *S(D)* is the unweighted mean of the applicable sub-dimensions: the content-origin sub-dimensions plus any physical-safety sub-dimensions whose application conditions are met. Sub-dimensions that do not apply (for example, safety-critical edge-case coverage for a purely informational dataset) are removed from the denominator rather than scored as zero, since a zero would falsely read as "assessed and safe" when the correct interpretation is "not applicable." A deploying organization applies a threshold to *S* (or to an individual sub-dimension) appropriate to its risk tolerance; the instrument supplies the measurement, the deployer owns the decision.

## IV. Implementation: The Library

The instrument is released as an importable Python library so that the score is not a paper artifact but a usable tool. A caller points the library at a dataset and its documentation and receives the per-sub-dimension scores and the composite. Scoring runs through a **pluggable proxy interface**: each sub-dimension is backed by a proxy object with a common signature, so a proxy can be swapped (a different toxicity classifier, a different label-noise estimator) without touching the scoring logic, and a new modality is added by registering a modality-specific proxy. Image-content proxies are a documented extension point in this interface, allowing the framework to be described as modality-extensible without overclaiming validation we have not performed.

Reproducibility is built in: every score is recorded with a cryptographic hash of the exact dataset version it was computed from, together with the proxy identifiers and parameters used, so a third party can reproduce or contest any value. The current release depends on Detoxify [10] for text harm-content scoring and standard scientific-Python libraries for the tabular proxies; the full implementation and replication scripts are in the released repository.

## V. Methodology

**a) From observational to controlled validation.** A correlational study across a handful of public datasets cannot establish that a pre-training score *causes* the downstream behavior it predicts; the available variance in any single risk dimension is small and confounded with everything else that differs across datasets. We therefore validate through **controlled injection**: starting from a clean dataset, we introduce a known dose of the specific risk a sub-dimension measures, and verify two things—that the automated proxy rises with the dose, and that a model trained on the injected data degrades on a *matched* downstream metric. Holding dataset size, model architecture, and training hyperparameters fixed across doses isolates the injected risk as the sole varying factor, yielding an interpretable dose-response.

**b) Datasets and modalities.** We validate two content-origin sub-dimensions in two modalities. **CivilComments** [11] (toxicity-labeled text) instantiates harm-content density. **Diabetes 130** [12], [13] (clinical readmission records) instantiates data integrity, where a mislabeled record is a genuine safety failure—an erroneous early-readmission prediction—rather than a mere accuracy loss. The two arms share one safety claim across distinct data types, model classes, and failure modes.

**c) Automated proxies.** Harm-content density is scored with Detoxify [10] over a stratified sample of 1,000 records. Data integrity is scored with a cross-validated, confident-learning-style estimator: a model's out-of-fold predicted probability for each record's assigned label is computed, and the mean disagreement (one minus that probability) estimates the corrupted fraction. Both proxies are automated, reproducible, and swappable.

**d) Text arm—models and fine-tuning.** To test sensitivity to model scale we fine-tune two causal language models from the same family, Pythia-160M and Pythia-410M [16], on the comment text with a standard language-modeling objective for one epoch (AdamW, learning rate 1×10⁻⁵, linear warmup over the first 10% of steps, gradient-norm clipping at 0.5, maximum sequence length 128), with all hyperparameters held fixed across doses so that only the training data varies. We use five random seeds per dose.

**e) Text arm—injection and matched metric.** We build fixed-size fine-tuning mixes (10,000 records) at six injected toxic fractions—0, 2.5, 5, 10, 20, 40%—drawing toxic examples from records labeled at or above 0.5 toxicity and clean examples from those at or below 0.1, holding mix size constant so dose is unconfounded with data volume. The matched downstream metric is *generated* toxicity: each fine-tuned model produces ten sampled continuations (nucleus sampling, top-p 0.9, 20 new tokens) for each of 1,000 RealToxicityPrompts [15], scored by Detoxify. We report Expected Maximum Toxicity (EMT, the mean over prompts of the most toxic continuation) and Toxicity Probability (the fraction of prompts with any continuation above 0.5).

**f) Tabular arm—models, injection, and matched metrics.** We train logistic regression and gradient-boosted trees (XGBoost) on a 75/25 stratified split, injecting symmetric label noise into the *training* labels only at 0, 5, 10, 20, 40%, with test labels never perturbed, across five seeds. The matched metrics are held-out AUC and expected calibration error (ECE); both are threshold-independent and therefore informative despite the dataset's class imbalance.

**g) Hypotheses.** The paper advances one claim—*an automated, pre-training safety score predicts the downstream harmful behavior of a model trained on the data*—supported by four evidentiary tests, with the zero-dose condition in each arm serving as an internal negative control:
- **H1 (calibration):** the proxy score increases monotonically with the injected dose.
- **H2 (predictive validity):** the proxy predicts the matched downstream harm.
- **H3 (generality):** H1–H2 hold across both modalities and model classes.
- **H4 (scale sensitivity):** text-arm predictive validity strengthens with downstream model capacity.

**h) Statistical approach.** For each arm we pool the (dose × seed) observations and regress the matched metric on the proxy score. We report the Pearson correlation *r* between proxy and downstream metric across injection conditions and a 95% bootstrap confidence interval (CI) obtained by resampling the observation points 10,000 times; a relationship is **confirmed** when the slope has the predicted sign and the CI for *r* excludes zero. We favor effect sizes and confidence intervals over null-hypothesis significance tests, pre-register all injection levels, seeds, proxies, and metrics before running, and report null results with equal prominence.

## VI. Results

We report the two arms and the scale analysis. Throughout, the zero-dose condition is the internal negative control. Table III maps each hypothesis to its result.

**a) H1—Proxy calibration.** In the text arm, the harm-content proxy recovers the injected dose almost exactly: across the six levels the proxy is linear in the injected proportion (score ≈ 0.298·p + 0.007) with r = 0.9996 and R² = 0.999 (Fig. 2). The small intercept reflects residual toxicity in the nominally clean pool, not measurement error. The calibration is non-circular: the injected fraction is defined by human CivilComments annotations while the proxy is computed by an independent classifier, so their agreement is genuine. In the tabular arm the integrity proxy likewise tracks injected label noise closely (r = 0.980, CI [0.975, 0.986]; Fig. 4, left). H1 is confirmed in both modalities.

**b) H2—Downstream predictive validity.** *Text:* using Pythia-410M, the pre-training proxy strongly predicts generated toxicity, r = 0.924, CI [0.843, 0.968] (Fig. 3, left). Mean EMT rises monotonically across every dose, from 0.073 at zero injection to 0.427 at 40%—a 5.9-fold increase—with per-dose standard deviations (0.01–0.09) a small fraction of the effect. Toxicity Probability makes the magnitude concrete: the share of prompts eliciting toxic output grows from 2.6% at baseline to 43.2% at the highest dose (r = 0.917, CI [0.822, 0.969]). *Tabular:* as estimated integrity degrades, downstream AUC falls monotonically (r = −0.868, CI [−0.921, −0.792]) from 0.642 on clean data to 0.545 near the 40% dose, and calibration error climbs steeply (ECE r = +0.976, CI [0.970, 0.983]; Fig. 4, right). H2 is confirmed in both modalities. (Balanced accuracy at a fixed 0.5 threshold is uninformative here—pinned near 0.505 by the 11% base rate—which is precisely why we report the threshold-independent AUC and ECE.)

**c) H3—Generality.** The proxy-predicts-harm relationship holds across two unrelated modalities, model classes, and failure modes: harmful-content reproduction in fine-tuned language models, and integrity failure in classical clinical classifiers. A single safety claim is therefore supported by independent evidence on text and tabular data. H3 is confirmed.

**d) H4—Scale sensitivity.** The text-arm relationship depends strongly on model scale, and the comparison converts an apparent null into evidence. The identical protocol on Pythia-160M yields only a weak, non-monotonic association (r = 0.20, CI [−0.16, 0.55], crossing zero) with per-dose variance as large as the dose effect; raising capacity to Pythia-410M, with data and design held fixed, lifts the correlation to r = 0.924 with a CI excluding zero (Fig. 3, right). The effect is unresolved at small scale rather than absent—the larger model is both more sensitive to the fine-tuning distribution and more stable to train—corroborating the observation that small models register brief fine-tuning shifts weakly. H4 is confirmed.

**TABLE III — HYPOTHESES-TO-RESULTS MAP**

| # | Claim | Result |
|---|---|---|
| H1 | Proxy increases with injected harm | Text r = 0.9996; Tabular r = 0.980 — **confirmed (both)** |
| H2 | Proxy predicts downstream harm | Text EMT r = 0.924 [0.843,0.968]; Tabular AUC r = −0.868 [−0.921,−0.792], ECE r = 0.976 — **confirmed (both)** |
| H3 | Holds across modalities | Text (LM toxicity) + Tabular (clinical integrity) — **confirmed** |
| H4 | Strengthens with model scale | 160M r = 0.20 (crosses 0) → 410M r = 0.924 (excludes 0) — **confirmed** |

## VII. Discussion and Limitations

The evidence supports a scoped, honest claim: *under controlled injection, on these datasets and models, an automated pre-training safety proxy predicts downstream harm.* We state this as feasibility, not as a universal law. Several limitations bound it and define the future work.

**Scale dependence.** Predictive validity emerged only at Pythia-410M, not at 160M. The proxy's value should therefore be evaluated against downstream models of sufficient capacity; we have not characterized where the threshold lies or how the relationship behaves at billion-parameter scale.

**Partial convergence by design.** One epoch leaves training loss above its floor; the models are deliberately *not* fully converged. Because we measure the *relative* toxicity shift across conditions identical except for dose, equal partial convergence across doses is a feature of the controlled design, not a confound—but absolute toxicity levels should not be read as production estimates.

**Single proxy per sub-dimension.** Each sub-dimension is validated through one proxy (Detoxify; a confident-learning-style estimator). The proxies encode design choices that we make transparent and swappable, but a human-rater reliability study calibrating proxy outputs against expert judgment remains future work.

**Coverage.** We validated two content-origin sub-dimensions in two modalities. The physical-safety group and an image modality are specified in the rubric and accommodated by the library's pluggable interface but not yet empirically validated; they are the natural next experiments. Likewise, a lifecycle re-assessment mechanism—re-scoring data when it is repurposed across application boundaries—follows directly from the framework and is a planned extension.

## VIII. Conclusion

AI risk frameworks classify deployed systems by risk tier, but the determinants of risk reside in the training data. We have proposed an automated, modality-aware instrument that scores a dataset's safety risk before it trains a model, decomposed into measurable sub-dimensions backed by transparent proxies and released as an importable library. Through controlled injection we showed, in two modalities, that the pre-training score is near-perfectly calibrated to injected harm and predicts the downstream model's harmful behavior—generated toxicity in fine-tuned language models (r = 0.92 at adequate scale) and AUC degradation and miscalibration in clinical classifiers (r = −0.87, +0.98). Dataset-level safety pre-screening is therefore feasible, reproducible through automated proxies, and complementary to existing system-level governance. Replication code is available at the anonymized repository.

## IX. Acknowledgment

The authors used the AI tool Claude (claude.ai) for code generation, debugging, benchmarking scripts, and formatting; all technical content and conclusions are the authors' responsibility.

## References

*Keep [1]–[14] from your current draft. Add the two now load-bearing citations and verify details:*

- **[15]** S. Gehman, S. Gururangan, M. Sap, Y. Choi, and N. A. Smith, "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models," in *Findings of EMNLP*, 2020.
- **[16]** S. Biderman et al., "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling," in *Proc. ICML*, 2023.
- *(optional, if you cite the proxy lineage)* C. G. Northcutt, L. Jiang, and I. L. Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels," *JAIR*, 2021.
- *Remove the TruthfulQA, Folktables, and German Credit citations — they belong to experiments no longer in the paper.*
