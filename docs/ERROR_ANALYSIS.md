# Error analysis (test split, frozen `v1.0-frozen` run)

Source: `results/eval/test_ft_1p5b/predictions.jsonl` (headline model: fine-tuned Qwen2.5-1.5B, greedy, 206 test criteria from 8 held-out patients). The model is wrong on **49 of 206** criteria (accuracy 0.762). The curated analysis below was compiled from those prediction records with AI assistance (see the AI-use appendix) and must be reviewed by the author. The automatically generated candidate list (all four models) follows it.

## 1. Where the 49 errors come from

| Gold → predicted | Type | Count | Share of errors |
|---|---|---:|---:|
| not enough information → not excluded | exclusion | 25 | 51% |
| not included → not enough information | inclusion | 7 | 14% |
| not applicable → not excluded | exclusion | 7 | 14% |
| included → not enough information | inclusion | 4 | 8% |
| not applicable → not enough information | inclusion | 3 | 6% |
| not excluded → not applicable | exclusion | 1 | 2% |
| not enough information → not applicable | exclusion | 1 | 2% |
| not enough information → included | inclusion | 1 | 2% |

Predicted label distribution compared with gold: `not excluded` 122 vs 91 and `not enough information` 53 vs 66. `not applicable` is 7 vs 15, and `not included` is **0 vs 7**. Almost every error is an *abstention-boundary* error: the model either guesses a decisive negative where the expert abstained, or abstains where the expert decided. It never flips the polarity of a decisive label (included ↔ not included, excluded ↔ not excluded).

## 2. Distinct error cases

1. **Absence treated as negation (ann. 169, exclusion).** "Estimated pre-operative GFR < 30 ml/min/1.73m²": the gold label is NEI and the prediction is *not excluded*, with no evidence cited. The note has no renal function values. The model reads "not mentioned" as "not present". This is the dominant failure mode (25 cases), and the app guards against it by routing uncited decisive labels to *needs review* (`unsupported_decision`).
2. **Same pattern, history item (ann. 174).** "Previous axillary surgery": gold NEI, predicted *not excluded*. A surgical history cannot be ruled out from an acute-presentation note.
3. **Same pattern despite expert evidence (ann. 193).** "Patients were moribund": the experts cite sentences 0–4 and still answer NEI, because the severity is suggestive but not conclusive. The model answers *not excluded* with no evidence. Judging overall clinical severity is harder than matching a fact.
4. **Procedure-type mismatch read as missing information (ann. 161–163, inclusion).** "Patients undergoing elective carotid endarterectomy / open AAA repair / endovascular AAA repair": the gold label is *not included* (the patient had cardiac catheterisation). The model answers NEI and cites sentences 0–2. It recognises the relevant procedure sentence but does not commit to "a different procedure means the criterion is not met". This explains all 7 *not included* errors (0 predicted).
5. **Qualitative vs numeric threshold (ann. 186).** "Heart rate of at least 90/min": the gold label is *included* from sentence 2 ("tachycardic"). The model cites the correct sentence 2 but abstains, because no number is given. Its evidence retrieval is right while its decision rule is stricter than the annotators'.
6. **Disjunctive criterion (ann. 185).** "Shock, ARDS, or both": gold *included* (sentences 2–3), predicted NEI (sentences 0, 2). Disjunctions with clinical synonyms need inference across sentences. The model only partly overlaps the expert evidence.
7. **Not applicable vs not excluded for sex-specific criteria (ann. 751).** "Breast-feeding" for a 5-year-old boy: gold *not applicable*, predicted *not excluded*. Both are defensible readings. The model blurs the NA/negative boundary on exclusion criteria (7 cases).
8. **Pregnancy/contraception criteria (ann. 322, 408).** These are labelled *not excluded* (322) and NEI (408) by experts, and the model predicts *not applicable* for both. The annotators themselves treat similar criteria inconsistently. This is label ambiguity rather than a pure model error.
9. **Conditional inclusion "If female, using birth control" (ann. 422, 745).** Gold *not applicable*, predicted NEI. Conditional criteria ("if female…") need a two-step check (does the condition apply? then is it met?) that the single-label output does not represent.
10. **Over-confident inclusion (ann. 762).** For a complex sore-throat criterion that needs "at least 2 of the following", the gold label is NEI and the prediction is *included* (sentences 0, 2). It is the only false decisive *included* error. Counting criteria is a known weakness of small models.
11. **Correct label, missing evidence (ann. 176).** "No medical history of central venous catheterisation": the label is correctly *included*, but no evidence is cited while the experts cite sentence 0. 27 correct-label rows have evidence sets that differ from the experts'. Label accuracy overstates explanation quality, which is why evidence F1 is reported separately.
12. **Correct label, extra evidence (ann. 195).** "Pain ≥ 3/10": correct *included*, citing sentences 0 and 2 against expert sentence 0. Extra citations lower evidence precision but are not necessarily wrong, because expert evidence is not exhaustive.

## 3. What this means for the design

- **Main risk:** false *not excluded* on missing information (case 1). For screening this is the unsafe direction, since a patient may look eligible when they are not. Mitigation in the app: any decisive label without cited evidence is flagged `unsupported_decision` and the trial verdict becomes *needs review*. Future work: class-weighted loss or oversampling of NEI exclusion rows (planned sweep, not run because of time).
- **Abstention calibration** dominates the remaining errors (cases 4–6, 9). A two-stage output (applicability, then decision) or more *not included* training examples (only 7 in test, and rare in train) would address it.
- **Annotation ambiguity** (cases 7–8) puts a ceiling on achievable accuracy for NA vs negative labels.
- Base models fail differently. Base 1.5B predicts *not applicable* for 109 of 206 test criteria (gold: 15) and predicts *excluded* 32 times (gold: 0). Fine-tuning reduces *not applicable* predictions to 7 and removes false *excluded* predictions entirely.

---

# Appendix: automated error candidates (all four models)

These are automated candidates for author review, not clinical conclusions. The tracked report omits patient-note text; inspect the local prediction records before correcting any diagnosis. Evidence-set disagreement does not prove that a citation is clinically irrelevant, and contradictions require human review.

Unique candidate annotations: 206.

| Annotation | Type | Gold -> prediction | Gold evidence IDs | Predicted evidence IDs | Signals | Criterion |
|---:|---|---|---|---|---|---|
| 169 (ft_1p5b) | exclusion | not enough information -> not excluded | [] | [] | missing_information_guess, unsupported_decision | Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2 |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The output marks a decisive judgment without cited evidence; this should remain a manual-review case. |
| 169 (ft_0p5b) | exclusion | not enough information -> not excluded | [] | [] | missing_information_guess, unsupported_decision | Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2 |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The output marks a decisive judgment without cited evidence; this should remain a manual-review case. |
| 169 (base_1p5b) | exclusion | not enough information -> not applicable | [] | [0] | applicability_confusion, evidence_set_discrepancy | Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2 |
|  |  |  |  |  | Diagnosis | The disagreement involves not-applicable; verify whether the criterion can apply to the patient described. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 174 (ft_1p5b) | exclusion | not enough information -> not excluded | [] | [] | missing_information_guess, unsupported_decision | Patients with previous axillary surgery |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The output marks a decisive judgment without cited evidence; this should remain a manual-review case. |
| 174 (ft_0p5b) | exclusion | not enough information -> not excluded | [] | [] | missing_information_guess, unsupported_decision | Patients with previous axillary surgery |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The output marks a decisive judgment without cited evidence; this should remain a manual-review case. |
| 174 (base_1p5b) | exclusion | not enough information -> not applicable | [] | [0] | applicability_confusion, evidence_set_discrepancy | Patients with previous axillary surgery |
|  |  |  |  |  | Diagnosis | The disagreement involves not-applicable; verify whether the criterion can apply to the patient described. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 179 (ft_0p5b) | exclusion | not excluded -> not excluded | [0] | [] | unsupported_decision, missed_gold_evidence | Had been performed central venous puncture or catheterization before. |
|  |  |  |  |  | Diagnosis | The output marks a decisive judgment without cited evidence; this should remain a manual-review case. The cited evidence does not overlap the expert evidence; inspect the sentence numbering and whether the model missed the supporting fact. |
| 179 (base_1p5b) | exclusion | not excluded -> excluded | [0] | [0] | decision_polarity_confusion | Had been performed central venous puncture or catheterization before. |
|  |  |  |  |  | Diagnosis | The predicted and expert labels are both decisive but disagree; verify polarity, negation, and criterion type against the note. |
| 179 (base_0p5b) | exclusion | not excluded -> not enough information | [0] | [] | unnecessary_abstention, missed_gold_evidence | Had been performed central venous puncture or catheterization before. |
|  |  |  |  |  | Diagnosis | The expert supplied a decisive label while the model abstained; check whether the relevant fact is explicit in the note. The cited evidence does not overlap the expert evidence; inspect the sentence numbering and whether the model missed the supporting fact. |
| 166 (ft_1p5b) | exclusion | not excluded -> not excluded | [] | [1, 2] | evidence_set_discrepancy | Significant upper limb peripheral arterial disease |
|  |  |  |  |  | Diagnosis | The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 166 (ft_0p5b) | exclusion | not excluded -> not excluded | [] | [] | unsupported_decision | Significant upper limb peripheral arterial disease |
|  |  |  |  |  | Diagnosis | The output marks a decisive judgment without cited evidence; this should remain a manual-review case. |
| 166 (base_1p5b) | exclusion | not excluded -> not applicable | [] | [0] | applicability_confusion, evidence_set_discrepancy | Significant upper limb peripheral arterial disease |
|  |  |  |  |  | Diagnosis | The disagreement involves not-applicable; verify whether the criterion can apply to the patient described. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 166 (base_0p5b) | exclusion | not excluded -> not enough information | [] | [] | unnecessary_abstention | Significant upper limb peripheral arterial disease |
|  |  |  |  |  | Diagnosis | The expert supplied a decisive label while the model abstained; check whether the relevant fact is explicit in the note. |
| 193 (ft_1p5b) | exclusion | not enough information -> not excluded | [0, 1, 2, 3, 4] | [] | missing_information_guess, unsupported_decision, missed_gold_evidence | Patients were moribund. |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The output marks a decisive judgment without cited evidence; this should remain a manual-review case. The cited evidence does not overlap the expert evidence; inspect the sentence numbering and whether the model missed the supporting fact. |
| 193 (ft_0p5b) | exclusion | not enough information -> not excluded | [0, 1, 2, 3, 4] | [] | missing_information_guess, unsupported_decision, missed_gold_evidence | Patients were moribund. |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The output marks a decisive judgment without cited evidence; this should remain a manual-review case. The cited evidence does not overlap the expert evidence; inspect the sentence numbering and whether the model missed the supporting fact. |
| 193 (base_1p5b) | exclusion | not enough information -> not applicable | [0, 1, 2, 3, 4] | [0] | applicability_confusion | Patients were moribund. |
|  |  |  |  |  | Diagnosis | The disagreement involves not-applicable; verify whether the criterion can apply to the patient described. |
| 193 (base_0p5b) | exclusion | not enough information -> not enough information | [0, 1, 2, 3, 4] | [] | missed_gold_evidence | Patients were moribund. |
|  |  |  |  |  | Diagnosis | The cited evidence does not overlap the expert evidence; inspect the sentence numbering and whether the model missed the supporting fact. |
| 307 (ft_1p5b) | inclusion | not enough information -> not enough information | [3] | [0, 1, 2, 3, 4] | evidence_set_discrepancy | Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right he |
|  |  |  |  |  | Diagnosis | The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 307 (ft_0p5b) | inclusion | not enough information -> not enough information | [3] | [0, 1, 2, 3, 4] | evidence_set_discrepancy | Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right he |
|  |  |  |  |  | Diagnosis | The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 307 (base_1p5b) | inclusion | not enough information -> included | [3] | [0, 1, 2, 3, 4] | missing_information_guess, evidence_set_discrepancy | Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right he |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 307 (base_0p5b) | inclusion | not enough information -> not enough information | [3] | [] | missed_gold_evidence | Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right he |
|  |  |  |  |  | Diagnosis | The cited evidence does not overlap the expert evidence; inspect the sentence numbering and whether the model missed the supporting fact. |
| 190 (ft_0p5b) | inclusion | not enough information -> not enough information | [] | [3] | evidence_set_discrepancy | the presence of acute decrease in PaO2/FIO2 to 200mmHg or lower, |
|  |  |  |  |  | Diagnosis | The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 190 (base_1p5b) | inclusion | not enough information -> included | [] | [0, 1, 2, 3, 4] | missing_information_guess, evidence_set_discrepancy | the presence of acute decrease in PaO2/FIO2 to 200mmHg or lower, |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 305 (ft_1p5b) | inclusion | not enough information -> not enough information | [] | [0, 1, 2, 3, 4] | evidence_set_discrepancy | Standard analytical parameters, defined by: Leukocytes ≥ 3000 Neutrophils ≥ 1500 Platelets ≥ 100,000 Aspartate aminotransferase (AST) / Alanine aminotransferase (ALT) ≤ 1.5 standard range institution creatinine ≤ 1.5 mg / dl |
|  |  |  |  |  | Diagnosis | The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 305 (base_1p5b) | inclusion | not enough information -> included | [] | [0, 1, 2, 3, 4] | missing_information_guess, evidence_set_discrepancy | Standard analytical parameters, defined by: Leukocytes ≥ 3000 Neutrophils ≥ 1500 Platelets ≥ 100,000 Aspartate aminotransferase (AST) / Alanine aminotransferase (ALT) ≤ 1.5 standard range institution creatinine ≤ 1.5 mg / dl |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 315 (base_1p5b) | exclusion | not enough information -> excluded | [] | [0] | missing_information_guess, evidence_set_discrepancy | BMI> 40 kg/m2. |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 323 (ft_0p5b) | inclusion | not enough information -> included | [0] | [0] | missing_information_guess | Immune Deficiency and early BMF in childhood |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. |
| 323 (base_1p5b) | inclusion | not enough information -> included | [0] | [0] | missing_information_guess | Immune Deficiency and early BMF in childhood |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. |
| 323 (base_0p5b) | inclusion | not enough information -> not enough information | [0] | [] | missed_gold_evidence | Immune Deficiency and early BMF in childhood |
|  |  |  |  |  | Diagnosis | The cited evidence does not overlap the expert evidence; inspect the sentence numbering and whether the model missed the supporting fact. |
| 325 (ft_1p5b) | inclusion | not enough information -> not enough information | [] | [0] | evidence_set_discrepancy | Genetic patients |
|  |  |  |  |  | Diagnosis | The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 325 (ft_0p5b) | inclusion | not enough information -> included | [] | [0] | missing_information_guess, evidence_set_discrepancy | Genetic patients |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 325 (base_1p5b) | inclusion | not enough information -> included | [] | [0] | missing_information_guess, evidence_set_discrepancy | Genetic patients |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |
| 389 (ft_1p5b) | exclusion | not enough information -> not excluded | [] | [] | missing_information_guess, unsupported_decision | Severe renal impairment (estimated or measured GFR <50%) |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The output marks a decisive judgment without cited evidence; this should remain a manual-review case. |
| 389 (ft_0p5b) | exclusion | not enough information -> not excluded | [] | [] | missing_information_guess, unsupported_decision | Severe renal impairment (estimated or measured GFR <50%) |
|  |  |  |  |  | Diagnosis | The expert label is not-enough-information while the model made a decisive judgment; inspect whether the note actually supports that judgment. The output marks a decisive judgment without cited evidence; this should remain a manual-review case. |
| 389 (base_1p5b) | exclusion | not enough information -> not applicable | [] | [0] | applicability_confusion, evidence_set_discrepancy | Severe renal impairment (estimated or measured GFR <50%) |
|  |  |  |  |  | Diagnosis | The disagreement involves not-applicable; verify whether the criterion can apply to the patient described. The model cited sentence IDs outside the expert evidence set; assess relevance locally because expert evidence may not be exhaustive. |

## Candidate-category counts

| Category | Model-record count |
|---|---:|
| applicability_confusion | 150 |
| decision_polarity_confusion | 28 |
| evidence_set_discrepancy | 243 |
| missed_gold_evidence | 121 |
| missing_information_guess | 85 |
| unnecessary_abstention | 152 |
| unsupported_decision | 224 |
