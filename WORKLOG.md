# Work log

One bullet per action. Newest section at the bottom. Each entry says what was
done, not what was intended.

## 2026-08-08 — Stage 3 input changed to a dense per-voxel field

- Built `3_global_tractography/dense_field.py`: a `DenseField` carrying Stage 1's
  e3 at every voxel, overwritten by HOST's first peak inside routed patches, with
  HOST's second peak as `vec2` where the patch is in Stage 2's CANDIDATE bucket.
- Verified the carrier: `|vec1 · e3| = 1.000000` at every voxel no patch painted;
  `vec2` non-zero only inside routed patches and bounded to the junction.
- Moved the patch-painting out of `field_export.py`, which now converts layout and
  writes NIfTI only; `LABEL_CODES` / `CODE_LABELS` re-exported for existing callers.
- Added label code `unclassified` (8) for voxels Stage 1 oriented but HOST never
  evaluated — 25,796 of 35,033 on the crossing phantom were being written as
  `background` while shipping a measured direction in `vec1`.
- Extended `test_frame_convention.py` with a synthetic `DenseField` case covering
  the patch-direction reversal, the e3 pass-through, and that `demote` retracts
  `vec2` and not `vec1`. Passes.
- Built `3_global_tractography/dense_tracker.py`: cone search over the dense field,
  all walkers advanced together, cone geometry precomputed per (θ, α).
- Measured the speed-up against the recorded scalar sweep: 16× to 171× faster
  (worst recorded cell 3171 s → 18.5 s), with identical path counts and traversal.
- Fixed a normalisation bias: dividing by `norm + 1e-12` shortened every step by
  1e-12, compounding along a tract and adding one sample before the volume
  boundary. Now divides by `max(norm, 1e-12)`; agreement with the scalar tracker
  is exact.
- Scored two-direction voxels by `max(|d·vec1|, |d·vec2|)`, which degrades exactly
  to the single-direction case because `vec2` is zero there.
- Built `3_global_tractography/dense_optimise.py`: TDI loss over the dense tracts,
  scored on the whole tissue mask instead of a neighbourhood of the HOST patches.
- Replaced the grid sweep with coordinate descent over (lookahead, θ); α and the
  seed budget held fixed as compute budgets, neither having a minimum to find.
- Built `3_global_tractography/single_fibre_check.py` as a standing gate: the angle
  between each tract's tangent and the field's own `vec1` where only one direction
  exists.
- The gate failed at 12.46°, against e3's own voxel-to-voxel coherence of 1.39°.
- Ruled out angular discretisation: the deviation is flat from α=0.4 to α=0.025,
  a 16× refinement (11.98° → 12.93°).
- Found and fixed director sign cancellation: 13.2% of adjacent voxel pairs are
  sign-opposed, so componentwise trilinear sampling of `vec1` returned |v| with
  median 0.78 and 31.8% of reads below 0.5. Now interpolates the dyadic `v vᵀ`,
  which has no sign to cancel. Worth 0.6°.
- Found the dominant cause: brightness was choosing the turn. Separated the two
  roles — direction now comes from alignment alone, intensity only sets the floor
  that ends a walk. Deviation 12.15° → 4.82°, with crossing traversal (0.104) and
  kissing bridge (0.000) unchanged.
- Gate now passes at 5.72° against a 5.73° tolerance tied to the cone's own
  angular resolution. A marginal pass, recorded as such.
- Capped both seed families independently. Seeding two-direction voxels at full
  density gave 53,322 seeds against a budget of 2,000 on i74, where 51,444 voxels
  carry a second direction.
- Repaired three imports left dangling when `field_refine.py` was archived by
  another session: `fill_invalid` now lives in `dense_field.py`, and the
  Frank-Oseen import is lazy behind `--refine`.
- Aligned `DenseField` with Stage 2's `BUCKET_CANDIDATE` triage, so the field the
  tracker walks and the junction list agree about where two fibres are claimed.
- Phantom validation: 2/2, exit 0. Crossing → CROSSING (S=1.032), kissing → no
  junction invented. Coverage 63.7% → 96.6%.
- Ran i74 (96³ at the volume centre) with both trackers. Cone: L 7.0104 → 5.4753
  by descent, → 2.6058 after demotion. Patch: L 1.0538 → 1.0538, no move found.
- Recorded that the two trackers produced identical verdicts — 25 junctions, 19
  demoted, same distribution — so the tractography is not currently contributing
  to the crossing/kissing decision. The verdict comes from S alone.
- Documented the dense input, the two support-function findings, and the new
  scripts in `PIPELINE.md` and `3_global_tractography/README.md`.

## 2026-08-08 — letting the tracts vote on each junction

- Added `dense_optimise.demote_by_evidence`: for each junction, track the field
  with its second direction kept and again with it retracted, and keep whichever
  explains the image better.
- Scored that comparison on the data term alone, with the verdict penalty
  switched off. The penalty is derived from S, so including it would let S vote
  on the question it is being tested against.
- Held the seeds identical across both arms, so the only difference between them
  is the field and not the size of the tractogram.
- Added `--demote {s,evidence,both}`; the default runs both rules on the same
  junctions and records where they disagree.
- Phantom result: on the one junction with known ground truth — a real crossing —
  retracting the second direction LOWERED the data term (mse 0.388112 →
  0.385999), so the evidence rule voted TANGENCY and was wrong. It agreed with S
  on 0 of 1 junction.
- Recorded the likely cause, which is already a stated caveat of the target:
  `D_obs = λ1 − λ3` cannot distinguish one bright fibre from two dim crossing
  ones, so the density term is close to blind to the exact question being asked
  of it.
- Phantom validation still 2/2, exit 0 — the evidence rule is reported alongside
  rather than replacing S, so it cannot change the pipeline's verdicts.
- Measured the data term's noise floor by re-running the loss with a different
  arbitrary subset of seeds and no change to the field at all: mse moved −0.38%,
  −0.45% and −0.74%.
- Concluded the evidence rule is reading noise. Retracting a real second
  direction moves the data term −0.54%, which is inside that band and smaller
  than one of the three controls. The TDI residual cannot resolve the presence
  of a second fibre at a junction.
- Left S as the sole decision rule. `--demote evidence` remains available but is
  not the default, and the phantom disagreement is recorded against it.


- Ran i74 (96³, 25 junctions) with both rules. 

- Retracting a second direction
  changed the data term by a median of +0.000003 on a baseline of 1.2298 — three
  parts per million — with the full range, −0.381% to +0.277%, inside the noise
  floor measured above.

  
- Recorded that 12 of 25 junctions "improved" on retraction, a 48% coin flip, and
  that the evidence rule agreed with S on 9 of 25 (36%): it called CROSSING at 10
  junctions S called TANGENCY, and TANGENCY at the one S called CROSSING.


- Closed the question: the TDI residual carries no usable information about
  whether a junction holds one fibre or two (phantom or on real data) .

## 2026-08-13 — Stage 4 vec2 zero-fill removed; the rho answer is grid-limited

- Fixed a silent zero-fill in `4_optimising_parameters/sigma_rho_match.py`. It
  loaded the Stage 3 field with `refined.vec2 = z["vec2"] if "vec2" in z else
  np.zeros_like(z["vec1"])`, so a field with no second direction produced a
  full-size all-zero `refined_vec2.nii.gz` — a file that reads on disk exactly
  like a computed result. `vec2` is now `None` when absent and every consumer
  handles that explicitly.
- Made the absence self-explaining. The run prints which of the two causes it
  can distinguish from the recorded state: the run used `--variant vec1`, so no
  second direction was ever carried, or Stage 2 produced none on the chunk
  (`n_two_dir_voxels = 0`).
- Stopped writing `refined_vec2.nii.gz` unless a second direction exists, and
  deleted any stale copy from a previous run — otherwise a zero volume sits
  beside fresh outputs looking current.
- Recorded `has_vec2` and `n_vec2_voxels` in `stage4_sigma_rho.json`, so a
  downstream reader cannot infer "exists and is zero" from a missing file.
- Verified on a re-run against `results/acc_r12`: `removed stale
  refined_vec2.nii.gz from an earlier run`, `has_vec2 False`,
  `n_vec2_voxels 0`, `paths written ['refined_vec1', 'best_simple_e3']`. No
  `refined_vec2.nii.gz` on disk; the 854-byte zero volume is gone.
- Established why vec2 was empty, which is upstream of superposition entirely:
  `triage` grants `BUCKET_CANDIDATE` only to the label `crossing`, and
  `DenseField.build` writes `vec2` only for that bucket. Superposition and
  `demote()` are subtractive — they zero an existing `vec2` and drop `n_dirs`
  to 1, and never create one. The chunk simply had no `crossing` patches.

- Found that the Stage 4 rho answer is limited by the grid, not measured by it.
  The same refined field gives rho = 1.7818 on a 7x7 grid and rho = 2.0000 on a
  5x5 grid. Both are one grid step above Stage 1's rho = 1.4142; the two grids
  place that step differently. The median error barely moves with the change —
  14.140 deg at n=7 against 14.353 deg at n=5 — so neighbouring grid points are
  near-equivalent and the surface is flat at this spacing.
- Corrected the earlier reading accordingly: the result is "rho shifted upward
  by roughly one step", NOT "rho = 1.78". Locating it needs a finer grid.
- Recorded the stronger statement the sweep supports: the refined field is
  ~14 deg from the closest plain structure tensor anywhere on the grid, so the
  best-matching (sigma, rho) is a weak summary of it. Stage 3 now produces a
  field a single structure tensor cannot reproduce, which is a more informative
  outcome than the scale shift.
- Control unchanged and exact: sweeping against the Stage 2 field returns
  sigma = 1.0000, rho = 1.4142 at 0.000 deg median error, confirming Stage 2's
  field IS the plain tensor at Stage 1's own pick and that the sweep is sound.
