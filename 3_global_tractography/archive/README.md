# Archived Stage 3 scripts

Moved here on 2026-08-11 when Stage 3 was narrowed to the **descent loop**
(`descent_loop.py` + `tdi_mse.py` + `viz_export.py` and their dependency
closure). Nothing here is imported by any file still in `3_global_tractography/`
— that was checked before the move, and re-checked after it, along with an
import test of every remaining entry point.

**The findings are not archived, only the code that produced them.** Every
result these scripts established is written up in the parent `README.md` and in
the project memory; archiving the script does not retract the result. If you
need to *re-derive* one, the script is here and still runs — it reaches its
dependencies through the same `sys.path.insert` the rest of the stage uses, and
those dependencies all stayed in the parent folder.

| script | what it established |
|---|---|
| `objective_search.py` | ranked ten candidate objectives by the TRUE angular error at each one's own optimum. The source of the `align` result: alignment alone hits the oracle (3.22°), every density term picks the no-turn cone, and scale-matching does **not** fix it |
| `phase3.py` | the joint global model — the thing that finally separated kissing from crossing where local rules failed |
| `kissing_gap_sweep.py` + `test_kissing_phantom.py` | global tractography stays clean down to a gap of ~5× the fibre radius; below that the phantom itself fuses (a resolution limit, not a bug) |
| `brain_phantom_pipeline.py` | whole-pipeline capstone, 5/5 at realistic SNR on an all-features phantom |
| `gradient_surface.py` | the 12×12 (L, θ) loss sheet: a high-loss ridge at low θ and a broad flat plateau — iteration 0 alone captures 91.5% of the improvement |
| `descent_mse.py` | the earlier `image_mse` descent, superseded by `descent_loop.py`. Its standing result — raw MSE 0.327 against a `mean(I²)` of 0.314, Pearson r = +0.009 — is what motivated the [0,1] normalisation in `tdi_mse.py` |
| `descent_curve.py` | the curved-phantom descent study that preceded it |
| `s_calibration.py` | superposition `S` threshold calibration |
| `loss_surface.py`, `loss_sensitivity.py` | early loss-landscape studies, superseded by `param_volume.py` and the sweep in `descent_loop.py` |
| `param_volume_figure.py` | figures for `param_volume.py`; its `iso_shell` construction is the ancestor of `viz_export.descent_volume_3d_png` |
| `field_before_after.py`, `toy_demo.py` | figure and demo scripts |

## What deliberately did NOT move

`peak_straightness.py` and `fibre_continuity.py` are unrelated to the loop but
stayed in the parent folder, because five scripts in `mass_conservation/` import
them — the one-way downstream→upstream dependency the parent README documents.
Archiving them would have broken that stage silently.

`run_pipeline.py` stayed as the end-to-end Stage 1→2→3 entry point, which drags
`dense_optimise.py`, `field_export.py`, `render_field.py` and `field_refine.py`
with it, and the two standing gates `test_frame_convention.py` and
`single_fibre_check.py` stayed so the frame-convention and follow-e3 checks
still run.
