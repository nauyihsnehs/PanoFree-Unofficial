# PanoFree Reproduction Plan
**Target:** method-level reproduction only  
**Scope:** focused on **Full Spherical Panorama** generation  
**Principle:** iterative decomposition of subtasks/modules, from **minimum runnable** to **correct and complete** reproduction

---

## 1. Objective

This project aims to reproduce the **methodological core** of **PanoFree** for **full spherical panorama generation**, without reproducing the original experimental setup, benchmark protocol, or reported metrics.

The reproduction target is therefore:

- to recover the **pipeline logic**
- to recover the **module interactions**
- to recover the **key behaviors** claimed by the method
- to produce a **working full-sphere generation system** whose outputs qualitatively reflect the intended design of the paper

This is **not** a paper-level reproduction in the strict benchmark sense. It is a **method reconstruction and implementation project**.

---

## 2. Reproduction Boundary

### Included
- sequential warping + inpainting framework
- bidirectional generation with symmetric guidance
- cross-view guidance via image-guided synthesis
- risky area estimation and erasing
- full spherical expansion from a 360° panorama
- upward/downward expansion and pole closure
- scene-structure-aware hallucination mitigation for top/bottom regions

### Excluded
- exact dataset reconstruction
- quantitative evaluation pipeline
- user study
- metric replication
- paper-level hyperparameter search for best scores
- extensive adapter/model swapping experiments

---

## 3. Reproduction Standard

The implementation should satisfy the following standard:

### Minimum standard
A working pipeline that:
1. starts from an initial view,
2. generates a 360° panorama,
3. expands it toward upper and lower regions,
4. closes the sphere at the north and south poles.

### Intermediate standard
The system explicitly implements:
- cross-view guidance,
- risky area erasing,
- symmetric bidirectional loop-closure logic,
- full-sphere hallucination control.

### Final standard
The reproduced system should preserve the **intended algorithmic structure** of PanoFree, even when some low-level implementation details must be reasonably approximated.

---

## 4. Recommended Reproduction Strategy

## Stage A — Build the minimum runnable baseline first

Start from the simplest viable system:

1. **Single-view initialization**
   - Generate or load the initial perspective view.
   - Fix a camera convention and projection convention early.

2. **Vanilla sequential warping + inpainting**
   - Implement view warping between adjacent viewpoints.
   - Use a pretrained text-guided inpainting model.
   - Do **not** add any guidance or risky-area logic yet.

3. **360° panorama first**
   - Reproduce only the horizontal panorama path first.
   - Ensure that the viewpoint schedule, warping, masking, and stitching are all stable.

4. **Then extend to full spherical panorama**
   - Use the generated 360° panorama as the center band.
   - Expand upward and downward separately.
   - Finally inpaint the two poles.

This order is important: **full-sphere reproduction should not begin before the 360° center band is stable**.

---

## 5. Canonical Pipeline to Reproduce

PanoFree’s full spherical generation can be reconstructed as the following method sequence:

### Phase 1 — Generate the 360° central band
- Initialize at the front view centered at `(pitch=0°, yaw=0°)`.
- Use **bidirectional generation with symmetric guidance** along the yaw axis.
- Merge the two horizontal paths at the back view to ensure loop closure.

### Phase 2 — Upward and downward expansion
- From the completed 360° panorama, warp to `(pitch=+φ, yaw=0°)` and `(pitch=-φ, yaw=0°)` to create initial views for vertical expansion.
- Apply the same PanoFree logic separately along the positive-pitch and negative-pitch directions.

### Phase 3 — Pole closure
- Warp to `(pitch=+90°, yaw=0°)` and `(pitch=-90°, yaw=0°)`.
- Inpaint the remaining unknown areas to close the sphere.

This should be treated as the **reference pipeline definition** for the reproduction.

---

## 6. Module Decomposition

## M0. Geometry and image-space infrastructure
### Goal
Build the non-learning infrastructure required by every later module.

### Deliverables
- camera parameter definition
- spherical / equirectangular / perspective conversion utilities
- viewpoint scheduler
- warping operator
- mask generation
- stitching / accumulation utilities
- visualization tools for debugging masks and warped views

### Reproduction feasibility
**High.** This is engineering-heavy but conceptually straightforward.

### Risk
Projection conventions can silently break the entire system. This module must be validated very early.

---

## M1. Vanilla sequential generation baseline
### Goal
Implement the paper’s underlying sequential warping-and-inpainting process before any correction module.

### Deliverables
- iterative generation from one view to the next
- text-conditioned inpainting
- panorama accumulation
- baseline 360° and full-sphere outputs

### Why this matters
This baseline is not only a starting point; it is also needed for ablations and debugging. PanoFree is explicitly built as a correction of **deficient conditions** in vanilla sequential generation.

### Reproduction feasibility
**High.**

### Expected issues
- style drift
- repeated objects
- image tearing
- hallucinations near top/bottom
- severe artifact propagation

These failures are expected and should be treated as evidence that the baseline is functioning as intended.

---

## M2. Cross-view guidance
### Goal
Correct **biased conditions** by conditioning the current generation on more than one prior view.

### Core idea
Instead of conditioning only on the immediately previous view, introduce a guidance image from earlier generated views. PanoFree instantiates this with **SDEdit-style guided image synthesis**.

### Minimum implementation
- select one guidance image from previous views
- paste it into masked regions of the warped image
- run image-guided diffusion/inpainting with sufficiently large noise level so that the guidance acts as a bias, not a hard copy

### Reproduction priority
**Very high.** This is one of the most important modules and, according to the paper’s own ablation, the strongest contributor among the correction modules.

### Reproduction feasibility
**Moderate to high.**
The method idea is clear, but implementation depends on how image-guided editing is integrated with the chosen diffusion/inpainting stack.

### Engineering recommendation
Implement a simplified but explicit version first:
- one guidance image only
- fixed guidance-selection rule
- fixed noise level / strength
- no adaptive policy in the first version

---

## M3. Risky area estimation and erasing
### Goal
Correct **noisy conditions** by erasing regions likely to propagate artifacts before the next inpainting step.

### Risk sources described by the paper
- distance from the initial view
- proximity to image edges
- abrupt color change
- abrupt smoothness / gradient change
- jagged mask edges
- sharp mask boundaries

### Minimum implementation
Implement in the following order:

1. **distance-based risk**
2. **edge-based risk**
3. **mask smoothing / anti-aliasing**
4. optional color-based risk
5. optional smoothness-based risk

### Reproduction priority
**High**, but incremental.

### Engineering recommendation
Do **not** implement all risk terms at once.
The most stable order is:
- distance + edge first
- then mask cleanup
- only then color / smoothness heuristics

### Reproduction feasibility
**Moderate.**
The logic is clear, but low-level details are heuristic-heavy.

### Why this is not fully straightforward
The paper describes the priors and formulas at a high level, but practical thresholding, weighting, filtering radii, and remasking behavior still require implementation choices.

---

## M4. Bidirectional generation with symmetric guidance
### Goal
Correct **partial conditions** and reduce long-range drift by generating from two symmetric directions and merging them.

### Core behavior
- split one long generation path into two symmetric paths
- generate from both directions
- use the symmetric view from the opposite path as guidance
- merge at a closing view

### Reproduction priority
**Very high** for 360° generation, because the full spherical pipeline depends on a stable central band.

### Reproduction feasibility
**Moderate.**
The conceptual design is clear, but the exact scheduling policy and symmetric correspondence bookkeeping require careful implementation.

### Engineering recommendation
Treat this as a scheduling/control module, not an image model module.
The main failure mode is not image quality but **incorrect synchronization of opposite paths**.

---

## M5. Full spherical expansion logic
### Goal
Extend the stable 360° center band into a complete sphere.

### Minimum implementation
- generate the 360° center band first
- create initial upward and downward expansion views at `±φ`
- run vertical expansion independently for the upper and lower branches
- close the top and bottom poles at `±90°`

### Reproduction priority
**Essential**, but only after M1–M4 are stable on the center band.

### Reproduction feasibility
**Moderate.**
The overall structure is stated clearly, but the paper does not fully specify the optimal `φ`, vertical scheduling density, or whether all reuse policies are identical to the horizontal case.

### Engineering recommendation
Start with a fixed `φ` and a minimal number of vertical steps.
Do not optimize coverage density before the pipeline is stable.

---

## M6. Scene-structure-aware hallucination mitigation
### Goal
Reduce top/bottom hallucinations caused by reusing the same prompt in regions where scene priors change drastically.

### Core idea
Use the initial image as a source of **scene-structure prior** and weaken prompt dominance during expansion and closure:
- extract prior visual content from the initial view
- reduce guidance scale when moving toward top/bottom
- enlarge field of view
- adjust initial noise variance

### Reproduction priority
**High** for full spherical panorama specifically.

### Reproduction feasibility
**Moderate to low.**
The idea is clear, but the paper gives limited operational detail on exact parameter schedules.

### Engineering recommendation
Implement a simplified first version:
- upper branch uses upper crop of the initial view as guidance
- lower branch uses lower crop analogously if appropriate
- use manually tuned reduced text guidance for pole closure

---

## 7. Implementation Order

A practical order is:

1. **M0** Geometry/projection/warping infrastructure  
2. **M1** Vanilla sequential 360° baseline  
3. **M4** Bidirectional generation with symmetric guidance  
4. **M2** Cross-view guidance  
5. **M3** Risky area estimation and erasing  
6. **M5** Full spherical expansion  
7. **M6** Hallucination mitigation for top/bottom  
8. Final cleanup, stabilization, and code refactoring

This order is preferable because it isolates failures:
- if M1 fails, geometry or inpainting is broken
- if M4 fails, scheduling or path symmetry is broken
- if M2/M3 fail, the correction modules are unstable
- if M5/M6 fail, the vertical extension logic is under-specified or weak

---

## 8. Feasibility Assessment by Module

| Module | Feasibility | Main reason |
|---|---:|---|
| Geometry / projection / warping | High | standard engineering problem |
| Vanilla sequential pipeline | High | directly reproducible with existing inpainting stacks |
| Bidirectional generation | Moderate | requires careful path design and synchronization |
| Cross-view guidance | Moderate-High | concept is clear, integration with diffusion stack needs engineering |
| Distance / edge risky erasing | Moderate | heuristic but well-motivated |
| Color / smoothness risky erasing | Moderate-Low | more sensitive, lower-level, noisier heuristics |
| Full spherical vertical expansion | Moderate | pipeline is clear, scheduling details are not fully specified |
| Hallucination mitigation for top/bottom | Moderate-Low | conceptually clear but under-specified in operational detail |
| Exact paper-faithful hyperparameters | Low | supplementary-dependent and partly heuristic |

---

## 9. Existing / Mature Code References

## Strong references
These are suitable as direct engineering bases:
- **Diffusers** for Stable Diffusion-based text generation and inpainting
- official or standard **Stable Diffusion inpainting pipelines**
- **SDEdit** as the conceptual and practical reference for image-guided editing
- standard image processing libraries for mask filtering, Gaussian smoothing, and median filtering

## Useful adjacent references
These are not direct drop-in reproductions of PanoFree, but can provide implementation ideas:
- **L-MAGIC / MMPano** for warping + inpainting panorama pipelines
- **Text2Room** for sequential view generation and geometry-aware iteration logic
- existing panorama projection / cubemap / equirectangular conversion utilities from public vision or graphics codebases

## Weak reference
- **PanoFree official repository** should not be assumed to be a usable reproduction base unless the actual implementation is fully available and complete. It should be treated as a possible pointer, not as a guaranteed dependency.

---

## 10. External Dependencies

A realistic implementation will likely depend on:

### Core generative stack
- PyTorch
- Diffusers
- a Stable Diffusion checkpoint for text generation
- a Stable Diffusion inpainting checkpoint

### Image processing / geometry
- NumPy
- OpenCV
- PIL
- SciPy or equivalent for filtering/interpolation
- custom spherical projection utilities

### Optional but likely useful
- xFormers or attention acceleration
- mixed precision / memory optimization utilities
- segmentation or saliency tools only if debugging is needed
- experiment logging / visual debugging tools

### Possibly required if aiming for closer faithfulness
- explicit SDEdit integration rather than an approximate image-guided inpainting substitute

---

## 11. Hard-to-Reproduce Parts

The following parts are likely to be the hardest:

### 11.1 Exact guidance scheduling
The paper explains the role of cross-view guidance, but the exact practical policy for:
- selecting the guidance image,
- setting the guidance strength,
- choosing noise level,
- varying this across panorama types,
is not fully operationalized.

### 11.2 Risk fusion details
The risky-area module is described by multiple risk terms, but exact:
- weighting coefficients,
- thresholds,
- filtering sizes,
- remasking schedule,
are still engineering choices.

### 11.3 Full spherical vertical policy
The paper clearly states the high-level full-sphere procedure, but the exact:
- pitch step size,
- number of vertical steps,
- whether horizontal and vertical branches share identical settings,
- best pole-closure configuration,
are not fully pinned down.

### 11.4 Hallucination-control schedule
The idea of using scene priors from the initial image and changing guidance scale / variance is clear, but exact schedules are not standardized in the paper.

### 11.5 Stitching policy and per-pitch resolution behavior
Full-sphere generation depends strongly on how viewpoint FOV and sampling density vary with pitch. This is method-critical but not exhaustively specified.

---

## 12. Under-Specified Points in the Paper

The following should be treated as **explicit implementation assumptions** in your codebase and documentation:

1. exact viewpoint schedule for full spherical generation  
2. exact pitch-step parameter `φ` and its tuning strategy  
3. exact guidance-image selection rule under all panorama modes  
4. exact noise level / strength schedule for SDEdit-based guidance  
5. exact weights for distance / edge / color / smoothness risks  
6. exact thresholding and mask post-processing details  
7. exact handling of top/bottom branch prompt modification  
8. exact stitching / blending strategy for vertical completion outputs  

These assumptions should be written down clearly so that the project remains reproducible even if it is not paper-identical.

---

## 13. Recommended Development Milestones

## Milestone 1 — Minimum runnable 360°
- warping works
- masks are correct
- inpainting runs end-to-end
- central 360° panorama can be closed

## Milestone 2 — Stable 360°
- bidirectional generation works
- symmetric guidance works
- severe loop-closure tearing is reduced

## Milestone 3 — Corrected 360°
- cross-view guidance is active
- risky area erasing is active
- visible artifact propagation is reduced

## Milestone 4 — Minimum full sphere
- upward/downward expansion runs
- pole closure runs
- output covers the full spherical domain

## Milestone 5 — Full method reproduction
- hallucination mitigation is active
- top/bottom failures are reduced
- implementation structure matches the paper’s intended design

---

## 14. Practical Success Criteria

Since this is a method-level reproduction, success should be judged by the following:

- The code structure reflects the paper’s module decomposition.
- The system generates a complete full spherical panorama.
- Cross-view guidance measurably improves stability over the vanilla baseline.
- Risky area erasing reduces obvious artifact propagation.
- Symmetric bidirectional generation improves loop closure.
- The top/bottom regions are meaningfully better than vanilla sequential expansion.
- All paper-dependent assumptions are documented explicitly.

---

## 15. Final Recommendation

The most reliable strategy is to treat this reproduction as a **controlled engineering reconstruction** rather than an exact paper-faithful clone.

In practice, the project should be framed as:

> **“Reproduce the algorithmic structure and qualitative behavior of PanoFree for Full Spherical Panorama generation, using stable modern diffusion tooling and explicitly documented implementation assumptions where the paper is under-specified.”**

That framing is both technically realistic and methodologically sound.

