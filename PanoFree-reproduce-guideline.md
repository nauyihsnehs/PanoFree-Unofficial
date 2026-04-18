# Reproduce Guideline: PanoFree (Full Spherical Panorama Only)

You are reproducing **PanoFree** for **method comparison**, not for best possible output quality.  
Your goal is to implement the **original method as faithfully as possible**, using only the **original paper and its supplementary material** as the methodological source of truth.

## 1. Scope

Reproduce **only** the **Full Spherical Panorama** pipeline of PanoFree.

### Included

- the inference-time method,
- view scheduling,
- warping,
- mask generation,
- inpainting orchestration,
- cross-view guidance,
- risky-area estimation and erasing,
- bidirectional / symmetric guidance for the 360° center band,
- upward / downward expansion,
- top / bottom pole closure,
- final spherical stitching.

### Excluded

- experiments,
- metric replication,
- user studies,
- ablations beyond what is needed for debugging,
- improvements beyond the original method,
- any optimization whose purpose is to outperform the paper.

---

## 2. Hard Constraints

1. **Follow the original PanoFree paper and supplementary only.**
2. **Ignore all later or external variants**, including any user-authored follow-up methods or reinterpretations.
3. **Do not redesign the method for stronger results.**
4. When the paper is underspecified, use the **smallest possible engineering completion** with **common conservative defaults**.
5. **Do not add extra prompt engineering** beyond what is explicitly required by the original method.
6. Use **one single global prompt throughout the whole pipeline**.
7. Use **standard engineering utilities only** where the paper leaves low-level implementation open.

---

## 3. Overall Pipeline

Implement the Full Spherical Panorama pipeline in this exact order:

1. Generate the **central 360° panorama band**.
2. Expand **upward** from the central band.
3. Expand **downward** from the central band.
4. Generate the **top pole-centered close-up**.
5. Generate the **bottom pole-centered close-up**.
6. Stitch all generated views into a full spherical panorama.

Do **not** reorder these stages.

---

## 4. Backend and Basic Representation

Use the following implementation choices:

- Framework: **Diffusers**
- Backbone models: **Stable Diffusion v2.0 generation model** and **Stable Diffusion v2.0 inpainting model**
- Per-view image resolution: **512 × 512**
- Spherical panorama canvas: **2048 × 4096** equirectangular
- Number of diffusion steps: **50**
- Warping primitive: **`cv.warpPerspective`**
- Warp border handling: fill unknown regions with black / empty values and use the generated mask to define inpainting regions

Do not replace these with newer models unless the exact original checkpoint is unavailable.

---

## 5. Central 360° Band

Implement the central 360° stage as bidirectional generation around the initial view.

### Camera / scheduling

- Start from the initial perspective view at **(pitch = 0°, yaw = 0°)**.
- Use **FoV = 80°**.
- Use **yaw stride = 40°**.
- Generate the horizontal band by stepping in both directions from the initial view.
- After symmetric left/right expansion, generate one **merging / loop-closure view** centered at **yaw = 180°**.

### Guidance

- Use **symmetric guidance whenever the symmetric counterpart exists**.
- If no symmetric counterpart exists, **fall back to the previous generated view**.

### Cross-view guidance

Implement SDEdit-style guidance as described by PanoFree:

- combine the warped image with the selected guidance image over the inpainting mask,
- run denoising from a partially noised guided state.

Use:

- `t0 = 0.98`

### Risk weights for this stage

Implement all four risk terms:

- distance,
- edge,
- color,
- smoothness.

But for the **central 360° stage**, use:

- `w = [0.8, 0.2, 0.0, 0.0]`

### Remasking

- Erase **5% of the known area per step** for this stage.

---

## 6. Upward and Downward Expansion

After the 360° center band is complete, expand to the upper and lower spherical regions.

### Camera / scheduling

- Keep the same horizontal center-band support over `pitch ∈ [-40°, 40°]`.
- Expand upward and downward using an additional **25° pitch offset**.
- Use **FoV = 110°**.
- Use **yaw stride = 80°**.
- Use **3 expansion steps upward** and **3 expansion steps downward**.

### Prior image for semantic / structure alignment

Use prior-image guidance exactly in the original spirit:

- For **upward expansion**, use the **upper 1/3** crop of the relevant reference image.
- For **downward expansion**, use the **lower 1/3** crop of the relevant reference image.
- Continue this same rule for later upward/downward stages.
- Use crop ratio as a configurable parameter, but default to **1/3**.

### Reference selection rule

- If a symmetric counterpart exists, use it.
- Otherwise use the **previous generated view**.

### Guidance / noise settings

For both upward and downward expansion:

- `t0 = 0.90`
- risk weights: `w = [0.6, 0.2, 0.1, 0.1]`
- guidance scale: **2.0**
- initial noise variance multiplier: **1.05**

### Remasking

- Erase **10% of the known area per step** during expansion.

### Prompt policy

- Use the **same global prompt** as the whole pipeline.
- Do **not** add special upward / downward prompt rewrites.
- The original method’s semantic stabilization should come from:
    - prior image crops,
    - reduced guidance scale,
    - widened FoV,
    - slightly increased noise variance.

---

## 7. Top and Bottom Pole Close-Up

After upward and downward expansion, generate the pole-centered close-up views.

### Pole viewpoints

- Top pole center: **(pitch = 90°, yaw = 0°)**
- Bottom pole center: **(pitch = -90°, yaw = 0°)**

### FoV and denoising

- Use **FoV = 90°**
- Use `t0 = 0.90`
- Use risk weights: `w = [0.6, 0.2, 0.1, 0.1]`
- Use guidance scale: **1.0**
- Use initial noise variance multiplier: **1.10**

### Prior image for pole closure

- For the **top pole**, use the **upper 1/3** of the nearest upward-expansion result.
- For the **bottom pole**, use the **lower 1/3** of the nearest downward-expansion result.
- Default crop ratio remains **1/3**.

### Remasking

- Erase **20% of the known area per step** during final pole close-up generation.

### Restrictions

- Do **not** add any pole-specific trick that is not in the original method.
- Do **not** add custom pole prompt engineering.
- Do **not** add extra stabilization modules.

---

## 8. Risky-Area Estimation and Erasing

Implement **all four** risky-area components:

1. **distance-based risk**
2. **edge-based risk**
3. **color-based risk**
4. **smoothness-based risk**

Follow the original method structure:

- compute each risk map,
- combine them linearly with stage-specific weight vector `w`,
- warp the previous risk map into the current view,
- remask risky regions,
- smooth the final remasked inpainting mask before inpainting.

### Conservative default values for underspecified low-level operators

If the paper does not specify exact kernel sizes or thresholds, use these defaults:

- Gaussian smoothing for risk maps: `kernel = 9`, `sigma = 2`
- Median filtering for jagged-mask cleanup: `kernel = 5`
- Risk threshold inside warped valid area: `0.5` after min-max normalization to `[0, 1]`
- Final mask smoothing: Gaussian blur + threshold back to binary

These defaults are only for low-level completion. Do not tune them for better results unless there is a clear implementation bug.

---

## 9. Warping

Use `cv.warpPerspective` for view-to-view warping.

### Rules

- Use perspective warping between adjacent target views.
- Unknown / invalid regions must remain empty and be represented by the inpainting mask.
- Do not introduce custom anti-aliasing or specialized resampling logic beyond standard OpenCV usage.
- Keep the implementation simple and standard.

---

## 10. Stitching and Blending

Use a **standard feather blending** approach only.

### Rules

- In overlap regions, blend using weights proportional to distance from image / valid-region boundaries.
- In non-overlap regions, directly copy valid pixels.
- Do not use Poisson blending, multiband blending, learned harmonization, or seam optimization.

This stage should be treated as standard engineering, not as a research contribution.

---

## 11. What Not to Add

Do **not** add any of the following:

- LLM / VLM prompt decomposition
- view-specific prompt rewriting
- extra style-transfer modules
- extra pre-inpainting stages
- adaptive trajectory optimization beyond the original schedule
- novel pole stabilization heuristics
- external geometry priors not used in PanoFree
- changes motivated by improving quality instead of reproducing the method

---

## 12. Required Outputs

Your implementation must save:

1. initial input view,
2. every warped view,
3. every inpainting mask,
4. every risk map component,
5. every combined risk map,
6. every guidance image used,
7. every generated intermediate view,
8. the central 360° panorama,
9. the upward expansion results,
10. the downward expansion results,
11. the top and bottom pole close-ups,
12. the final stitched full spherical panorama.

This is necessary for debugging and for fair comparison against another method.

---

## 13. Reproduction Philosophy

Use this exact decision rule throughout the project:

- **If the original method specifies it, reproduce it.**
- **If the supplementary specifies it, reproduce it.**
- **If neither specifies it, use the most conservative standard implementation possible.**
- **Do not improve the method.**
- **Do not silently replace weak original design choices with stronger alternatives.**

The final output should be a **faithful PanoFree reproduction**, not a better panorama generator.

## 14. Implementation Strategy:

Implement the reproduction in **at most 5 phases**.  
Each phase must end in a **complete, runnable, testable intermediate result**.  
Do **not** leave a phase in a half-integrated state.

### General rules

- Each phase must produce a concrete artifact that can be visually inspected.
- A later phase may extend an earlier one, but must not break the earlier phase’s expected output.
- If a phase is incomplete, do not start the next phase.
- Prefer minimal, working integration over broad but partial implementation.

### Phase 1 — Core infrastructure

Implement:

- camera / spherical projection utilities,
- `cv.warpPerspective` view warping,
- mask generation,
- inpainting wrapper,
- debug save pipeline.

Expected result:

- given one source view and one target view, the system can:
    - warp the source image,
    - generate the missing-region mask,
    - run inpainting,
    - save all intermediate outputs.
- This phase is complete only if one-step warp + inpaint works end-to-end.

### Phase 2 — Central 360° band

Implement:

- the full horizontal 360° schedule,
- bidirectional generation,
- symmetric guidance when available,
- fallback to previous view otherwise,
- merging / loop-closure view,
- stitching of the central band.

Expected result:

- a complete **central 360° panorama band** is generated and stitched.
- This phase is complete only if the entire horizontal band is runnable end-to-end from one input view.

### Phase 3 — Risky-area estimation and erasing

Implement:

- all four risk terms:
    - distance,
    - edge,
    - color,
    - smoothness,
- stage-specific risk weighting,
- remasking,
- mask smoothing / cleanup.

Expected result:

- the central 360° band can now be generated **with full risky-area logic enabled**,
- all risk maps and remasked masks are saved and inspectable.
- This phase is complete only if risky-area erasing is fully integrated into the 360° pipeline, not partially stubbed.

### Phase 4 — Upward and downward expansion

Implement:

- upward expansion,
- downward expansion,
- upper / lower 1/3 prior extraction,
- stage-specific FoV, guidance scale, and variance settings,
- stitching of expanded regions into the spherical canvas.

Expected result:

- the system can generate the **full sphere except the top and bottom pole close-ups**.
- This phase is complete only if both upward and downward regions are generated end-to-end and stitched into the sphere.

### Phase 5 — Pole closure and final full-sphere output

Implement:

- top pole close-up,
- bottom pole close-up,
- prior extraction from upward / downward results,
- final blending / feather stitching,
- final spherical export.

Expected result:

- a complete **full spherical panorama** is produced from start to finish,
- with all intermediate outputs saved.
- This phase is complete only if the entire Full Spherical Panorama pipeline runs end-to-end without placeholder modules.

### Final requirement

At the end of each phase, produce:

1. runnable code,
2. saved intermediate outputs,
3. a short note stating whether the phase reached its expected result.

Do not mark a phase as complete if any core component in that phase is still a stub, placeholder, or partially connected.