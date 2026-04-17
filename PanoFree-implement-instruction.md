# 《PanoFree 复现计划（聚焦 Full Spherical Panorama）》

## 1. 复现目标与边界

- **复现目标**：只复现 `PanoFree` 的**方法链路**，不复现实验设置、指标、用户研究与大规模对比。
- **复现范围**：集中在 `Full Spherical Panorama`，但**必须承认它依赖一个先生成好的 360° panorama 作为中间结果**，因此 360° 中段不是可选项，而是 full spherical 的前置子任务。
- **复现原则**：`最小可运行 -> 功能补全 -> 贴近论文完整方法 -> 再考虑效果优化`
- **推荐策略**：先做**文本到 full spherical**；先不做“任意预训练 T2I 模型兼容”，先锁定 `Stable Diffusion 2.0 / 2-inpainting + Diffusers`。

---

## 2. 总体判断

### 结论
`PanoFree` 的方法复现 **可行**，而且比很多需要训练的 panorama 方法更适合工程复现；但如果目标是“正确+完整复现”，难点不在主干框架，而在**若干关键超参数和补充细节论文没有说透**。

### 为什么可行
1. 主框架是传统而清晰的 `warp -> mask -> inpaint -> stitch/merge`。
2. 论文明确说实现基于 `Diffusers + PyTorch + SD v2.0 generation/inpainting`。
3. full spherical 的补充流程图给出了比较明确的阶段划分：  
   `先 360° -> 再上下扩展 -> 再上下极点闭合`。

### 为什么不能一步到位
1. **full spherical 本身不是独立模块**，它建立在“360° pano 已经可用”之上。
2. `cross-view guidance / risky area / semantic-density tuning` 这几个模块虽然思路明确，但**关键数值与实现细节并不完整**。
3. 补充材料说明了流程，但没有给出一套足够可直接照抄的参数表。

---

## 3. 方法拆解与复现优先级

## M0. 环境与基础依赖
**目标**：搭出能稳定跑文本生成与 inpainting 的最小环境。

- **建议实现**
  - Python + PyTorch
  - HuggingFace Diffusers
  - Stable Diffusion 2.0 / SD 2 Inpainting
  - OpenCV / PIL / torchvision
- **现成度**：很高
- **外部依赖**：成熟
- **难点**：几乎没有
- **是否必须**：必须

---

## M1. 透视视图 warping 与 mask 生成
**目标**：从当前视图生成下一视图的 warped image 和 unknown mask。

- **输入**
  - 当前视图图像 `x_i`
  - 视角变换 `P_i^{i+1}`（本质是 pitch/yaw 变化）
- **输出**
  - warped image `x̂_i`
  - mask `m_i`
- **建议实现**
  - 先不追求论文级通用 correspondence 建模
  - 先做基于球面/equirectangular 坐标的几何映射
  - 实现：
    1. perspective view <-> spherical/equirectangular
    2. 根据下一视角采样已有内容
    3. 空洞位置记为 inpaint 区域
- **现成度**：中等
- **成熟代码参考**
  - panorama/perspective 投影代码很多，可自写
  - 不依赖训练
- **难点**
  - 坐标系统与视角定义容易写错
  - 上下扩展与极点附近畸变更明显
- **是否必须**：必须

---

## M2. 360° central panorama 最小版
**目标**：先得到一个能闭环的 360° panorama，哪怕质量一般。

- **建议先做最小版**
  - 单向 sequential generation
  - 固定 yaw 方向若干步
  - 每步：
    `warp -> inpaint -> 拼回全景画布`
- **这一版先不加**
  - cross-view guidance
  - risky area erasing
  - bidirectional merge
- **验收标准**
  - 能从 prompt 生成完整 360°
  - 左右闭环可见，但允许有明显缝和风格漂移
- **现成度**：中等
- **难点**：闭环处最容易崩
- **是否必须**：必须（full spherical 前置）

---

## M3. Bidirectional Generation + Symmetric Guidance（先用于 360°）
**目标**：复现 PanoFree 的核心增益模块之一，先把 360° 做稳。

- **论文核心**
  - 将单向路径拆成双向路径
  - 最终在 `(pitch=0°, yaw=180°)` 处 merging
  - 当前视图除依赖上一视图外，还用“另一条路径上的对称视图”作 guidance
- **建议实现**
  - 左右两条 yaw 对称路径
  - 对称索引规则固定写死
  - guidance 先只支持 1 张参考图
  - 使用类似 SDEdit / img2img 风格的 guided inpainting
- **最小实现**
  - 先只做 360°，不要立即接 full spherical
- **现成度**：中等
- **外部依赖**
  - Diffusers 能做
  - 不需要额外训练
- **难点**
  - guidance 图如何与 warped 图融合
  - 双路径合并时如何避免 tearing
- **论文说明充分度**
  - 思路清楚
  - 但具体的 guidance 选择策略、融合细节、参数并不完整
- **是否必须**：对“正确复现”几乎必须

---

## M4. Cross-view Guidance（SDEdit 风格）
**目标**：纠正 sequential generation 的 style/content drift。

- **论文给出的形式**
  - 用过往视图中的一张 `x_g` 作为 guidance
  - 与当前 warped 图按 mask 混合后，再进入 guided synthesis
- **建议实现**
  - 用 `img2img/inpaint` 的 strength 模拟论文里的 `t0`
  - 先做：
    `x̂_g = m_i * x_g + (1-m_i) * x̂_i`
  - 然后喂给 inpainting / img2img
- **最小实现**
  - 只在 360° 阶段开启
  - full spherical 阶段沿用同一接口
- **现成度**：较高
- **外部依赖**：成熟
- **难点**
  - `t0/strength`、CFG、noise 注入强度没被完整写清
  - 论文只说 `t0 ∈ [0.9, 1.0)`，但不同任务最佳值不同
- **是否必须**：强烈建议尽早加入  
  > 从消融看，它是贡献最大的模块之一。

---

## M5. Risky Area Estimation & Erasing
**目标**：减少 artifact propagation。

- **论文拆成两层**
  1. **基于距离/边缘** 的风险估计  
  2. **基于颜色/平滑度** 的风险估计
- **建议工程顺序**
  - 先实现 `Distance + Edge`
  - 暂时不做 `Color + Smoothness`
- **原因**
  - 消融显示前者收益更明显，后者增益较小
  - 论文自己也承认 color/smoothness 受低层噪声影响更大
- **最小实现**
  - 当前视图生成完后，维护一个 risk map
  - 对下一步 warped 结果做 remask
  - 对 remask 做 Gaussian/median 平滑
- **现成度**：中等
- **难点**
  - `w` 组合权重、阈值函数 `Mr`、滤波核大小，论文没有给定
  - 需要自行调参
- **是否必须**
  - 对“能跑”不是必须
  - 对“接近论文效果”很重要

---

## M6. Full Spherical：上下扩展
**目标**：从 360° pano 向上、向下扩展到更高 pitch 区域。

- **补充材料给出的阶段**
  1. 已有 360° pano
  2. 先 warp 到 `(ϕ, 0°)` 与 `(-ϕ, 0°)` 生成上下扩展初始视图
  3. 分别沿 pitch 方向继续扩展
- **建议实现**
  - 先固定一个人工 `ϕ`（如 45° 或 50°）做工程近似
  - 上下两个方向完全复用同一套代码
- **现成度**：中等
- **难点**
  - `ϕ` 没写明
  - 扩展步数、每步 pitch 增量、FOV 选择未明确
  - 极区畸变会放大 artifact
- **是否必须**：必须（这是 full spherical 的主体）

---

## M7. Full Spherical：上下极点闭合
**目标**：生成 top / bottom pole，闭合整个球面。

- **补充材料给出的阶段**
  - 最终 warp 到 `(90°, 0°)` 与 `(-90°, 0°)`，对未知区域做 inpainting
- **建议实现**
  - 将上下扩展结果先拼回完整 equirectangular
  - 再从 pole-centered perspective view 做最后一次修补
- **现成度**：中等
- **难点**
  - 极点附近最容易出现 hallucination、拉伸、重复纹理
  - 若没有 scene structure prior，会明显崩
- **是否必须**：必须

---

## M8. Scene Structure Prior（full spherical 专属）
**目标**：抑制“天空里长出城市 / 水下出现地面建筑”这类 hallucination。

- **论文思路**
  - 从初始视图抽取 scene prior
  - 例如 upward expansion 的首视图，使用初始图像的**上 1/3 区域**作为 guidance
  - 通过**降低 guidance scale、增大 FOV、调初始噪声方差**，让生成更服从结构先验而不是原始文本局部语义
- **建议实现顺序**
  1. 先只做“上 1/3 / 下 1/3 图像先验”
  2. 再做 guidance scale 降低
  3. 最后调噪声方差
- **现成度**：中等
- **难点**
  - 这是论文里**最像经验工程**的一块
  - guidance scale/FOV/noise variance 的具体数值没给
- **是否必须**
  - 对 full spherical 的“正确性”非常关键
  - 不加它，极区 hallucination 风险很高

---

## 4. 推荐复现路线（按迭代）

## Iter-0：能跑通的基础框架
- 完成 M0 + M1
- 验证单步 `warp -> mask -> inpaint`

## Iter-1：最小 360°
- 完成 M2
- 目标：先生成一个闭环但粗糙的 360° pano

## Iter-2：双向 360°
- 完成 M3
- 目标：把 360° 的闭环和整体一致性拉起来

## Iter-3：加入 cross-view guidance
- 完成 M4
- 目标：减少风格漂移、重复物体、远距离内容突变

## Iter-4：先做 full spherical 骨架
- 完成 M6 + M7，但先**不加 risky area / scene prior**
- 目标：流程上得到完整球面

## Iter-5：加入 scene structure prior
- 完成 M8
- 目标：先解决“上下极区 hallucination”

## Iter-6：加入 risky area（先 Distance + Edge）
- 完成 M5 的主干
- 目标：减少扩展阶段的 artifact propagation

## Iter-7：补 color/smoothness risk
- 只在前面都稳定后再做
- 这是“完整复现”而不是“最小可运行”的内容

---

## 5. 可行度判断

## 5.1 有现成/成熟代码参考的部分
- Diffusers + SD2.0 / SD2-inpainting
- img2img / inpaint 风格的 SDEdit 替代实现
- panorama / perspective 的投影与反投影
- mask 后处理（Gaussian / median / threshold）
- equirectangular stitching

**判断**：这些都属于工程工作量，不属于研究性阻塞点。

## 5.2 需要自写，但难度可控的部分
- 双向路径与对称 guidance 调度
- risk map 的维护与 remask
- full spherical 上下扩展/极点闭合调度
- scene prior 注入

**判断**：难，但都是“能靠试验补全”的工程问题。

## 5.3 难以严格复现、论文说明不足的部分
1. `ϕ` 的具体取值  
2. upward/downward 扩展的 exact step schedule  
3. guidance image 的精确选择策略  
4. risky area 的各项权重、阈值、滤波参数  
5. scene structure prior 中 guidance scale / FOV / noise variance 的具体数值  
6. full spherical 阶段 prompt 细节与 prompt engineering 规则  
7. 不同阶段是否使用同一 inpainting 配置，论文没有给完整参数表

**判断**：  
这意味着你更适合做的是**“方法等价复现”**，而不是“位级别复现论文结果”。

---

## 6. 复现难点排序

### 最难
1. **full spherical 的上下扩展与极点闭合**
2. **scene structure prior 的参数化落地**
3. **risk map 的实际有效阈值设计**

### 中等
4. **双向路径 + symmetric guidance**
5. **cross-view guidance 与 inpainting 的融合**

### 最容易
6. **基础 warping / stitching / mask**
7. **Diffusers 接入**
8. **vanilla sequential baseline**

---

## 7. 最小可运行版本定义（建议）

## MVP-A：方法骨架版
- M0 + M1 + M2 + M6 + M7
- 不做 CG / risky area / scene prior
- 结果：能产出 full spherical，但 artifact 和 hallucination 会很多

## MVP-B：可用版
- MVP-A + M3 + M4 + M8
- 结果：基本具备 PanoFree 的主要思想，full spherical 可看

## MVP-C：接近论文版
- MVP-B + M5
- 结果：更接近论文完整方法

---

## 8. 我的建议

如果目标是“最短时间拿到一个像样的 PanoFree Full Spherical 复现”，最合理路线是：

1. **先把 360° 中段做好**，不要一开始就碰极点；
2. **优先做 cross-view guidance**，因为它是最核心、最确定有收益的模块；
3. **full spherical 先加 scene structure prior，再加 risky area**；
4. `Color/Smoothness risk` 放到最后，它不是第一优先级；
5. 接受现实：你大概率做出的会是**工程等价复现**，而不是论文作者内部参数的严格复刻。

---

## 9. 一句话结论

`PanoFree` 很适合按“子模块迭代”方式复现；  
**最关键的前置不是 full spherical 本身，而是先做稳 360° central panorama；最关键的增强不是 risky area，而是 cross-view guidance + scene structure prior。**
