# KDES：固定源码镜像的完整 Python 移植

2026-10-31，后续更新。此前作者站点404的问题仍存在，但已通过GitHub检索找到原作者署名源码的镜像，因此不再停留在只实现局部公式/等待软件的状态。

## 来源与许可

- 固定镜像：[flyfj/RGBDMaze，提交7973cb3](https://github.com/flyfj/RGBDMaze/tree/7973cb37319449c1dc76593a2973412d72ce5bc2/KernelDescriptor)。
- 独立镜像：[Watch-n-Patch Feature Extraction，提交e5898c0](https://github.com/Buzz-Beater/Watch-n-Patch-Feature-Extraction/blob/e5898c06d409382c89227ae14f6c1ea843dae622/feature/visual/KernelDescriptor/kdes/gradkdes_dense.m)。两份gradkdes_dense.m在统一CRLF后内容完全一致。
- 作者署名Liefeng Bo，后由Xiaofeng Ren优化；包中的BSD许可、署名、README及16份相关MATLAB源文件保留在 `third_party/kdes_matlab/`，本地SHA256与提交信息见PROVENANCE.json。
- `kdes_native.py`是这些源码的Python移植，保留来源声明；不是宣称由我们从零发明，也不是作者对该镜像的背书。其软件版本是否恰好是2015主论文实验使用的版本，仍无法确认。

## 已移植步骤

1. 默认basis生成：`gen_basis.m`/`gradkdes_basis.m`中的10×10梯度坐标和5×5空间坐标，共2500个joint basis；加权RBF参数[5,5,3,3]，中心化KPCA，200维。忠实保留镜像kpca.m中**不按特征值白化**的行为。
2. 局部提取：sigma=0.8的5×5高斯导数滤波、filter2相应零边界相关、1e-5幅度下限、contrast=0.8的范数下限、16×16块、步长8和居中起点；空间坐标除以16；保留MATLAB列主序的kernel拼接和1-based patch center。
3. 训练码本：每张训练图最多10个patch，1000词；随机初始化、固定50次Lloyd更新、空中心保留。使用可记录NumPy随机种子，无法重现作者未公开的MATLAB RNG状态。
4. 图像编码：RBF参数0.001，词核矩阵加1e-6 I，G=chol(inv(K))；每个空间格子按最大kernel匹配去除同词冗余patch，然后均值、G投影及分层权重。没有用普通BoW直方图代替EMK/CKSVD。
5. RGB的[1,2,4]层产生21000维；Depth的[1,2,3]层产生14000维，按主文V-C选择，覆盖SDK示例中不同任务的pyramid设置。
6. 可加载作者格式的gradkdes_params MAT文件，保留给定basis，避免必须重新生成KPCA的特征空间；也支持NPZ参数。

SDK镜像的方形100个orientation basis与2010参考论文中25个圆周basis的文字设置不同；新模块明确跟随固定SDK镜像。此前 `kdes.py`保留为原论文式(12)的显式参数数学参考，两者不混称逐字一致。

## 运行

先提供人工裁剪/预处理、最长边不超过300像素的图片，以及带sample_id、rgb_path、depth_path的CSV。所有训练/验证/测试的RGB可用，但只读取训练行的Depth；训练码本仅用训练行。灰度转换使用PIL L；若需要消除与MATLAB rgb2gray的细微差异，应先在统一环境提供已转换的灰度图。当前不会暗中选择不同的resize/眼点/人体裁剪规则。

行人数据的帧/身份选择可以在特征提取前完成：

```bash
python -m itml_repro.reid --stage select --manifest data/reid/raw_images.csv \
  --config configs/reid_protocol.json --output runs/reid_selection

python -m itml_repro.kdes_native \
  --manifest runs/reid_selection/all_selected.csv \
  --training-manifest runs/reid_selection/training_images.csv \
  --depth-unit millimeters_to_meters --output data/reid/kdes.npz

python -m itml_repro.reid --stage prepare \
  --manifest runs/reid_selection/all_selected.csv --features data/reid/kdes.npz \
  --config configs/reid_protocol.json --output runs/reid_input
```

`--depth-unit`必须显式选native_im2double或millimeters_to_meters，取决于实际数据，不可假定所有PNG都是毫米。可用 `--parameters path/to/gradkdes_params.mat` 或NPZ指定已有参数。

输出：特征NPZ、basis NPZ、RGB/Depth码本与G、来源JSON；保留训练sample IDs、输入CSV哈希和生产代码哈希。输出及其伴随文件禁止覆盖。默认1000词在训练patch不足时失败，不重复补样本或降低词数。

## 验证与边界

- 高斯导数及局部特征与独立逐像素标量实现一致；包括镜像源码的1e-5下限所导致的非零平坦patch响应。
- EMK的同词冗余抑制通过手算例；1000词时实际输出21000/14000维。
- 码本重复运行一致；GᵀG与原核矩阵逆吻合；训练patch不足会失败。
- 实际生成200×2500投影，在两张本地ORL图上运行局部特征、16词**小型流程码本**和336维图像编码，全部通过，见 `runs/kdes_source_port_20261031/acceptance.json`。
- 没有MATLAB/Octave运行时，尚未用原MATLAB执行结果逐元素交叉验证；不同eigsh/eigs的符号或简并子空间、PIL灰度转换和RNG均可能产生差别。源码移植及标量对照不等于证明与作者历史软件逐bit相同。
- 没有对应真实行人数据，因此未生成或报告任何真实BIWI/替代RGB-D行人实验成绩。以上流程检查不作为算法泛化证据。
