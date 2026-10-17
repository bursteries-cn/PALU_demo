# seed103 完整更新 profiling

此工具只重放现有 seed103 冻结实验的前 6 次更新，不启动三分支实验，不执行 FQ/MU，也不导出模型权重。需要已有完整 `frozen/`、历史 `prefix/loss_history.jsonl` 和 `fork/receipt.json`；这些私有实验输入不随公开源码发布。

```bash
STUDY_DIR=/absolute/path/to/seed103_study \
PROFILE_DIR=/absolute/path/to/new_profile_output \
FOUR_GPUS=0,1,2,3 bash scripts/palu/run_training_profile.sh

.venv/bin/python scripts/palu/analyze_training_profile.py \
  --runs /absolute/path/to/new_profile_output \
  --study /absolute/path/to/seed103_study \
  --output /absolute/path/to/profile_analysis
```

两个后端顺序执行，均从原 Full 起点、seed103 和相同前 192 对 F/R 开始；保留四卡每卡 batch1×累积8、整步 token 分母、每条样本 padding、原优化器/精度/梯度检查点及 linear60/warmup6。运行预热为 updates1–3，统计 updates4–6；这个预热区间不改变学习率 warmup。

`baseline` 只做轻量阶段计时；`trace` 在第6次完整更新采集四rank的 CPU/CUDA Kineto trace。两个后端都只在step6计时前加一次对齐barrier，不在阶段或微批边界插入CUDA同步。原来的日志、标量读取、梯度累积和更新代码通过AST上下文包装保留，原实现文件不修改；测试会去除包装并检查AST相同。

主线程互斥阶段分为学生F/R前向、教师前向、损失构建、反向（包含梯度检查点重计算）、优化器、输入/计数和日志/标量同步。它们包含CUDA等待，不能称为纯GPU计算。分析取每次完整更新最慢rank的阶段，再对3次更新取平均。

GPU时间线按区间并集统计NCCL、其他GPU活动、两者重叠、仅NCCL活跃及没有本进程GPU活动的时间。NCCL kernel驻留包含等待，不等于纯传输；通信已经包含在主线程阶段中，不能额外相加。单个通信类型和阶段GPU并集也可能相互重叠。

autograd工作线程上的kernel可能没有同线程的PALU标记，其发起归属保留为unattributed；另外按GPU活动与主线程反向等阶段的时间区间交集统计，不伪造跨线程父子关系。损失构建的原始标记包含嵌套前向，其时间窗口不可与子窗口相加。分析默认顺序展开trace；RAM充足时可用`--workers 4`并行解析四份JSON。

完成后逐项比较两次六步loss/梯度范数/LR/全部rank配对和历史前缀；最终四rank的模型、FP32 master、Adam持久状态、scheduler、RNG与计数必须与历史step6一致。保留轻量日志、插桩源码、状态指纹和压缩trace。trace可能产生较大CPU与文件开销，分析另外报告相同step6的实测墙钟比值；单次顺序对照不能完全分离采集开销与资源波动。

已有输出拒绝覆盖。原实验目录只读；`fork_weights`不反序列化、不修改。此工具不支持任意seed、其他batch布局或缺少历史第6步receipt的实验。
