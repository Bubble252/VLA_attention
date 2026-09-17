# 实验配置规范

配置使用 JSON，避免在 P1 前引入 YAML/Hydra 依赖。每个正式实验都有一个不可变 JSON manifest，包含：

```text
experiment_id / stage / hypothesis
model + revision + adapter
teacher + frozen configuration
dataset manifest + split
action interface
seed / budget / metrics / stopping rules
Git commit
```

只在 validation/calibration 数据上更新 teacher、层、阈值或超参数。测试 manifest 不得在结果后修改；新尝试新建 experiment ID。
