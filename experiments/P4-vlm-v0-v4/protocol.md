# P4：VLM V0–V4 protocol

未运行。P1 四个接口通过、teacher calibration 完成后才填写并提交。

```text
V0 checkpoint-only
V1 SFT
V2 DB-style feature retention
V3 best-single T_sem → A_lang
V4 V2 + V3
```

主 benchmark：Flickr30k Entities；外部指代：RefCOCOg；视觉 OOD：固定 Flickr task-preserving perturbations；保留诊断：VL-Think static QA。具体 data manifest、teacher configuration、预算和 metrics 在运行前填入 JSON 并 commit。
