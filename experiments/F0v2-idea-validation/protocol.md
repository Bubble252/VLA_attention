# F0v2 idea validation protocol

- Held-out manifest: 64 image-phrase records from official Entities test, image-disjoint from F0 train, seed 23.
- Evaluate V0--V4 on the same records with pointing, mass-in-box, IoU and map validity.
- Compare V3 vs V1, V4 vs V2, and correct teacher vs wrong-word/wrong-image/random maps.
- This is directional evidence only; pass requires all trends before F1-10k.

当前 held-out manifest 已生成于 101：64 条、seed 23、SHA256 `7b16778e…`。F0 smoke 原先只保存 loss JSON，没有 adapter checkpoint，因此下一步必须补跑带 checkpoint save/load 的短训练，才能进行 V0–V4 held-out 指标比较；不能把 smoke loss 当作 held-out 结果。
