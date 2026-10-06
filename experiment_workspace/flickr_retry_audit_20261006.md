# Flickr SD1.5 五条失败样本定向重试审计

- 服务器：`vla101`（本次审计时 hostname 为 `di-20260613120619-84fwm`）
- teacher：`stable-diffusion-v1-5`
- seed：`17`
- diffusion steps：`20`
- inner steps：`10`
- resolution：`16`
- retry audit：`vla_workspace/experiment_workspace/results/cache_safety_snapshots/20261006T_retry/retry_audit.json`
- retry 结果：5/5 成功，失败数 0；原始 cache 未覆盖，使用独立 retry overlay。

| part | sample_id | map SHA256 | metadata SHA256 | retry manifest SHA256 |
|---:|---|---|---|---|
| 0 | `4766439161:caption:3` | `a17ae59669f9472510c8e4355d5d08be3c7a8e073e0d1bcfa6495d33ce5eb105` | `00571cb70e3f67c0ea13f697a97d0f56c44a4590a0e252427ab1168e106b0d09` | `01379bea528c39e2f42cc73ab2ff17b80edefff91d986400a8595e400935ffff` |
| 1 | `363503748:caption:3` | `8cc11b77e7880e3895dd88faea804ecfadddec1bdf6cb6d0f04b523f76e977a7` | `855ed7989c42df7c0fd3e2f327906c15d8750b264bed1d7cb73c0f88ffee0018` | `e10bd989a7747674cfc01aa58b4113143bac4f63477b33cb0731716857eeb3ad` |
| 2 | `6337783711:caption:4` | `defc1a30de48fdc005ed6e816a38e21e0fd5f425f997b3f787fa5048f1d0a9e9` | `d958bc3fa7fca8c5f1c5ae92041e29937f3240bd610d2788f11cf7b9b2eb35b0` | `55d634db1ba111045a726c33fc4db327286d962e34e5e051775f69728bcb013f` |
| 3 | `3732420759:caption:2` | `aac47060a673f2f6af50512f120b21191c1ea35b813d3d39dc279e6a66d1790` | `23596fc12d86212a2d2a49c9f655b6179a6cad5003b093243ace5af13c693230` | `e69fa5ce5e2ccb07dc78f255ea5bf7c1446946353d325805ed511213a2b94337` |
| 3 | `2355098740:caption:3` | `3e28660a228bee37b1e5b99cbc0dfa4e981178200e247cdbb0cc7a28a35717e2` | `4ddc27a6d87718b795e45757eb09759637521bc55758b8f782a6c0c7629c2ca0` | `e69fa5ce5e2ccb07dc78f255ea5bf7c1446946353d325805ed511213a2b94337` |

正式训练 manifest 可保留这 5 条的 retry overlay；不能把原始 cache 的失败列表静默删除。完整 Flickr audit 已报告 `9952/9952`、`failures=[]`、`overlay_not_merged_into_original=true`。
