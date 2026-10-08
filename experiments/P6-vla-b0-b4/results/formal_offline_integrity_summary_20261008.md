# P6 B0–B4 formal offline artifact integrity summary（2026-10-08）

本文件只记录 seed17 formal offline 训练后产物完整性。它不包含 checkpoint 或 cache，只包含路径、SHA/fingerprint 和关键字段。

- 远端结果根目录：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/formal_offline_20261007`
- 远端 repo：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention`
- 远端 Git SHA：`unavailable:CalledProcessError(128, ['git', '-C', '/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention', 'rev-parse', 'HEAD'])`
- 结论边界：完整性通过只证明产物可恢复和可审计，不证明 rollout success。

| group | final status | metrics lines | final SHA256 | train_metrics SHA256 | ckpt2520 | ckpt5040 | ckpt7560 | ckpt10080 |
|---|---:|---:|---|---|---|---|---|---|
| B0 | passed | 10080 | `bf038f5d4eea` | `dac2a3dce623` | `ok:ba54b1c6c926` | `ok:f63287c09ee2` | `ok:61349200109c` | `ok:53bbfe71509c` |
| B1 | passed | 10080 | `1d23a9d3bcec` | `84f5044e409f` | `ok:11f13faf10be` | `ok:d59c3f234d02` | `ok:46fc6ed50ed9` | `ok:10535f13487a` |
| B2 | passed | 10080 | `87f710cb05d9` | `c288cafadebe` | `ok:7653bb44ebd0` | `ok:ff08b74ede44` | `ok:6658f92a3170` | `ok:9569ef0e1509` |
| B3 | passed | 10080 | `87b46c071e79` | `522dab9f2f86` | `ok:e028c86e11bb` | `ok:7a6ba1c9bf98` | `ok:9aeaa5a21c70` | `ok:01dd850ec8ee` |
| B4 | passed | 10080 | `2fa7918ee412` | `677a69277a8a` | `ok:ad6def21ab72` | `ok:0055d79bfe3d` | `ok:eb7f7d1843ef` | `ok:6f456b763585` |

## final.json 关键字段与 train_metrics 最后一行

### B0

- run dir: `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/formal_offline_20261007/B0_seed17`
- final key fields: `{"group": "B0", "seed": 17, "status": "passed", "steps": 10080}`
- train_metrics last line: `{"optimizer_step": 10080, "max_steps": 10080, "progress_fraction": 1.0, "group": "B0", "seed": 17, "episode_id": "046f864859938594fd054d25025a6756e2dbc278f98ff99cd0d88be8adc9305a", "timestep": 114, "action_loss": 0.261794775724411, "retention_loss": 0.0, "semantic_loss": 0.0, "containment_loss": 0.0, "total_loss": 0.261794775724411, "effective_lambda_semantic": 0.1, "effective_lambda_retention": 0.05, "effective_lambda_containment": 0.02, "retention_warmup_scale": 1.0, "containment_warmup_scale": 1.0, "gradient_norm_pre_clip": 20.5, "learning_rate": 0.0, "step_seconds": 1.154660701751709, "peak_memory_bytes": 22801471488, "finite": true, "attribution_create_graph": false, "phrase_positions": {}}`

### B1

- run dir: `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/formal_offline_20261007/B1_seed17`
- final key fields: `{"group": "B1", "seed": 17, "status": "passed", "steps": 10080}`
- train_metrics last line: `{"optimizer_step": 10080, "max_steps": 10080, "progress_fraction": 1.0, "group": "B1", "seed": 17, "episode_id": "046f864859938594fd054d25025a6756e2dbc278f98ff99cd0d88be8adc9305a", "timestep": 114, "action_loss": 0.23711887001991272, "retention_loss": 0.30078086256980896, "semantic_loss": 0.0, "containment_loss": 0.0, "total_loss": 0.25215792655944824, "effective_lambda_semantic": 0.1, "effective_lambda_retention": 0.05, "effective_lambda_containment": 0.02, "retention_warmup_scale": 1.0, "containment_warmup_scale": 1.0, "gradient_norm_pre_clip": 22.375, "learning_rate": 0.0, "step_seconds": 1.1548912525177002, "peak_memory_bytes": 22844771840, "finite": true, "attribution_create_graph": false, "phrase_positions": {}}`

### B2

- run dir: `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/formal_offline_20261007/B2_seed17`
- final key fields: `{"group": "B2", "seed": 17, "status": "passed", "steps": 10080}`
- train_metrics last line: `{"optimizer_step": 10080, "max_steps": 10080, "progress_fraction": 1.0, "group": "B2", "seed": 17, "episode_id": "046f864859938594fd054d25025a6756e2dbc278f98ff99cd0d88be8adc9305a", "timestep": 114, "action_loss": 0.31606683135032654, "retention_loss": 0.0, "semantic_loss": 0.03908082842826843, "containment_loss": 0.0, "total_loss": 0.3199748992919922, "effective_lambda_semantic": 0.1, "effective_lambda_retention": 0.05, "effective_lambda_containment": 0.02, "retention_warmup_scale": 1.0, "containment_warmup_scale": 1.0, "gradient_norm_pre_clip": 22.0, "learning_rate": 0.0, "step_seconds": 1.8969714641571045, "peak_memory_bytes": 38676335616, "finite": true, "attribution_create_graph": true, "phrase_positions": {"source": [12, 13, 14, 15], "target": [28, 29]}}`

### B3

- run dir: `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/formal_offline_20261007/B3_seed17`
- final key fields: `{"group": "B3", "seed": 17, "status": "passed", "steps": 10080}`
- train_metrics last line: `{"optimizer_step": 10080, "max_steps": 10080, "progress_fraction": 1.0, "group": "B3", "seed": 17, "episode_id": "046f864859938594fd054d25025a6756e2dbc278f98ff99cd0d88be8adc9305a", "timestep": 114, "action_loss": 0.22878047823905945, "retention_loss": 0.3017066717147827, "semantic_loss": 0.1392802596092224, "containment_loss": 0.0, "total_loss": 0.2577938437461853, "effective_lambda_semantic": 0.1, "effective_lambda_retention": 0.05, "effective_lambda_containment": 0.02, "retention_warmup_scale": 1.0, "containment_warmup_scale": 1.0, "gradient_norm_pre_clip": 20.75, "learning_rate": 0.0, "step_seconds": 1.9318571090698242, "peak_memory_bytes": 38721098240, "finite": true, "attribution_create_graph": true, "phrase_positions": {"source": [12, 13, 14, 15], "target": [28, 29]}}`

### B4

- run dir: `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/formal_offline_20261007/B4_seed17`
- final key fields: `{"group": "B4", "seed": 17, "status": "passed", "steps": 10080}`
- train_metrics last line: `{"optimizer_step": 10080, "max_steps": 10080, "progress_fraction": 1.0, "group": "B4", "seed": 17, "episode_id": "046f864859938594fd054d25025a6756e2dbc278f98ff99cd0d88be8adc9305a", "timestep": 114, "action_loss": 0.32235828042030334, "retention_loss": 0.3035193681716919, "semantic_loss": 3.803358554840088, "containment_loss": 0.000940276775509119, "total_loss": 0.7178889513015747, "effective_lambda_semantic": 0.1, "effective_lambda_retention": 0.05, "effective_lambda_containment": 0.02, "retention_warmup_scale": 1.0, "containment_warmup_scale": 1.0, "gradient_norm_pre_clip": 1776.0, "learning_rate": 0.0, "step_seconds": 2.3452656269073486, "peak_memory_bytes": 46261135360, "finite": true, "attribution_create_graph": true, "phrase_positions": {"source": [12, 13, 14, 15], "target": [28, 29]}}`

