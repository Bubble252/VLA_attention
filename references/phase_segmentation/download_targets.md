# Phase segmentation download targets

Use proxy `http://127.0.0.1:7897` when available.

```bash
cd /home/bubble/类脑计算/VLM终局
mkdir -p references/phase_segmentation/papers
curl -L --fail --retry 2 --connect-timeout 20 --proxy http://127.0.0.1:7897 \
  http://proceedings.mlr.press/v80/shiarlis18a/shiarlis18a.pdf \
  -o references/phase_segmentation/papers/taco_2018_icml.pdf
curl -L --fail --retry 2 --connect-timeout 20 --proxy http://127.0.0.1:7897 \
  https://arxiv.org/pdf/2109.13841 \
  -o references/phase_segmentation/papers/buds_2109.13841.pdf
curl -L --fail --retry 2 --connect-timeout 20 --proxy http://127.0.0.1:7897 \
  https://arxiv.org/pdf/2505.15517 \
  -o references/phase_segmentation/papers/robo2vlm_2505.15517.pdf
curl -L --fail --retry 2 --connect-timeout 20 --proxy http://127.0.0.1:7897 \
  https://arxiv.org/pdf/2605.08774 \
  -o references/phase_segmentation/papers/procvlm_2605.08774.pdf
```

Other sources:

- CHAMP DOI: https://doi.org/10.1109/ICRA.2015.7139383
- IROS 2012 unstructured demonstrations DOI: https://doi.org/10.1109/IROS.2012.6386006
- RoboSegNet project: https://berkeleyautomation.github.io/RoboSegNet/
- ROVER project: https://rover-vlm.github.io/
```
