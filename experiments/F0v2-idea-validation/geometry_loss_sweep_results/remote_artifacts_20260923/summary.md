# Geometry loss sweep: held-out summary

All four map objectives use the frozen calibration grid/threshold and three training seeds. Values are means across the same image-disjoint held-out set; seed mean±sample-SD are reported below.

| loss | seeds | calibrated IoU | soft-IoU | top20 IoU | mass-in-box | pointing |
|---|---|---:|---:|---:|---:|---:|
| js | 17,29,41 | 0.2812±0.0103 | 0.0711±0.0053 | 0.2717±0.0013 | 0.3208±0.0096 | 0.3229±0.0771 |
| kl | 17,29,41 | 0.2838±0.0064 | 0.0748±0.0040 | 0.2709±0.0015 | 0.3266±0.0089 | 0.3438±0.0563 |
| kl_moment | 17,29,41 | 0.2893±0.0073 | 0.0790±0.0027 | 0.2748±0.0011 | 0.3304±0.0045 | 0.3333±0.0180 |
| kl_rank | 17,29,41 | 0.2758±0.0051 | 0.0799±0.0062 | 0.2732±0.0009 | 0.3306±0.0061 | 0.3385±0.0861 |
