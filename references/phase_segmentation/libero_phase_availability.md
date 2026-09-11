# Does LIBERO provide phase labels?

Short answer: **No explicit phase labels are provided by default**, but LIBERO provides enough demonstration information to derive weak phase labels for D1.

## What LIBERO provides

According to the local archived LIBERO repository:

- official human teleoperation demonstration datasets for `libero_spatial`, `libero_object`, `libero_goal`, and `libero_100`;
- HDF5 demonstration files under the LIBERO dataset folder after download;
- per-demo `actions`;
- per-demo flattened MuJoCo `states`;
- dataset metadata including `problem_info`, `env_args`, `bddl_file_name`, and language instruction;
- optional observations generated through `scripts/create_dataset.py`, including RGB cameras, depth, and low-dimensional robot states depending on creation flags;
- algorithm config support for `joint_states`, `gripper_states`, and `ee_states` as low-dimensional modalities.

Relevant local code paths:

- `references/repos/libero/README.md`: dataset download instructions and HuggingFace mirror.
- `references/repos/libero/scripts/get_dataset_info.py`: reads HDF5 structure, trajectory lengths, actions, env metadata, language instruction.
- `references/repos/libero/scripts/create_dataset.py`: replays raw states/actions and can save RGB/depth/proprio observations.
- `references/repos/libero/libero/lifelong/datasets.py`: Robomimic `SequenceDataset` wrapper; reads observations and `actions`.
- `references/repos/libero/benchmark_scripts/check_task_suites.py`: checks `.hdf5` demo files and uses `data/demo_0/states`.

## What LIBERO does not directly provide

The default dataset is not a phase-segmentation dataset. It does not appear to ship labels like:

```text
approach_source / grasp / lift_or_move / approach_target / release
```

It also does not appear to provide ready-made `source → mixed → target` labels for our D1 `L_phase`.

## How this affects our plan

LIBERO is still useful for D1 because it gives the ingredients to derive weak phase boundaries:

```text
states + actions + env replay + optional low-dim observations
```

Recommended implementation:

1. Download the official LIBERO demos when starting P5/P6.
2. For the selected pick-place task, inspect one `.hdf5` file with:

```bash
python references/repos/libero/scripts/get_dataset_info.py --dataset <path/to/task_demo.hdf5> --verbose
```

3. If only raw MuJoCo `states` and `actions` are present, replay the demo with `create_dataset.py` to save RGB and low-dimensional observations.
4. Derive event boundaries:

```text
t_grasp: gripper closing and EEF close to source
t_lift: source height increases or source moves with EEF
t_near_target: source/object close to target/container
t_release: gripper opening near target
```

5. Convert boundaries into soft D1 labels:

```text
before grasp                  -> source, α = 0.9
after grasp before near target -> mixed,  α = 0.5
after near target              -> target, α = 0.1
```

## Practical caveat

Because phase labels are derived rather than provided, the first D1 step should be diagnostic:

- visualize inferred phases on replayed RGB frames;
- plot `mass_source(t)` and `mass_target(t)`;
- compare against random / reversed / fixed-percentage phase baselines;
- only enable `L_phase` after the inferred boundaries are stable.

## Current local status

At the time of this note, the workspace has the LIBERO code repository and paper archived, but **does not have the LIBERO HDF5 demonstration datasets downloaded locally**. The local search found no `.hdf5` or `.h5` dataset files under the project workspace.
