# treesample
Sample strings from open LMs (e.g. OLMo and its checkpoints), parse them with benepar, analyze the trees.

    python scripts/01_sample.py   experiment=olmo2_1b_ckpts
    python scripts/02_parse.py    experiment=olmo2_1b_ckpts
    python scripts/03_analyze.py  experiment=olmo2_1b_ckpts

Override anything from the CLI, e.g. `model=olmo2_7b model.revision=stage1-step100000-tokens420B backend=vllm`.
