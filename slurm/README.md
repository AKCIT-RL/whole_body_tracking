# slurm/ — treinar policies low-pass G1 numa OVX (L40S)

Batch BeyondMimic (`whole_body_tracking`) para os **21 motions low-pass**
(gerados em `CopyCat/lowpass_mocap/`): **Apptainer + Slurm, tudo em `/raid`**,
seguindo o *Manual do Cluster* e o padrão do
[`AKCIT-RL/isaac-arenas/apptainer`](https://github.com/AKCIT-RL/isaac-arenas/tree/main/apptainer).

**2 arquivos que importam:** [`isaaclab_wbt.def`](isaaclab_wbt.def) (a imagem) e
[`train_lowpass.sbatch`](train_lowpass.sbatch) (o job array — 1 task por motion:
`csv_to_npz` → registry W&B → `train.py`, com auto-resume + requeue).

Nenhum valor da sua conta/cluster está aqui — vão em `--export=...` e no `.env`.

---

## 1. Setup (login node, uma vez)

```bash
cd /raid/$USER
git clone git@github.com:AKCIT-RL/CopyCat.git          # SSH, nunca https://TOKEN@...
cd CopyCat/train/whole_body_tracking
```

### 1a. Construir a imagem

Precisa de uma NGC API key (grátis): <https://ngc.nvidia.com/setup/api-key>.

```bash
export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD='<sua-NGC-API-key>'       # começa com nvapi-
export APPTAINER_TMPDIR=/raid/$USER/tmp/apptainer          # scratch fora da home
mkdir -p /raid/$USER/containers "$APPTAINER_TMPDIR"

apptainer build /raid/$USER/containers/isaaclab_wbt.sif slurm/isaaclab_wbt.def   # ~20-40 min
```

> Rode de `train/whole_body_tracking/` (o `%files` do `.def` copia `source/` e
> `scripts/` de lá). Rebuild só quando mexer em `source/` ou `scripts/`.
>
> Se o `git clone` do fork IsaacLab dentro do `%post` falhar: o repo já tem
> `IsaacLab/`; adicione `IsaacLab  /opt/IsaacLab` ao `%files` e apague as 2 linhas
> do `git clone` (comentário no `.def`).

Smoke test: `apptainer exec /raid/$USER/containers/isaaclab_wbt.sif /isaac-sim/python.sh -c 'import torch,wandb,rsl_rl;print(torch.__version__)'`

### 1b. Segredos do W&B

```bash
mkdir -p /raid/$USER/wbt_lowpass
cp slurm/.env.example /raid/$USER/wbt_lowpass/.env
chmod 600 /raid/$USER/wbt_lowpass/.env
vi /raid/$USER/wbt_lowpass/.env        # WANDB_API_KEY (de https://wandb.ai/authorize) + WANDB_ENTITY
```

### 1c. Colocar os CSVs no `/raid`

```bash
rsync -a /caminho/CopyCat/lowpass_mocap/output/csv/  /raid/$USER/CopyCat/lowpass_mocap/output/csv/
# (ou já vem no clone, se lowpass_mocap/output/csv/ estiver commitado)
```

---

## 2. Rodar

Ache a partição com `sinfo` (o node é `ovx-l40s-01`), passe org e partição na flag:

```bash
cd slurm

# todos os 21 motions
sbatch --partition=<part> --export=ALL,WANDB_ENTITY=<org> train_lowpass.sbatch

# só alguns (índice = posição em MOTIONS[] no .sbatch: 0=aceno ... 14=run ... 20=what_is_love)
sbatch --partition=<part> --export=ALL,WANDB_ENTITY=<org> --array=14      train_lowpass.sbatch
sbatch --partition=<part> --export=ALL,WANDB_ENTITY=<org> --array=0,4,14  train_lowpass.sbatch

# L40S apertada de VRAM
sbatch --partition=<part> --export=ALL,WANDB_ENTITY=<org>,NUM_ENVS=2048   train_lowpass.sbatch

# limitar quantas rodam ao mesmo tempo (= nº de GPUs que você pode usar)
sbatch --partition=<part> --export=ALL,WANDB_ENTITY=<org> --array=0-20%2  train_lowpass.sbatch
```

Overrides via `--export=ALL,VAR=valor` (defaults no topo do `.sbatch`):
`RAID SIF RUN_DIR CACHE CSV_DIR ENV_FILE WANDB_ENTITY NPZ_PROJECT TRAIN_PROJECT
PREFIX TASK NUM_ENVS MAX_ITER EXTRA_ARGS`.

### Monitorar

```bash
squeue -u $USER
tail -f slurm/logs/lp-g1-<ARRAYJOBID>_<TASKID>.out
tail -f /raid/$USER/wbt_lowpass/.isaac-cache/logs/*.log     # logs do Isaac Sim
scancel <jobid>            # ou  <arrayjob>_<task>  /  -n lp-g1  /  -u $USER
```

---

## 3. O que o job faz

Por task (1 motion):

1. **`csv_to_npz.py`** — replay no Isaac Sim, gera `motion.npz` (pose/vel/accel via FK),
   sobe pro registry W&B `<org>/<NPZ_PROJECT>/<PREFIX><motion>`. Pula se já existe.
2. **`train.py`** — PPO (`Tracking-Flat-G1-v0`), `MAX_ITER` iters, checkpoint a cada 500.
   Loga curvas em `<org>/<TRAIN_PROJECT>`.

Dentro do container (`apptainer exec --nv --no-home`):

- **caches todos em `/raid`** (`$CACHE`): Isaac kit/kit-data, `~/.cache/{ov,pip}`,
  GLCache, ComputeCache, omniverse logs/data/documents
- `/tmp` → `/raid/$USER/tmp` (o `/tmp` do node é pequeno/read-only)
- `HOME` → dir gravável em `$CACHE/home` (Isaac resolve cache por `$HOME` mesmo com `--no-home`)
- segredos via `--env-file $RUN_DIR/.env`
- CWD = `$RUN_DIR` → `logs/`, `artifacts/`, `motions/` caem no `/raid`
- **nada** escrito na home do host

1ª execução: Isaac Sim leva ~10-15 min cacheando extensions (só a 1ª paga).

---

## 4. Checkpoints / interrupções

- `$RUN_DIR/logs/rsl_rl/<PREFIX><motion>/<run>_<ts>/model_*.pt` (a cada 500 iters), no `/raid`.
- **Auto-resume:** ao (re)iniciar, a task acha o run mais recente com checkpoint e passa
  `--resume True --load_run <run>`.
- **Manutenção / walltime:** `#SBATCH --signal=B:SIGUSR1@300` + `--requeue` → 300 s antes
  do limite o job pega `SIGUSR1`, roda `scontrol requeue` e volta pra resumir do último
  checkpoint completo (perda ≤ 500 iters).
- **W&B contínuo:** `WANDB_RUN_ID=lp_<motion>` + `WANDB_RESUME=allow`.

Recomeçar um motion do zero: apague `$RUN_DIR/logs/rsl_rl/<PREFIX><motion>/` antes.

---

## 5. Avaliar / exportar policy

Num `salloc` (continua Slurm), reaproveitando a função `run()`:

```bash
salloc --partition=<part> --gpus=1 --cpus-per-task=8 --mem=32G --time=1:00:00
cd slurm && export WANDB_ENTITY=<org>
# copie o bloco 'run()' do train_lowpass.sbatch, ou:
apptainer exec --nv --no-home --cleanenv --pwd /raid/$USER/wbt_lowpass \
  -B /raid/$USER/wbt_lowpass:/raid/$USER/wbt_lowpass --env-file /raid/$USER/wbt_lowpass/.env \
  --env HOME=/home/$USER /raid/$USER/containers/isaaclab_wbt.sif \
  bash -lc 'cd /raid/$USER/wbt_lowpass && /isaac-sim/python.sh /opt/wbt/scripts/rsl_rl/play.py \
    --task Tracking-Flat-G1-v0 --num_envs 4 --wandb_path '"$WANDB_ENTITY"'/LowPass_G1_Training/<run-id>'
```

`play.py` já exporta ONNX/JIT para `logs/.../exported/`.

---

## 6. Troubleshooting

| sintoma | correção |
|---|---|
| `sbatch: invalid partition` | `sinfo`; use o nome certo em `--partition=` |
| `apptainer build` → `401` | `APPTAINER_DOCKER_PASSWORD` = sua NGC key (`$oauthtoken` como user) |
| `RuntimeError: NVIDIA driver too old` | imagem fixa `torch 2.11.0+cu128`; se persistir troque p/ wheel `+cu121`/`+cu126` no `.def` |
| `CUDA out of memory` | `--export=ALL,...,NUM_ENVS=2048` |
| `csv_to_npz` "trava" após *"Logging to wandb"* | é o teardown; o script `os._exit(0)` após salvar |
| `wandb: permission denied` | `WANDB_ENTITY` = org e você membro; `WANDB_API_KEY` no `.env` |
| `CSV not found` | `rsync` os CSVs p/ `$CSV_DIR` no `/raid` (passo 1c) |
| job requeue em loop | é o `--requeue` após `SIGUSR1`; ok — ele resume. Aumente `--time` se a fila deixar |

---

## Arquivos

```
slurm/
├── isaaclab_wbt.def      imagem: Isaac Lab 2.1.0 + fork 2.3.2 + torch cu128 + wbt + assets G1
├── train_lowpass.sbatch  o job array (config no topo, tudo via --export)
├── .env.example          → copie p/ $RUN_DIR/.env  (WANDB_API_KEY, WANDB_ENTITY)
└── README.md
```
