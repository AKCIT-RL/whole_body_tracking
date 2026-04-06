#!/usr/bin/env bash
# Roda play.py para múltiplas runs do W&B.
# O play.py exporta os modelos em ./logs/rsl_rl/temp/exported/
# Este script move os arquivos exportados para output/politicas/<nome>/ após cada run.
#
# Uso (a partir da raiz do repo):
#   bash scripts/batch_play.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x /isaac-sim/python.sh ]]; then
  py_cmd=/isaac-sim/python.sh
else
  py_cmd=python
fi

# formato: "wandb_path|nome_pasta"
RUNS=(
  "gabrielruotolo-federal-univesity-of-goias/Booster_t1/e8g8xo1r|trote_medio_vertical"
  "gabrielruotolo-federal-univesity-of-goias/Booster_t1/6dqj81se|trivela"
  "gabrielruotolo-federal-univesity-of-goias/Booster_t1/nlfvc6rz|pedalada_newcut"
  "gabrielruotolo-federal-univesity-of-goias/Booster_t1/klah13pq|neyPenalti_cut"
  "gabrielruotolo-federal-univesity-of-goias/Booster_t1/4xqkzas4|pedalada_lenta"
  "gabrielruotolo-federal-univesity-of-goias/Booster_t1/xbvzf36g|penaltiNeymar"
)

TEMP_EXPORT="$REPO_ROOT/logs/rsl_rl/temp/exported"

for entry in "${RUNS[@]}"; do
  run="${entry%%|*}"
  name="${entry##*|}"
  out_dir="$REPO_ROOT/output/politicas/$name"
  mkdir -p "$out_dir"

  # Limpa exported anterior para não misturar arquivos de runs diferentes
  rm -rf "$TEMP_EXPORT"

  echo "========== play: $name ($run) =========="
  $py_cmd scripts/rsl_rl/play.py \
    --task=Tracking-Flat-T1-Wo-State-Estimation-v0 \
    --num_envs 1 \
    --wandb_path "$run" \
    --headless \
    --video \
    --video_length 1

  # Move os arquivos exportados para a pasta desta run
  if [[ -d "$TEMP_EXPORT" ]]; then
    mv "$TEMP_EXPORT"/* "$out_dir"/
    echo "[INFO] Arquivos exportados salvos em: $out_dir"
  else
    echo "[WARN] Nenhum arquivo exportado encontrado em $TEMP_EXPORT"
  fi
done

echo "========== batch_play finalizado =========="
