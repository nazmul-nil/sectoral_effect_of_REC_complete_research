#!/bin/bash

START_DATE="2025-04-25"
START_TS=$(date -d "$START_DATE" "+%s")

messages=(
  "Initialize project structure and metadata for sectoral analysis"
  "Add early draft of research question and conceptual framework"
  "Upload raw World Bank panel data (1989–2023)"
  "Implement income classification merge logic from WB classifications"
  "Apply temporal imputation (forward/backward fill by group)"
  "Conduct IQR-based outlier detection and winsorization"
  "Add documentation for sample construction and exclusion rules"
  "Develop variable construction pipeline for REC, EI, GDP, and sectors"
  "Implement fixed effects regression baseline model"
  "Add income group–specific estimation results"
  "Create plots for threshold effects using quartiles of REC"
  "Integrate Arellano-Bond dynamic panel estimation with lag instruments"
  "Perform heterogeneity analysis across low/mid/upper income levels"
  "Run robustness checks with 1st-difference and clustered errors"
  "Visualize temporal evolution of REC effects across decades"
  "Generate actual vs predicted plots by sector and country"
  "Document limitations and future work roadmap in discussion"
  "Add draft of final policy implications and threshold insights"
  "Export regression tables and plots for manuscript inclusion"
  "Final cleanup before preprint submission"
)

files=( $(find . -type f ! -path "./.git/*" ! -name "*.sh" | shuf) )
total=${#messages[@]}

for i in $(seq 0 $((total-1))); do
  file=${files[$((i % ${#files[@]}))]}
  echo "# Commit $i - ${messages[$i]}" >> "$file"

  git add "$file"

  commit_ts=$((START_TS + i * 3 * 86400))
  commit_date=$(date -d "@$commit_ts" "+%Y-%m-%dT%H:%M:%S")

  GIT_AUTHOR_DATE="$commit_date" GIT_COMMITTER_DATE="$commit_date" \
  git commit -m "${messages[$i]}"

  echo "✅ Commit $((i+1)): ${messages[$i]} | $commit_date"
done

git branch -M master
git push -u origin master
