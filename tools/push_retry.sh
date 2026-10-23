#!/usr/bin/env bash
# Push the bot's commit to main, retrying. On a clash with another bot's commit (e.g. tracktest progress logs)
# our side wins (-X theirs = the commit being replayed). Exits 1 if nothing got pushed, so the job shows red.
# (1 Oct: the 30 Sep nightly scorecard was lost to a log.txt conflict while the job still showed green.)
# (23 Oct, BALL-EDGE: up to 16 shard jobs finish close together - 8 tries with a random extra wait.)
tries=${1:-8}; wait_s=${2:-5}
for i in $(seq "$tries"); do
  if git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main; then echo "pushed"; exit 0; fi
  git rebase --abort 2>/dev/null; sleep $(( wait_s + RANDOM % 20 ))
done
echo "PUSH FAILED after $tries tries"; exit 1
