#!/usr/bin/env bash
# Push the bot's commit to main, retrying. On a clash with another bot's commit (e.g. tracktest progress logs)
# our side wins (-X theirs = the commit being replayed). Exits 1 if nothing got pushed, so the job shows red.
# (1 Oct: the 30 Sep nightly scorecard was lost to a log.txt conflict while the job still showed green.)
tries=${1:-3}; wait_s=${2:-5}
for i in $(seq "$tries"); do
  if git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main; then echo "pushed"; exit 0; fi
  git rebase --abort 2>/dev/null; sleep "$wait_s"
done
echo "PUSH FAILED after $tries tries"; exit 1
