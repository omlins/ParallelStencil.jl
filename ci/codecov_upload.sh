#!/bin/bash
# Upload the coverage report lcov.info to Codecov from a CSCS CI job (run in a GitLab mirror of the GitHub repository).
# Usage: ci/codecov_upload.sh <flag>
# The upload is skipped if CODECOV_TOKEN is not set (secret variable on the CSCS CI setup page) and never fails the job.
set -uo pipefail
flag=$1
slug=omlins/ParallelStencil.jl

if [ -z "${CODECOV_TOKEN:-}" ]; then echo "CODECOV_TOKEN is not set: skipping the Codecov upload."; exit 0; fi
if [ ! -f lcov.info ]; then echo "lcov.info not found: skipping the Codecov upload."; exit 0; fi

# Commit and branch of the original GitHub repository: PR pipelines check out the PR head commit with the ref __CSCSCI__pr<N>;
# branch pipelines use the branch name. NOTE: the branch must be given explicitly, else the GitLab ref name would be used.
if [[ "${CI_COMMIT_REF_NAME}" =~ ^__CSCSCI__pr([0-9]+)$ ]]; then
    pr=${BASH_REMATCH[1]}
    ref_args=(--pr "$pr")
    branch=$(curl -fsSL "https://api.github.com/repos/$slug/pulls/$pr" | python3 -c 'import json,sys; print(json.load(sys.stdin)["head"]["ref"])') && ref_args+=(--branch "$branch")
else
    ref_args=(--branch "${CI_COMMIT_REF_NAME}")
fi

# Codecov CLI for the architecture of the runner (verified with its SHA256 checksum).
case "$(uname -m)" in
    aarch64) platform=linux-arm64 ;;
    *)       platform=linux ;;
esac
curl -fsSL -o codecov "https://cli.codecov.io/latest/$platform/codecov" &&
curl -fsSL -o codecov.SHA256SUM "https://cli.codecov.io/latest/$platform/codecov.SHA256SUM" &&
sha256sum -c codecov.SHA256SUM && chmod +x codecov || { echo "Codecov CLI download failed: skipping the Codecov upload."; exit 0; }

./codecov upload-process --disable-search --fail-on-error \
    --git-service github --slug "$slug" --sha "${CI_COMMIT_SHA}" "${ref_args[@]}" \
    --flag "$flag" --name "$flag: ${CI_JOB_NAME}" --file lcov.info --token "$CODECOV_TOKEN" ||
    echo "Codecov upload failed (ignored)."
