# Release runbook: protea-method v0.1.0 (LAFA-ready)

> Manual checklist for the maintainer. Each step is a single command or
> short block to run from the project root (`cd
> ~/Thesis2/repositories/protea-method`). **Nothing in this slice's PR
> executes any of these steps**; the docs are drafts only. Run them by
> hand when you are ready to publish.

## Pre-flight

The image is built from `develop` of `frapercan/protea-method` at the
commit that merges PR #19 (`LAFA-CONTAINER.1`), PR #20 (`F2C.6`), and
PR #21 (`LAFA-EMB.1`). As of this runbook draft that commit is
`6fdc2e0`. Re-verify before building (a later merge could have moved
the tip).

```bash
cd ~/Thesis2/repositories/protea-method
git fetch origin
git log -1 --oneline origin/develop
# Expect: 6fdc2e0 (or later) feat(LAFA-EMB.1)...
```

## Step 1. Pick a release tag

For the first LAFA-ready cut use `v0.1.0` (matches `pyproject.toml`).
Bump to `v0.1.1` for a doc-only follow-up, `v0.2.0` for any new flag or
backend, and `v1.0.0` only after the FARM-EXP.5 v226 regression-fixture
slice has shipped and the bind-mount path is locked.

```bash
RELEASE_TAG=v0.1.0
```

## Step 2. Build the image locally

```bash
cd ~/Thesis2/repositories/protea-method
docker build -t "frapercan/protea-method:${RELEASE_TAG}" .
docker tag    "frapercan/protea-method:${RELEASE_TAG}" frapercan/protea-method:latest
```

For the slim bind-mount-only variant:

```bash
docker build \
    --build-arg INSTALL_EXTRAS="" \
    -t "frapercan/protea-method:${RELEASE_TAG}-slim" .
```

## Step 3. Smoke test locally on a toy FASTA

Use the bundled example script against a small (5-protein) input set.
If you do not have one handy, take the first five records of
`train_sequences.fasta` and use them as both query and training set
just to verify the entrypoint runs end-to-end.

```bash
cd ~/Thesis2/repositories/protea-method
mkdir -p /tmp/lafa-smoke/data /tmp/lafa-smoke/output /tmp/lafa-smoke/hf-cache

# Drop a 5-protein test_sequences.fasta, a train_sequences.fasta, a small
# goa_uniprot_sprot.gaf.gz and a go-basic.obo into /tmp/lafa-smoke/data.

IMAGE="frapercan/protea-method:${RELEASE_TAG}" \
DATA_DIR=/tmp/lafa-smoke/data \
OUTPUT_DIR=/tmp/lafa-smoke/output \
HF_CACHE_DIR=/tmp/lafa-smoke/hf-cache \
BACKEND_ID=esm2_t33_650M \
    bash docker/example_run_selfcontained.sh

head -5 /tmp/lafa-smoke/output/predictions.tsv.gz | zcat
```

Expected: three tab-separated columns per row (Query_ID, GO_Term,
Score), no header, scores in `[0, 1]`. `esm2_t33_650M` is the lighter
backend; switch to the champion `esm2_t36_3B` for the
reproducibility-grade smoke test.

If the smoke test fails, fix and re-tag before pushing. Do not push a
broken image; DockerHub does not let you unpublish a tag without
deleting the repository.

## Step 4. DockerHub login

```bash
docker login docker.io
# Username: frapercan
# Password: <personal access token from hub.docker.com/settings/security>
```

Use a Personal Access Token (Read, Write, Delete scoped to the
`protea-method` repository), not the account password. Never commit
the token; the `~/.docker/config.json` file already stores it base64
encoded.

## Step 5. Push the image

```bash
docker push "frapercan/protea-method:${RELEASE_TAG}"
docker push  frapercan/protea-method:latest
# (slim variant, if built)
docker push "frapercan/protea-method:${RELEASE_TAG}-slim"
```

Note the `digest: sha256:...` line emitted at the end of each push.
Save it; it goes into the FunctionBench form in step 8.

## Step 6. Pull and verify the digest

From a clean shell (or another machine) confirm the digest pins to a
single immutable image:

```bash
docker pull "frapercan/protea-method:${RELEASE_TAG}"
docker image inspect "frapercan/protea-method:${RELEASE_TAG}" \
    --format '{{index .RepoDigests 0}}'
# -> docker.io/frapercan/protea-method@sha256:...
```

The digest from the inspect output must match the one printed during
`docker push`.

## Step 7. Update the DockerHub repository description

On https://hub.docker.com/r/frapercan/protea-method/general (Settings
tab), paste the contents of `docker/DOCKERHUB_README.md` into the "Full
description" field. The "Short description" should be a one-liner; the
suggested text is:

> LAFA submission image for PROTEA. KNN over PLM embeddings + per-aspect LightGBM rerankers + GO ancestor propagation.

Save.

## Step 8. Submit on functionbench.net

1. Sign in at https://functionbench.net/ (account associated with
   `frapercan1@gmail.com`).
2. Open the "Submit a method" form.
3. Paste the contents of `docker/FUNCTIONBENCH_METHODCARD.md` into the
   long-form description field. Fill the structured fields (name,
   short id, image, version, contact) from the Identification table
   at the top of that file.
4. In the image-pin field paste the `sha256:...` digest recorded in
   step 6.
5. If the form asks for a reference annotation file
   (`train_terms.tsv`), upload the GAF.GZ used during the LB.2
   validation run; the same file is bundled with the LAFA
   `Sep_2025` reference release on HuggingFace
   (`anphan0828/lafa`).
6. Submit.

## Step 9. Tag a GitHub release

```bash
cd ~/Thesis2/repositories/protea-method
git fetch origin
git tag -a "${RELEASE_TAG}" \
    -m "${RELEASE_TAG}: first LAFA-ready cut. KNN + per-aspect rerankers + GO propagation." \
    origin/develop
git push origin "${RELEASE_TAG}"
```

Then on https://github.com/frapercan/protea-method/releases/new pick
the tag, copy the relevant section of the changelog into the release
notes, and link the DockerHub image (`frapercan/protea-method:v0.1.0`)
with the pinned digest.

## Step 10. Sanity-check after publication

```bash
# Pull on a fresh host (no local layers) and re-run the smoke test.
docker pull "frapercan/protea-method:${RELEASE_TAG}"
bash docker/example_run_selfcontained.sh
```

If FunctionBench provides a self-test endpoint, run it. Otherwise wait
for the grader to email back; expect a 24 to 72 hour latency before
results appear on the leaderboard.

## Rollback

DockerHub does not allow re-publishing a deleted tag for several days.
If a bad image was pushed:

* Build and push a new tag (`v0.1.1`) with the fix.
* Update the FunctionBench submission to point at the new digest.
* Add a note to the DockerHub description explaining the deprecation.

Do **not** force-push tags on GitHub; cut a `v0.1.1` and supersede the
release notes there too.
