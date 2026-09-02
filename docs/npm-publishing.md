# Publishing the JS package to npm — durable instructions

The JS half of this project lives in `js/` (npm package name:
**`keras-tinygrad`**) and publishes via `.github/workflows/npm-publish.yml`
on `js-v*` tags. This document is written to outlive any particular setup:
it covers first-time setup, re-setup after losing access, and what to do
if the name situation changes. Written 2026-08-30; where npm's UI has
moved since, the *sequence* still holds — verify details against
docs.npmjs.com (search: "trusted publishers").

## The model (why it's shaped this way)

- npm has NO name reservation: **publishing is registering.** Empty
  placeholder packages violate npm's squatting policy and can be disputed
  away — the way to hold a name is a minimal-but-real release.
- Steady state is **Trusted Publishing** (OIDC from GitHub Actions):
  no token exists anywhere, ever. Same discipline as the PyPI setup.
- Trusted publishing generally requires the package to already exist, so
  the FIRST publish bootstraps with a short-lived token that never
  touches a dev box (browser → GitHub secret → used once → revoked).

## First-time publish (or re-publish after starting over)

1. **Browser** — npmjs.com: create/log into the account; enable **2FA**
   (publishing requires it).
2. **Browser** — npmjs.com → Access Tokens → Generate New Token →
   **Granular**: scope = publish, only-selected-packages if offered,
   shortest expiry available.
3. **Browser** — GitHub → this repo → Settings → Secrets and variables →
   Actions → new repository secret **`NPM_TOKEN`** = that token.
   (The token exists only in the npm UI and GitHub's secret store.)
4. **GitHub** — Settings → Environments → create **`npm`**; optionally
   add required reviewers for a one-click approval gate per publish.
5. **Any box with the repo** — make sure `js/package.json`'s `version`
   is what you intend, commit everything, then:
   `git tag js-v<version> && git push origin js-v<version>`
   The workflow verifies (tag↔version guard, module smoke, the LCG
   Python-parity check, pack dry-run) and publishes with
   `npm publish --provenance` using the secret.
6. **Browser** — confirm npmjs.com/package/keras-tinygrad is live and
   shows the provenance attestation.
7. **Browser** — package Settings → Trusted Publisher → add GitHub
   Actions: org/user `c4ffein`, repo `keras-tinygrad`, workflow
   **`npm-publish.yml`** (filename must match exactly), environment
   `npm`.
8. **Browser** — DELETE the `NPM_TOKEN` GitHub secret and REVOKE the
   token on npmjs.com. From the next tag on, publishes are pure OIDC.
9. Sanity, any box: `bun add keras-tinygrad` in a scratch dir,
   `import { lcg32 } from "keras-tinygrad"`.

## Subsequent releases (steady state)

Bump `version` in `js/package.json` → commit → `git tag js-v<version> &&
git push origin js-v<version>`. Nothing else. npm versions are immutable:
a broken release means a patch bump (or `npm deprecate`), never a
re-upload.

## Requirements pinned in the workflow (update if npm moves them)

- npm CLI ≥ 11.5.1 and Node ≥ 22.14 for trusted publishing — the
  workflow uses Node 24 (bundles npm 11). If a "Set up job" or publish
  step fails on versions, check these first.
- The trusted-publisher binding is to the WORKFLOW FILENAME
  (`npm-publish.yml`) and the `npm` environment — renaming either breaks
  publishing until reconfigured on npmjs.com.

## If access is lost (account, 2FA, token chaos)

The trusted-publisher config lives on npmjs.com under the package — with
account recovery done, publishing resumes with zero repo changes. If the
GitHub repo moved (rename/org), update the trusted-publisher entry to
the new location; the workflow itself is location-agnostic.

## If the name is taken by someone else

Checked free on 2026-08-29 and 2026-08-30 — but there is no reservation,
so if someone registers `keras-tinygrad` first:

1. Don't squat-war. Check what they published: if it's an empty/spam
   placeholder, npm's package name dispute policy
   (docs.npmjs.com → "package name disputes") applies — open a dispute;
   these are routinely granted against empty packages.
2. If it's a real (even hostile) package, fall back to the scoped name
   **`@c4ffein/keras-tinygrad`** — scopes are namespaced to the account
   and cannot be taken. Update `js/package.json` `name`, the README, and
   the trusted-publisher config; everything else is unchanged.
3. Either way, note the situation in this file with the date.

## Relationship to the Python package

Same project, same repo, two registries: `pip install keras-tinygrad`
(the backend + exporter) and `npm install keras-tinygrad` (the runner —
loads exported WebGPU bundles; roadmap: in-tab tracing). The `lcg32`
helper is bit-identical across both (CI enforces it against baked
values); treat any drift as a breaking change on either side.
