# Code-only publishing checklist

The repository is designed to show the pipeline without distributing the research data. The supplied [`.gitignore`](../.gitignore) blocks the private `analysis/data/` tree, environments, caches, logs, generated outputs, and serialized models.

## Before staging

Run source scans before `git add`:

~~~powershell
rg -n -F 'C:\Users\' analysis/config.py analysis/master.py analysis/prog
rg -n -i 'api[_-]?key|token|password|secret' analysis/config.py analysis/master.py analysis/prog
~~~

The original hard-coded Windows roots have been replaced with configuration derived from the checkout. Review every scan hit; variable names such as `token` may be harmless, while values resembling credentials or private paths must be removed.

Then initialize and use an explicit allow-list:

~~~powershell
git init -b main
git add .gitignore .gitattributes LICENSE README.md docs analysis/README.md analysis/data/README.md
git add analysis/config.py analysis/master.py analysis/requirements.txt analysis/prog
git status --short
git diff --cached --name-only
~~~

## Required checks

Confirm that no private data are staged:

~~~powershell
git ls-files analysis/data
~~~

The only expected results are:

~~~text
analysis/data/README.md
analysis/data/final/.gitkeep
analysis/data/raw/.gitkeep
analysis/data/temp/.gitkeep
~~~

Repeat the scan over tracked files:

~~~powershell
git grep -n -I -F "C:\Users\" -- analysis/config.py analysis/master.py analysis/prog
git grep -n -I -E "api[_-]?key|token|password|secret" -- analysis/config.py analysis/master.py analysis/prog
~~~

Do not commit while this command reports a real credential, private path, or embedded observation. Inspect notebooks, logs, screenshots, generated charts, and file metadata separately if any are added later.

## What should be public

- source code under `analysis/prog/`;
- the orchestrator and direct dependency specification;
- documentation and non-sensitive schema examples;
- the all-rights-reserved copyright notice in `LICENSE`.

## What should stay private

- raw, temporary, and final datasets;
- downloaded Google Trends histories;
- provider credentials, cookies, tokens, and local environment files;
- generated model binaries and large prediction artifacts;
- logs containing absolute paths or provider responses;
- third-party material without redistribution permission.

## GitHub presentation

After the content audit, add a short repository description such as:

> Mixed-frequency ML pipeline for high-frequency inflation tracking across 15 G20 economies using Google Trends, financial data, and expanding-window evaluation.

Useful repository topics include `inflation`, `nowcasting`, `time-series`, `lstm`, `xgboost`, `google-trends`, `mixed-frequency`, and `econometrics`.

Do not label the repository as fully reproducible until a clean-machine run with legally redistributable or automatically reacquired data has passed.

The current release is public for review but is not open source. Replacing `LICENSE` with a permissive license such as MIT would grant reuse rights and should be an explicit owner decision.
