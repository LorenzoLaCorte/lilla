# Deploy the repeater explorer

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FLorenzoLaCorte%2Flilla)

1. Commit and push the current visualization implementation and deployment files
   to GitHub. Render builds the code on GitHub, including the bundled tree editor;
   it cannot see uncommitted local files.
2. Click **Deploy to Render** above, sign in, and connect the repository. For a
   private repository, grant Render's GitHub app access to it. Alternatively use
   **New → Blueprint**, select this repository, and keep `render.yaml` as the path.
3. Review and deploy. Open the service's HTTPS URL when Render reports **Live**.
   No database, API keys, Node build, or GPU setup is required.

The root [render.yaml](../../render.yaml) supplies the build command, launch
command, free compute plan, environment, and `/_stcore/health` health check.
The root `.python-version` selects Python 3.11 with Render's latest matching
patch release. Dependencies come from [requirements-deploy.txt](../../requirements-deploy.txt),
which includes the UI, numerical backend, and Bayesian optimization without the
full research environment. Single runs, sweeps, optimization, inline tree edits,
saved-run comparisons, and downloads use the same app as locally.

Automatic redeployment is off. After pushing an update, use the service's
**Manual Deploy → Deploy latest commit**. Enable automatic deployment in Render
if desired. For a different branch or a fork, use **New → Blueprint** to select it.

## Run the same launch command locally

From the repository root, once a Python 3.10 or 3.11 virtual environment is active:

```bash
python -m pip install -r requirements-deploy.txt
sh src/visualization/start.sh
```

Open <http://localhost:8501>. After installation, launching is one command.
The launcher defaults to localhost. On another hosting service, set
`LILLA_BIND_ADDRESS=0.0.0.0` and its `PORT` environment variable; it then listens
on that interface and port. Keep Streamlit's default CORS and XSRF protections.

## Saved results

| Environment | Default save directory | Lifetime |
|---|---|---|
| Local | `results/visualization/` | Remains on your computer |
| Render blueprint | `/tmp/lilla-results` | Lost on redeploy, restart, or free-tier spin-down |
| Custom storage | `LILLA_RESULTS_DIR` | Depends on the mounted storage |

The free service is suitable for a demo. It sleeps after inactivity and can take
time to wake up; download results you want to keep. Larger horizons, broad sweeps,
or concurrent users may require a larger compute plan. All visitors to one
deployment share its saved-run library; the app has no built-in user accounts.

For persistent saved runs, choose a paid service plan, attach a Render persistent
disk at `/var/data`, and set `LILLA_RESULTS_DIR=/var/data/visualization`.
Only files under the disk's mount path persist. If maintaining this through the
Blueprint, change `plan` to the paid plan you selected, change the results
environment value, and add this under the service:

```yaml
disk:
  name: visualization-results
  mountPath: /var/data
  sizeGB: 1
```

The CLI uses the same `LILLA_RESULTS_DIR` default; `--output PATH` still takes
precedence. Existing local results are not uploaded during deployment.

## Quick checks

The health endpoint should return `ok`:

```bash
curl --fail https://YOUR-SERVICE.onrender.com/_stcore/health
```

For an end-to-end check, load `src/visualization/examples/smoke.json` in the app,
press **Run**, and confirm SKR is approximately `0.215706` bits/slot. Reload its
saved run and edit a tree node. Load the supplied `sweep.json` or `optimize.json`
through the CLI to check those modes with the same deployment dependencies.

Provider references: [Render deploy button](https://render.com/docs/deploy-to-render),
[Python versions](https://render.com/docs/python-version),
[port binding](https://render.com/docs/web-services#port-binding),
[free services](https://render.com/docs/free), and
[persistent disks](https://render.com/docs/disks).

Validated locally with a clean Python 3.10 environment installed only from
`requirements-deploy.txt`: single run, three-point sweep, four-trial Bayesian
optimization, and a Streamlit UI run/save passed. The launch script served HTTP
200 at both `/` and `/_stcore/health`, using a host-supplied port and a working
directory outside the repository. All 96 visualization tests passed, including
the environment-based save directory and explicit CLI output override. The
configured Render runtime is Python 3.11; a live deployment still requires
pushing the code and connecting your Render account as above.
