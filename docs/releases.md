# Release process

SAMStars uses one publishing workflow for stable and experimental releases.
Publishing a GitHub release triggers `.github/workflows/workflow.yml`, which
builds and checks the distributions, runs the full CI suite against the tagged
commit, and publishes to PyPI only when every check passes. Merging into `main`,
pushing a tag, or saving a draft release does not publish to PyPI.

## Prepare a release

1. Create a short branch from the latest `main`.
2. Update `[project].version` in `pyproject.toml`. Use `0.3.0a1` for the first
   alpha, `0.3.0a2` for the next alpha, and `0.3.0` for the eventual stable release.
3. Open a pull request with the version change and relevant documentation. Wait
   for CI to pass, then merge it into `main`.
4. Create and push an annotated tag on the merged commit. The tag must be `v`
   followed by the exact package version, for example `v0.3.0a1`.
5. Publish a GitHub release for that tag, with short notes describing the changes.
   Mark alpha, beta, release candidate, and development versions as **prereleases**.
   Leave that option off for stable versions.
6. Wait for **Publish to PyPI** to succeed. Check that the package appears on PyPI
   and verify its version and import from a separate environment.

For example, after the release PR has merged:

```bash
git switch main
git pull --ff-only
git tag -a v0.3.0a1 -m "samstars 0.3.0a1"
git push origin v0.3.0a1
```

Then publish the GitHub prerelease. The workflow rejects tags that do not match
`pyproject.toml`, and rejects a GitHub prerelease flag that disagrees with the
Python package version. The existing PyPI trusted publisher uses this workflow
and the `pypi` environment; no separate alpha publisher is needed.

Keep published versions and tags fixed. Further changes belong in a new version,
such as `0.3.0a2`.

## Test an experimental release

Within a separate environment using a supported Python version, follow the
[installation instructions](installation.md) for CPU or GPU dependencies, then
install the exact alpha:

```bash
python -m pip install --upgrade "samstars==0.3.0a1"
python -m pip check
python -c "from importlib.metadata import version; from samstars import SamStars; print(version('samstars'))"
```

Use an explicit version to opt into a particular alpha. Ordinary installations
continue to select stable releases by default. The `a1` suffix identifies the
Python prerelease; an `experimental` PR label or GitHub release title alone does
not change how pip selects versions.
