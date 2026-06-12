# Publishing `ariadx` to PyPI

> Internal guide — follow in order. Total time ~30 minutes the first time, ~5 minutes for later releases.

---

## 0. One-time account setup

1. Create an account at https://pypi.org/account/register/ (and https://test.pypi.org for rehearsal).
2. Enable 2FA (required by PyPI for new uploads).
3. Create an **API token**: PyPI → Account settings → API tokens → "Add API token"
   - First upload: scope = "Entire account" (you can't scope to a project that doesn't exist yet).
   - After the first upload: create a new token scoped to `ariadx` and delete the broad one.
4. Save the token — it is shown only once. It looks like `pypi-AgEIcHlwaS5vcmc...`

---

## 1. Pre-publish checklist

- [ ] `pyproject.toml` version bumped (PyPI never allows re-uploading the same version)
- [ ] `PYPI_README.md` is current (this is the PyPI landing page)
- [ ] `LICENSE` file exists at repo root (Apache 2.0)
- [ ] No secrets anywhere in the package tree (`.env` is excluded; double-check)
- [ ] SDK smoke test passes:
  ```powershell
  cd C:\Users\ASUS\Desktop\ARIA\backend
  C:\Users\ASUS\anaconda3\envs\aria\python.exe -c "from aria.sdk import diagnose; print('SDK OK')"
  ```

---

## 2. Build the distribution

```powershell
cd C:\Users\ASUS\Desktop\ARIA\backend

# Install build tooling (once)
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m pip install --upgrade build twine

# Clean previous builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Build sdist + wheel
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m build
```

This produces:
- `dist/ariadx-0.1.0.tar.gz` (source distribution)
- `dist/ariadx-0.1.0-py3-none-any.whl` (wheel)

**Verify the wheel contents** (no data files, no .env, no scripts/):

```powershell
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m zipfile -l dist/ariadx-0.1.0-py3-none-any.whl
```

You should see only `aria/`, `api/`, `adapters/` modules + metadata. If `data/` or anything private appears, STOP and fix `[tool.setuptools.packages.find]`.

---

## 3. Validate metadata

```powershell
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m twine check dist/*
```

Both files must report `PASSED`.

---

## 4. Rehearsal upload to TestPyPI (recommended first time)

```powershell
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m twine upload --repository testpypi dist/*
# username: __token__
# password: <your TestPyPI token>
```

Then verify install in a clean environment:

```powershell
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ariadx
C:\Users\ASUS\anaconda3\envs\aria\python.exe -c "from aria.sdk import diagnose; print('install OK')"
```

---

## 5. Real upload to PyPI

```powershell
C:\Users\ASUS\anaconda3\envs\aria\python.exe -m twine upload dist/*
# username: __token__
# password: <your PyPI token>
```

Done — the package is live at `https://pypi.org/project/ariadx/` within a minute, and anyone can:

```bash
pip install ariadx
```

---

## 6. Post-publish

- [ ] Create a project-scoped API token, delete the account-wide one
- [ ] Tag the release in git: `git tag v0.1.0; git push origin v0.1.0`
- [ ] Create a GitHub Release with changelog notes
- [ ] Update the README install instructions to `pip install ariadx`

---

## Releasing a new version later

1. Bump `version` in `pyproject.toml` (semver: `0.1.1` fixes, `0.2.0` features)
2. `Remove-Item -Recurse -Force dist; python -m build; python -m twine check dist/*; python -m twine upload dist/*`

---

## Notes / gotchas

- **The import name stays `aria`** (`from aria.sdk import diagnose`) while the PyPI name is `ariadx`. This is normal (like `pip install beautifulsoup4` → `import bs4`). If a conflicting `aria` module ever appears in dependencies, rename the package directory in a major release.
- **PyPI versions are immutable** — you can "yank" a release but never replace it. Always test on TestPyPI first.
- **The PyPI page README** is `backend/PYPI_README.md` — keep it short and product-focused; the full docs live on GitHub.
