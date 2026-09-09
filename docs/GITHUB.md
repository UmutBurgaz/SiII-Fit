# Working with this repository

Repository: [UmutBurgaz/Silicon_github](https://github.com/UmutBurgaz/Silicon_github). Keep visibility private while validating the measurements.

## Your local copy

The working folder is `D:\Codex\Silicon_github`. The repository contains source, tests, documentation, and the four small example spectra. It excludes virtual environments, build files, caches, credentials, and generated results.

Open PowerShell in the folder to inspect changes:

```powershell
git status
git diff
```

Before saving a code change, run the checks using the prepared environment:

```powershell
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

To save a specific reviewed change and upload it:

```powershell
git add path/to/changed_file.py
git commit -m "Describe the change"
git push
```

Git may ask you to sign in to GitHub the first time you push from the command line. The initial upload used the connected GitHub app; browser/app sign-in does not itself configure command-line authentication. Never paste an access token into project files or commit messages.

On another computer, sign in with access to the private repository and clone it:

```bash
git clone https://github.com/UmutBurgaz/Silicon_github.git
cd Silicon_github
```

Then follow the installation instructions in the root README. The Actions tab shows automated checks after code uploads. Before changing visibility to public, complete the remaining items in [the review](REVIEW.md); no public release is part of this initial upload.
