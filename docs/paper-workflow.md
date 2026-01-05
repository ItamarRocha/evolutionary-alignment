# Paper Workflow (Overleaf)

The paper lives in `paper/` as a separate git repo (gitignored from main repo) synced with Overleaf.

## Setup (one-time)

The remote is configured to use `$OVERLEAF_API_KEY` from `~/.bashrc`:

```bash
source ~/.bashrc
cd paper
git remote set-url origin "https://git:${OVERLEAF_API_KEY}@git.overleaf.com/69432e82f368efcf6edce21e"
```

## Push local changes to Overleaf

```bash
cd paper
git add -A
git commit -m "your message"
git push
```

## Pull collaborator changes from Overleaf

```bash
cd paper
git pull
```

## If you get merge conflicts

```bash
cd paper
git pull --rebase
# resolve conflicts, then:
git add .
git rebase --continue
git push
```

## Notes

- The `paper/` directory is gitignored from the main `evolutionary-alignment` repo
- Each repo is independent - commit to paper for LaTeX, commit to main for code
- Overleaf auto-compiles on push, check the PDF there after pushing
