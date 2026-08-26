# AlphaRat agent instructions

## Working environment

- Do all source edits, builds, tests, profiling, and benchmarks in a native WSL checkout. On this workstation, use Ubuntu 20.04 at `/home/minh-tri/projects/alpharat`.
- On a replacement workstation, create an equivalent checkout in the WSL Linux filesystem before changing source. Do not use a repository under `/mnt/*` as the active worktree.
- From Windows, invoke repository work through WSL. Treat the Windows checkout as coordination or recovery storage, not as a source-working copy.
- Use Windows directly only for work that inherently requires a Windows application or session, such as browser coordination. State the exception before touching source.
- Before changing source, verify the WSL path, branch, and worktree status. Keep dirty sibling worktrees separate and never absorb them implicitly.

## Project context

- Read `.mt/CLAUDE.md` before using project memory or making planning, review, documentation, UX, or project-context decisions.
