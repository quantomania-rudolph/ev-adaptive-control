# Contributing

The whole workflow, in the order you'll do it. (New to Git? Read `docs/GITHUB_SETUP.md` first.)

## Every time you start work

```bash
git checkout main
git pull                          # get everyone else's latest changes
git checkout -b yourname/short-description    # e.g. sam/add-rate-limit
```

You never work directly on `main`. A **branch** is your private copy of the timeline, so a half-finished
idea can't break anyone else.

## While you work

- Edit files inside your segment only.
- Run `make test` often. It takes seconds.
- Before committing: `make format` (auto-fixes style) then `make ci`.

## Sharing it

```bash
git add src/evcontrol/actuator tests/actuator
git commit -m "Add speed-dependent torque limit"
git push -u origin yourname/short-description
```

Then open the link Git prints (or go to the repo on GitHub and click **Compare & pull request**).
Fill in the template. GitHub runs the pipeline automatically; green checks mean you're good to merge
after a teammate approves.

## Rules of thumb

- **Small pull requests.** One idea each. Easier to review, easier to undo.
- **A change without a test is unfinished.** Even one test is fine.
- **Never commit data files, secrets, or passwords.** `.gitignore` blocks common data formats.
- **Don't edit someone else's segment** without telling them. Open an issue instead.
- **Commit messages** say *what and why* in plain English: "Clip torque request to motor limit", not "fix stuff".
- Red pipeline? Click the failing job on the PR, read the last lines. It prints exactly which test failed.
