# GitHub setup: from zero to a shared team repo

Written for someone who has never done this. About 20 minutes. **Only the repo admin (you) does Parts 1-3 once.**
Teammates only do Part 4.

## Vocabulary (the five words you need)

| Word | Plain meaning |
|------|---------------|
| **Repository (repo)** | The project folder plus its full history of changes |
| **Commit** | A saved snapshot with a message describing it |
| **Branch** | A separate line of work so you don't disturb `main` |
| **Pull request (PR)** | "Please merge my branch into main": where review and the automatic tests happen |
| **CI/CD** | Robots that run your tests (CI) and package releases (CD) on every change |

## Part 1: One-time setup on your computer

1. Make a free account at <https://github.com>.
2. Install **Git** from <https://git-scm.com/downloads>.
3. Tell Git who you are (once per computer):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"   # use the same email as your GitHub account
   ```
4. Easiest login method: install **GitHub Desktop** (<https://desktop.github.com>) *or* the GitHub CLI (<https://cli.github.com>, then run `gh auth login`). Either one handles passwords/tokens for you. Don't try to type your GitHub password into the terminal; GitHub no longer accepts it.

## Part 2: Create the shared repo and upload this folder

1. On github.com click **+ -> New repository**.
   - Name: `ev-adaptive-control`
   - Owner: your **university/lab GitHub organization** if you have one (better for long-term), otherwise your account.
   - **Private** (research code; you can open it later).
   - Do **not** tick "Add a README/.gitignore/license". This folder already has them.
2. Open a terminal **inside the `ev-adaptive-control` folder** and run:
   ```bash
   git init -b main
   git add .
   git commit -m "Initial pipeline skeleton"
   git remote add origin https://github.com/YOUR-NAME-OR-ORG/ev-adaptive-control.git
   git push -u origin main
   ```
   (GitHub shows these exact commands, with your URL filled in, on the empty repo page.)
3. Click the **Actions** tab. You should see the **CI** workflow running. Green check = the skeleton works on GitHub's machines, not just yours.

## Part 3: Add teammates and protect `main`

1. **Invite people:** repo **Settings -> Collaborators** (or **Manage access**) -> **Add people**. They need GitHub accounts; give them *Write* access.
2. **Protect main** (this is what makes it a real CI/CD gate): **Settings -> Branches -> Add branch ruleset** (or "Add rule"):
   - Branch name pattern: `main`
   - Require a pull request before merging (1 approval)
   - Require status checks to pass -> select `Lint (ruff)`, `Integration tests` once they've run at least once
   - Block force pushes
3. **Fill in `.github/CODEOWNERS`** with real GitHub usernames (replace every `@REPLACE-...`), and the owner names in `README.md`.
4. Optional: **Settings -> General -> Pull Requests** -> tick "Automatically delete head branches" to keep things tidy.

> Note: branch protection on **private** repos needs a paid GitHub plan for individuals. It's free on public repos, and GitHub Education / university organizations often include it. If it's unavailable, the pipeline still runs and shows red/green; you just agree as a team not to merge red PRs.

## Part 4: What each teammate does (send them this)

1. Accept the email invite from GitHub.
2. Get the code:
   ```bash
   git clone https://github.com/YOUR-NAME-OR-ORG/ev-adaptive-control.git
   cd ev-adaptive-control
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   make install
   make test
   ```
   (GitHub Desktop users: **File -> Clone repository**.)
3. Read `CONTRIBUTING.md`. Branch, commit, push, open a PR.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `git push` asks for a password / rejected | Log in via GitHub Desktop or `gh auth login` |
| "rejected ... fetch first" | Someone pushed before you: `git pull`, then push again |
| Merge conflict | Two people edited the same lines. Open the file, keep the right version between the `<<<<` / `>>>>` markers, then `git add` and `git commit`. Segmented folders make this rare |
| CI red on lint only | Run `make format`, commit, push |
| Pushed a big data file by mistake | Tell the admin **before** more commits pile on. Removing it from history is painful |
| `make: command not found` (Windows) | Use Git Bash/WSL, or run the command shown in the Makefile directly |

## Releasing a frozen version (the CD part)

When results are worth freezing (a paper figure, a milestone):
```bash
git checkout main && git pull
git tag v0.2.0
git push origin v0.2.0
```
`release.yml` runs the unit tests, then the integration tests, builds the package, and attaches that build to a GitHub Release.

A unit test has to pass. A skipped or failing unit test stops the release. The three placeholder tests are skipped on purpose, so a plain tag will not publish until those tests are real or you explicitly override the gate. An override is an annotated tag whose message starts with a reason:
```bash
git tag -a v0.2.0 -m "Override-Unit-Tests: placeholder segment tests are still skipped"
git push origin v0.2.0
```
The Actions tab can run the same checks without publishing: **Release → Run workflow**. Set "Deploy even though unit tests failed or were skipped" and fill in the reason.
