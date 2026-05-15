# GitHub Setup

The ChatGPT GitHub connector available in this session can write to existing repositories, but it did not expose a create-repository operation.

To create and push this repo manually:

```bash
cd metabolic-intelligence-lab
git init
git add .
git commit -m "Initial metabolic intelligence lab scaffold"
gh repo create lbsage/metabolic-intelligence-lab --private --source=. --remote=origin --push
```

Or create `metabolic-intelligence-lab` on GitHub first, then:

```bash
git remote add origin git@github.com:lbsage/metabolic-intelligence-lab.git
git branch -M main
git push -u origin main
```
