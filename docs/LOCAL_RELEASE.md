# Release and hosting

- Public website: [robovla.github.io/More-from-Less](https://robovla.github.io/More-from-Less/).
- Repository: [RoboVLA/More-from-Less](https://github.com/RoboVLA/More-from-Less).
- Clone: `git clone https://github.com/RoboVLA/More-from-Less.git`.
- GitHub Pages publishes the root of `main`. Site assets use relative paths so the project subdirectory works correctly.

The website temporarily omits the manuscript download entry. The retained anonymous PDF is unchanged; hiding a button does not revoke access to previously published files or URLs.

Current project pages, documentation and public manuscript metadata omit author names, affiliations and contact details. Current main-branch commits use an anonymous identity. This does not guarantee that provider caches, historical commit URLs or contributor records are anonymous. Third-party citations and required copyright notices are retained.

## Local preview

```bash
python scripts/serve.py --port 18765
```

Use the loopback URL printed in the terminal only on the machine running the server. It is not a shareable public project URL. If occupied, choose another port; do not stop unrelated services. The server binds to loopback, blocks hidden paths and directory listings, and does not deploy.

Use an anonymous commit identity for future public updates. Do not merge old author-identifying history into the review branch. Keep private backups outside the public repository.
