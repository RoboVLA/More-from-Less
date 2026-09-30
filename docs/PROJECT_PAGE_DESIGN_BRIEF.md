# Project page

Updated 2026-09-22 to follow the academic paper-page layout commonly used by computer-vision conference projects. References: [Nerfies](https://nerfies.github.io/) and [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template). This describes the presentation style, not a claim of conference acceptance.

- Center the complete paper title, an anonymous-review note, and dark rounded Paper / Code / Videos / BibTeX buttons.
- Place two existing real-task clips beneath the publication header, side by side on desktop and stacked on phones. Playback is user controlled.
- Use white and light-gray sections, neutral text, blue links, and restrained decoration. Keep CSS, JavaScript, icons, and media local.
- Follow Abstract → Method Overview → Experiments → Code & Resources → BibTeX. Display all three method figures continuously, rather than hiding them in tabs.
- Preserve the manuscript figures, clips, citation, and Table 1 / Table 4 cells. Keep their different statistical units explicit; result tabs switch only the displayed table.
- Retain the source browser and reproducibility notes. Do not add guessed GitHub/arXiv links, publication badges, or experimental claims.

Entry files: `index.html`, `code.html`, `static/css/project.css`, and `static/js/project.js`. The preview remains at `http://127.0.0.1:18765/`; port 8765 belongs to another project. No GitHub upload or manuscript edit accompanies this style change. Attribution and license boundaries are recorded in `THIRD_PARTY_NOTICES.md`.
