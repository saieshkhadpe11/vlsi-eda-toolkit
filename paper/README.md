# Paper — Preprint

This folder contains the LaTeX source for the technical preprint:

**"VLSI EDA Toolkit: A Python Framework for Physical Design Automation with a Novel Physics-Inspired Agent-Based Floorplanner"**

- **Author**: Saiesh Khadpe (ORCID: [0009-0003-7524-9927](https://orcid.org/0009-0003-7524-9927))
- **Zenodo DOI**: [10.5281/zenodo.18693046](https://doi.org/10.5281/zenodo.18693046)

## How to Compile

### Option A — Overleaf (Easiest, no install needed)
1. Go to [overleaf.com](https://www.overleaf.com)
2. Click **New Project → Upload Project**
3. Upload `preprint.tex`
4. Overleaf will compile it automatically → click **Download PDF**

### Option B — Local (requires LaTeX)
```bash
pdflatex preprint.tex
pdflatex preprint.tex   # Run twice to resolve references
```

## What to do with the PDF

Once compiled, upload `preprint.pdf` to Zenodo as a **new upload** with:
- `upload_type`: `publication`
- `publication_type`: `technicalnote`

This gives you a **second, separate DOI** for the paper itself.
