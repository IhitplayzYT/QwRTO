# QwRTO document ingestion

The ingestors in `src.ingestor` return a list of `Document` objects. PDFs
produce one document per page. Plain-text and DOCX files produce one document
per file. Each document has `text`, `source`, an optional `page`, and a
`metadata` dictionary.

```python
from src.ingestor import load_docx, load_pdf, load_text

pages = load_pdf("report.pdf", ocr=True)
plain_text = load_text("notes.txt")
word_doc = load_docx("report.docx")
```

`load_root("documents", recursive=True)` loads supported files from the
directory tree and returns a mapping from each file path to its list of
documents. Set `recursive=False` to load only files directly inside the root.
It handles PDF, DOCX, and common text formats such as TXT, Markdown, CSV, JSON,
HTML, YAML, TOML, and log files. Unsupported extensions are skipped. Pass an
`exclude` mapping to skip matching relative paths or basenames, for example
`exclude={"drafts/*": "draft files"}`.

`Document.text` contains searchable text. PDF page metadata also contains
positioned elements so a downstream indexer can preserve page structure.
DOCX metadata contains paragraphs and tables in source order.

```python
from src.ingestor import load_pdf

pages = load_pdf("report.pdf", ocr=True)
for page in pages:
    print(page.page, page.text)
    for element in page.metadata["layout"]:
        print(element["type"], element["bbox"], element["page"])
```

Layout entries have a `type` of `text`, `table`, `image`, or `vector`. Entries
include their one-based page number and a bounding box in PyMuPDF page
coordinates (points, origin at the top left). Text entries retain line and span
coordinates and basic font details. Table entries contain extracted rows. Image
entries contain format, dimensions, color information, and byte size; set
`include_image_data=True` to additionally include base64 image data. Keep that
option off when writing metadata into a vector database, since image payloads
can be large. Vector entries describe detected vector blocks.

The layout array is spatially ordered from top to bottom and left to right.
For multi-column pages, use each element's bounding box to group columns before
building a reading sequence; a single global text order cannot reliably capture
every publication's column flow. Page text is still included as a convenient
search string, and the element coordinates preserve the information needed for
more specialized reconstruction.

OCR is attempted for pages with fewer than 40 extractable characters. Install
Tesseract and the selected language data (English by default) to enable it.
OCR status is recorded under `metadata["ocr"]`; if the OCR engine is unavailable,
native PDF extraction still returns normally.
