from __future__ import annotations
from dataclasses import dataclass, field
import base64
from fnmatch import fnmatchcase
import os
from pathlib import Path
from typing import Any
import pymupdf
from docx import Document as WordDocument

TEXT_EXTENSIONS = {
    "txt", "md", "markdown", "csv", "tsv", "json", "jsonl", "xml",
    "html", "htm", "yaml", "yml", "toml", "ini", "log",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {"pdf", "docx"}

@dataclass
class Document:
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_text(path: str, *, encoding: str = "utf-8") -> list[Document]:
    source = os.fspath(path)
    with open(source, "r", encoding=encoding) as text_file: text = text_file.read()
    return [Document(text=text, source=source, metadata={"type": "text"})]


def load_docx(path: str) -> list[Document]:
    source = os.fspath(path)
    word_doc = WordDocument(source)
    content: list[str] = []
    layout: list[dict[str, Any]] = []
    for order, item in enumerate(word_doc.iter_inner_content()):
        if hasattr(item, "rows"):
            rows = [[cell.text for cell in row.cells] for row in item.rows]
            table_text = "\n".join("\t".join(row) for row in rows)
            content.append(table_text)
            layout.append({"order": order, "type": "table", "rows": rows})
        else:
            text = item.text
            if text: content.append(text)
            layout.append({"order": order,"type": "paragraph","text": text,"style": item.style.name if item.style else None})
    return [Document(text="\n".join(part for part in content if part),source=source,metadata={"type": "docx", "layout": layout})]

# Convcert from str to actual type
def _json_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)): return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)


# Get the text out of pymupdf block
def _block_text(block: dict[str, Any]) -> str:
    line_texts = []
    for line in block.get("lines", []):
        prev,parts = None,[]
        for span in line.get("spans", []):
            text = span.get("text", "")
            bbox = span.get("bbox")
            if prev is not None and bbox and prev.get("bbox"):
                gap = bbox[0] - prev["bbox"][2]
                size = max(float(prev.get("size") or 0), float(span.get("size") or 0))
                # Add space if gap is there between two spans
                if gap > max(1.0, size * 0.2) and text and not text[0].isspace(): parts.append(" ")
            parts.append(text)
            prev = span
        line_texts.append("".join(parts))
    return "\n".join(line_texts).strip()


def _layout_elements(page_no: int,blocks: list[dict[str, Any]],*,include_image_data: bool) -> list[dict[str, Any]]:
    elements = []
    for order, block in enumerate(blocks):
        block_type = block.get("type", 0)
        common = {"page": page_no,"order": order,"bbox": _json_value(block.get("bbox"))}
        if block_type == 0:
            # Text block
            lines = []
            for line in block.get("lines", []):
                spans = [{"text": span.get("text", ""),"bbox": _json_value(span.get("bbox")),"font": span.get("font"),"size": span.get("size"),"flags": span.get("flags")} for span in line.get("spans", [])]
                lines.append({"text": "".join(span["text"] for span in spans),"bbox": _json_value(line.get("bbox")),"direction": _json_value(line.get("dir")),"spans": spans})
            text = _block_text(block)
            if text: elements.append({**common, "type": "text", "text": text, "lines": lines})
        elif block_type == 1:
            # IMage block
            image = {"format": block.get("ext"),"width": block.get("width"),"height": block.get("height"),"colorspace": block.get("colorspace"),"bits_per_component": block.get("bpc"),"size_bytes": block.get("size")}
            if include_image_data and block.get("image"): image["data_base64"] = base64.b64encode(block["image"]).decode("ascii")
            elements.append({**common, "type": "image", "image": image})
        elif block_type == 3:
            # Vector 
            elements.append({**common,"type": "vector","stroked": block.get("stroked"),"is_rectangle_or_line": block.get("isrect"),"continues_path": block.get("continues"),"color": block.get("color"),"alpha": block.get("alpha")})
    elements.sort(key=lambda item: ((item.get("bbox") or [0, 0, 0, 0])[1],(item.get("bbox") or [0, 0, 0, 0])[0]))
    for order, element in enumerate(elements): element["order"] = order
    return elements


def _table_elements(page: Any, page_no: int) -> list[dict[str, Any]]:
    find_tables = getattr(page, "find_tables", None)
    if find_tables is None: return []
    elements = []
    try:
        table_finder = find_tables()
        for table_no, table in enumerate(table_finder.tables):
            rows = table.extract()
            elements.append({"page": page_no,"type": "table","table": table_no,"bbox": _json_value(getattr(table, "bbox", None)),"rows": [[cell if cell is not None else "" for cell in row] for row in rows]})
    except (RuntimeError, ValueError, TypeError): return []
    return elements


def load_pdf(path: str,*,ocr: bool = True,ocr_language: str = "eng",ocr_dpi: int = 300,include_image_data: bool = False) -> list[Document]:
    if ocr_dpi <= 0: raise ValueError("ocr_dpi must be a positive integer")
    source = os.fspath(path)
    documents: list[Document] = []
    with pymupdf.open(source) as pdf:
        page_count = len(pdf)
        for page_no, page in enumerate(pdf, start=1):
            flags = getattr(pymupdf, "TEXTFLAGS_DICT", 0)
            flags |= getattr(pymupdf, "TEXT_COLLECT_VECTORS", 0)
            native_data = page.get_text("dict", flags=flags, sort=True)
            blocks = native_data.get("blocks", [])
            native_text = page.get_text("text", sort=True).strip()
            extracted_text = native_text
            ocr_s = "NotNeeded"
            if ocr and len(native_text) < 40:
                ocr_s = "Attempted"
                try:
                    text_page = page.get_textpage_ocr(language=ocr_language,dpi=ocr_dpi,full=True)
                    ocr_data = page.get_text("dict", textpage=text_page, sort=True)
                    ocr_blocks = ocr_data.get("blocks", [])
                    ocr_text = page.get_text("text", textpage=text_page, sort=True).strip()
                    if ocr_text:
                        native_nontext = [block for block in blocks if block.get("type") != 0]
                        ocr_text_blocks = [block for block in ocr_blocks if block.get("type") == 0]
                        blocks = sorted([*ocr_text_blocks, *native_nontext],key=lambda block: ((block.get("bbox") or (0, 0, 0, 0))[1],(block.get("bbox") or (0, 0, 0, 0))[0]))
                        extracted_text = ocr_text
                        ocr_s = "Used"
                    else:
                        ocr_s = "NoTextFound"
                except Exception as error:
                    ocr_s = "Unavailable"
                    ocr_error = str(error)
                else:
                    ocr_error = None
            else:
                ocr_error = None
            layout = _layout_elements(page_no,blocks,include_image_data=include_image_data)
            layout.extend(_table_elements(page, page_no))
            layout.sort(key=lambda item: ((item.get("bbox") or [0, 0, 0, 0])[1],(item.get("bbox") or [0, 0, 0, 0])[0],item["type"]))
            for order, element in enumerate(layout): element["order"] = order
            p_meta: dict[str, Any] = {"type": "pdf","page_count": page_count,"page_width": float(page.rect.width),"page_height": float(page.rect.height),"ocr": {"status": ocr_s, "language": ocr_language if ocr_s != "not_needed" else None},"layout": layout}
            if ocr_error: p_meta["ocr"]["error"] = ocr_error
            if extracted_text.strip() or layout: documents.append(Document(text=extracted_text,source=source,page=page_no,metadata=p_meta))
    return documents


def load_file(path: str,ext: str | None = None) -> list[Document]:
    suffix = (ext or Path(path).suffix.lstrip(".")).lower()
    if suffix == "pdf": return load_pdf(path)
    elif suffix == "docx": return load_docx(path)
    elif suffix in TEXT_EXTENSIONS: return load_text(path)
    raise ValueError(f"Unsupported file extension: {suffix or '(none)'}")

def load_root(path: str,exclude: dict[str, str] | None = None,recursive: bool = True) -> dict[str, list[Document]]:
    root = Path(path)
    if not root.is_dir(): raise NotADirectoryError(os.fspath(root))
    patterns = tuple((exclude or {}).keys())
    elems = root.rglob("*") if recursive else root.iterdir()
    files = sorted((candidate for candidate in elems if candidate.is_file()),key=lambda candidate: candidate.relative_to(root).as_posix().casefold())
    loaded: dict[str, list[Document]] = {}
    for file_path in files:
        rel_path = file_path.relative_to(root).as_posix()
        if any(fnmatchcase(rel_path, pattern) or fnmatchcase(file_path.name, pattern) for pattern in patterns): continue
        extension = file_path.suffix.lstrip(".").lower()
        if extension not in SUPPORTED_EXTENSIONS: continue
        source = os.fspath(file_path)
        loaded[source] = load_file(source, extension)
    return loaded
