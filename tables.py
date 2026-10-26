"""Represent simple HTML tables as labeled text rows."""

from bs4 import NavigableString


def format_simple_tables(soup):
    """Replace simple tables in place; leave complex tables untouched."""
    seen = set()
    promoted = []

    for table in list(soup.find_all("table")):
        if table.find_parent("table") is not None:
            continue

        html_rows = table.find_all("tr")
        if not html_rows:
            continue

        cell_rows = [
            tr.find_all(["th", "td"], recursive=False)
            for tr in html_rows
        ]
        if any(not cells for cells in cell_rows):
            continue

        # A single header row and equally wide data rows are required.
        if not any(cell.name == "th" for cell in cell_rows[0]):
            continue
        width = len(cell_rows[0])
        if width < 2 or any(len(cells) != width for cells in cell_rows[1:]):
            continue
        if any(
            cell.get("colspan", "1") != "1"
            or cell.get("rowspan", "1") != "1"
            for cells in cell_rows
            for cell in cells
        ):
            continue

        rows = [
            [cell.get_text(" ", strip=True) for cell in cells]
            for cells in cell_rows
        ]
        if len(rows) < 2 or any(not header for header in rows[0]):
            continue

        signature = tuple(tuple(row) for row in rows)
        if signature in seen:
            table.decompose()
            continue
        seen.add(signature)

        headers = rows[0]
        lines = [
            "Table row: " + "; ".join(
                f"{header}: {value}"
                for header, value in zip(headers, row)
            )
            for row in rows[1:]
        ]
        promoted.extend(lines)
        table.replace_with(NavigableString("\n".join(lines)))

    return promoted
