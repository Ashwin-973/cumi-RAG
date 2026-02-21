"""
CUMI Abrasive Product Scraper
------------------------------
Uses Playwright (async) to scrape product data from cumiabrasive.in
and exports results to a formatted Excel (.xlsx) file.

Requirements:
    pip install playwright pandas openpyxl
    playwright install chromium
"""

import asyncio
import json
import re
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_URL    = "https://cumiabrasive.in"
SHOP_URL    = f"{BASE_URL}/shop-abrasives/jsf/epro-loop-builder/pagenum"
OUTPUT_FILE = "cumi_products.xlsx"
HEADLESS    = False          # Set True to run silently in the background
NAV_TIMEOUT = 30_000         # ms – page navigation timeout
SEL_TIMEOUT = 8_000          # ms – element-wait timeout
# ──────────────────────────────────────────────────────────────────────────────


async def get_all_product_urls(page) -> list[str]:
    """
    Crawl every page of the shop grid and collect all product URLs.
    Handles pagination automatically.
    """
    urls: list[str] = []
    page_number=1
    current_url = f"{SHOP_URL}/{page_number}"

    while current_url:
        print(f"  [SHOP] Fetching grid: {current_url}")
        await page.goto(current_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)

        # Each product card exposes its link inside h3.elementor-heading-title a
        anchors = await page.query_selector_all(".product-card-loop-item h3.elementor-heading-title a")
        for a in anchors:
            href = await a.get_attribute("href")
            if href and href not in urls:
                urls.append(href.strip())

        # Follow "next page" pagination if present; stop when exhausted
        page_number=page_number+1
        current_url = f"{SHOP_URL}/{page_number}" if page_number<=14 else None

    print(f"  [SHOP] Found {len(urls)} product URLs total.")
    return urls


async def scrape_product(page, url: str) -> dict:
    """
    Visit a single product page and extract all required fields.
    Every field is wrapped in its own try/except so one missing element
    never aborts the entire record.
    """
    record = {
        "Product Name":    None,
        "Product URL":     url,
        "Brand Name":      None,
        "Brand URL":       None,
        "Main Category":   None,
        "Sub Category":    None,
        "Specifications":  None,
        "Catalogue PDF URL": None,
    }

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    except PWTimeout:
        print(f"    [WARN] Timeout navigating to {url}")
        return record

    # ── Product Name (page <title> or h1) ─────────────────────────────────────
    try:
        name_el = await page.query_selector("h1.product_title, h1.entry-title, h1")
        if name_el:
            record["Product Name"] = (await name_el.inner_text()).strip()
    except Exception:
        pass

    # ── Brand Name & Brand URL ─────────────────────────────────────────────────
    try:
        brand_el = await page.query_selector(".jet-listing-dynamic-terms__link")
        if brand_el:
            record["Brand Name"] = (await brand_el.inner_text()).strip()
            href = await brand_el.get_attribute("href")
            record["Brand URL"] = href.strip() if href else None
    except Exception:
        pass

    # ── Breadcrumb Categories ──────────────────────────────────────────────────
    try:
        crumb_items = await page.query_selector_all(".woocommerce-breadcrumb ul li")
        texts = [(await li.inner_text()).strip() for li in crumb_items]
        # Index 1 → Main Category; second-to-last → Sub Category
        if len(texts) >= 2:
            record["Main Category"] = texts[1]
        if len(texts) >= 3:
            record["Sub Category"] = texts[-2]   # second-to-last
    except Exception:
        pass

    # ── Specifications (key-value repeater) ───────────────────────────────────
    try:
        spec_items = await page.query_selector_all(
            ".spec-card-wrap .jet-listing-dynamic-repeater__item"
        )
        specs: dict[str, str] = {}
        for item in spec_items:
            try:
                key_el = await item.query_selector(".sped__card-title")
                val_el = await item.query_selector(".sped__card-details")
                key = (await key_el.inner_text()).strip() if key_el else ""
                val = (await val_el.inner_text()).strip() if val_el else ""
                if key:
                    specs[key] = val
            except Exception:
                continue
        record["Specifications"] = json.dumps(specs, ensure_ascii=False) if specs else None
    except Exception:
        pass

    # ── Catalogue PDF URL ──────────────────────────────────────────────────────
    try:
        pdf_el = await page.query_selector(".info-download-wrap a")
        if pdf_el:
            href = await pdf_el.get_attribute("href")
            if href:
                href = href.strip()
                # Resolve relative paths to absolute
                if href.startswith("/"):
                    href = BASE_URL + href
                record["Catalogue PDF URL"] = href
    except Exception:
        pass

    print(f"    [OK] {record['Product Name'] or '(no name)'} — {url}")
    return record


def apply_excel_formatting(filepath: str) -> None:
    """
    Post-process the raw xlsx with professional formatting:
    frozen header row, auto-column widths, styled header, and table borders.
    """
    wb = load_workbook(filepath)
    ws = wb.active
    ws.title = "Products"

    header_fill  = PatternFill("solid", fgColor="1F4E79")   # dark navy
    header_font  = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    cell_font    = Font(name="Arial", size=10)
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_align   = Alignment(horizontal="left",   vertical="top",    wrap_text=True)
    thin_border  = Border(
        left=Side(style="thin"),  right=Side(style="thin"),
        top=Side(style="thin"),   bottom=Side(style="thin"),
    )

    # Style header row
    for cell in ws[1]:
        cell.fill      = header_fill
        cell.font      = header_font
        cell.alignment = center_align
        cell.border    = thin_border

    # Freeze the header row so it stays visible while scrolling
    ws.freeze_panes = "A2"

    # Style data rows & compute column widths
    col_widths = {cell.column: len(str(cell.value or "")) for cell in ws[1]}
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font      = cell_font
            cell.alignment = left_align
            cell.border    = thin_border
            col_widths[cell.column] = max(
                col_widths.get(cell.column, 0),
                min(len(str(cell.value or "")), 60),   # cap width at 60
            )

    for col_idx, width in col_widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = max(width + 4, 15)

    # Give the Specifications column (index 7) a fixed wider width
    ws.column_dimensions[get_column_letter(7)].width = 50

    wb.save(filepath)
    print(f"  [FORMAT] Formatting applied → {filepath}")


async def main():
    products: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        )
        # Use a single shared page for the shop grid listing
        shop_page = await context.new_page()
        product_urls = await get_all_product_urls(shop_page)

        # Use a second dedicated page for product detail scraping
        detail_page = await context.new_page()
        for idx, url in enumerate(product_urls, start=1):
            print(f"\n[{idx}/{len(product_urls)}] Scraping: {url}")
            record = await scrape_product(detail_page, url)
            products.append(record)

        await browser.close()

    # ── Export to Excel ────────────────────────────────────────────────────────
    print(f"\n[EXPORT] Writing {len(products)} records to {OUTPUT_FILE} …")
    df = pd.DataFrame(products, columns=[
        "Product Name", "Product URL", "Brand Name", "Brand URL",
        "Main Category", "Sub Category", "Specifications", "Catalogue PDF URL",
    ])
    df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")

    # Apply professional formatting pass
    apply_excel_formatting(OUTPUT_FILE)
    print(f"[DONE] Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())