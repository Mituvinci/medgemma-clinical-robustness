"""
JDCR Case Challenge PDF Scraper
================================
Downloads JDCR (JAAD Case Reports) Case Challenge PDFs from journal volume pages
using Selenium. Each volume page is scraped for the "JDCR Case Challenge" section
and the corresponding PDF is downloaded.

Output format: MM_YYYY_JDCR.pdf (e.g., 01_2022_JDCR.pdf)

Usage:
    # Download using default URL list (1_1_urls_jdcr_53.txt)
    python 2_scrape_jdcr_direct.py --output ./jdcr_cases

    # Download using a custom URL file
    python 2_scrape_jdcr_direct.py --urls my_urls.txt --output ./jdcr_cases

    # Run headless (no browser window)
    python 2_scrape_jdcr_direct.py --output ./jdcr_cases --headless

Requirements: see requirements_jdcr.txt
"""

import argparse
import os
import re
import time
import logging
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

SCRIPT_DIR = Path(__file__).parent


def load_urls(urls_path: str) -> list:
    """Load volume URLs from a text file (one URL per line)."""
    path = Path(urls_path)
    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {path}")
    urls = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    logger.info(f"Loaded {len(urls)} URLs from {path}")
    return urls


def setup_driver(output_dir: str, headless: bool = False):
    """Setup Chrome driver with download preferences."""
    chrome_options = Options()
    download_dir = os.path.abspath(output_dir)

    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "plugins.plugins_disabled": ["Chrome PDF Viewer"],
    }
    chrome_options.add_experimental_option("prefs", prefs)

    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def accept_cookies(driver):
    """Handle cookie consent popup if it appears."""
    try:
        time.sleep(2)
        for xpath in [
            "//button[contains(text(), 'Accept all cookies')]",
            "//button[contains(text(), 'Accept all')]",
            "//button[contains(text(), 'Accept')]",
        ]:
            try:
                button = driver.find_element(By.XPATH, xpath)
                if button.is_displayed():
                    button.click()
                    logger.info("Accepted cookies")
                    time.sleep(2)
                    return
            except Exception:
                continue
    except Exception:
        pass


def extract_month_year_from_page(driver):
    """Extract month and year from the current page headings."""
    try:
        headings = driver.find_elements(By.CSS_SELECTOR, "h1, h2, h3")
        for heading in headings:
            text = heading.text.strip()
            match = re.search(
                r"(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+(\d{4})",
                text, re.IGNORECASE,
            )
            if match:
                month_name = match.group(1).lower()
                year = match.group(2)
                month_num = MONTH_MAP.get(month_name, "00")
                return month_num, year, month_name.capitalize()
    except Exception:
        pass
    return None, None, None


def find_jdcr_case_challenge_pdf(driver):
    """Find the PDF link in the JDCR Case Challenge section."""
    try:
        headings = driver.find_elements(
            By.CSS_SELECTOR, "h2, h3, h4, h5, div.section-heading"
        )
        jdcr_section = None
        for heading in headings:
            if "JDCR Case Challenge" in heading.text:
                try:
                    jdcr_section = heading.find_element(
                        By.XPATH,
                        "./following-sibling::div | ./parent::*/following-sibling::div "
                        "| ./ancestor::section",
                    )
                    break
                except Exception:
                    try:
                        jdcr_section = heading.find_element(By.XPATH, "./parent::*")
                    except Exception:
                        pass

        if jdcr_section:
            for strategy in [
                './/a[contains(text(), "PDF")]',
                'a[href*=".pdf"]',
                './/a[contains(translate(@href, "PDF", "pdf"), "pdf")]',
            ]:
                try:
                    if strategy.startswith("."):
                        links = jdcr_section.find_elements(By.XPATH, strategy)
                    else:
                        links = jdcr_section.find_elements(By.CSS_SELECTOR, strategy)
                    if links:
                        return links[0]
                except Exception:
                    continue

        # Fallback: search entire page
        all_pdf_links = driver.find_elements(
            By.XPATH, '//a[contains(text(), "PDF") or contains(@href, ".pdf")]'
        )
        for link in all_pdf_links:
            try:
                parent = link.find_element(
                    By.XPATH,
                    "./ancestor::article | ./ancestor::section | ./ancestor::div[@class]",
                )
                parent_text = parent.text
                if "JDCR Case Challenge" in parent_text or "JDCR" in parent_text:
                    if not any(
                        x in parent_text
                        for x in ["Case Reports", "Case Series", "Images in Dermatology"]
                    ):
                        return link
            except Exception:
                continue

    except Exception as e:
        logger.error(f"Error finding PDF: {e}")

    return None


def wait_for_download(download_dir: str, timeout: int = 60):
    """Wait for a Chrome download to complete and return the filename."""
    for _ in range(timeout):
        time.sleep(1)
        files = os.listdir(download_dir)
        if not any(f.endswith(".crdownload") for f in files):
            pdf_files = [f for f in files if f.endswith(".pdf")]
            if pdf_files:
                pdf_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(download_dir, x)),
                    reverse=True,
                )
                return pdf_files[0]
    return None


def download_pdf_via_selenium(driver, pdf_link_element, target_filename, output_dir):
    """Download PDF by clicking the link element."""
    try:
        existing_pdfs = set(f for f in os.listdir(output_dir) if f.endswith(".pdf"))
        pdf_link_element.click()

        downloaded_file = wait_for_download(output_dir, timeout=30)
        if downloaded_file and downloaded_file not in existing_pdfs:
            old_path = os.path.join(output_dir, downloaded_file)
            new_path = os.path.join(output_dir, target_filename)
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(old_path, new_path)
            logger.info(f"  Saved: {target_filename}")
            return True
        else:
            logger.error("  Download did not complete or file not found")
            return False
    except Exception as e:
        logger.error(f"  Download error: {e}")
        return False


def process_volume_url(driver, url: str, index: int, total: int, output_dir: str):
    """Process a single volume URL: navigate, find PDF, download."""
    logger.info(f"[{index}/{total}] Processing: {url}")
    driver.get(url)
    time.sleep(3)
    accept_cookies(driver)
    time.sleep(2)

    month_num, year, month_name = extract_month_year_from_page(driver)
    if not month_num or not year:
        logger.error("  Could not extract month/year")
        return False

    logger.info(f"  {month_name} {year}")

    pdf_element = find_jdcr_case_challenge_pdf(driver)
    if not pdf_element:
        logger.error("  Could not find JDCR Case Challenge PDF")
        return False

    filename = f"{month_num}_{year}_JDCR.pdf"
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        logger.info(f"  SKIPPED (already exists): {filename}")
        return True

    return download_pdf_via_selenium(driver, pdf_element, filename, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download JDCR Case Challenge PDFs from JAAD Case Reports"
    )
    parser.add_argument(
        "--urls",
        default=str(SCRIPT_DIR / "1_1_urls_jdcr_53.txt"),
        help="Path to text file with volume URLs (one per line). "
             "Default: 1_1_urls_jdcr_53.txt",
    )
    parser.add_argument(
        "--output", "-o",
        default="jdcr_cases",
        help="Output directory for downloaded PDFs (default: jdcr_cases)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (no browser window)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    volume_urls = load_urls(args.urls)

    logger.info("=" * 60)
    logger.info("JDCR Case Challenge PDF Scraper")
    logger.info(f"URLs: {len(volume_urls)}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    driver = setup_driver(output_dir, headless=args.headless)

    try:
        success_count = 0
        failed_count = 0

        for i, url in enumerate(volume_urls, 1):
            try:
                if process_volume_url(driver, url, i, len(volume_urls), output_dir):
                    success_count += 1
                else:
                    failed_count += 1
                time.sleep(2)
            except Exception as e:
                logger.error(f"  ERROR: {e}")
                failed_count += 1

        logger.info("=" * 60)
        logger.info(f"COMPLETE! Downloaded: {success_count}, Failed: {failed_count}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

    finally:
        time.sleep(2)
        driver.quit()


if __name__ == "__main__":
    main()
