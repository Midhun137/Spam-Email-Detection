"""Dataset preparation and email parsing for spam detection.

Supports downloading the Apache SpamAssassin public corpus, parsing raw
RFC 822/2822 email messages with headers and MIME structures, extracting
textual, sender, and structural message characteristics, handling duplicates,
and generating offline synthetic fixtures.
"""

from __future__ import annotations

import argparse
import email
import email.header
import email.utils
import hashlib
import json
import logging
import os
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Apache SpamAssassin Public Corpus Archives
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
DEFAULT_ARCHIVES = {
    "ham": ["20030228_easy_ham.tar.bz2", "20030228_hard_ham.tar.bz2"],
    "spam": ["20030228_spam.tar.bz2", "20030228_spam_2.tar.bz2"],
}

URL_REGEX = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)


def decode_mime_header(header_value: Optional[str]) -> str:
    """Safely decode an RFC 2047 MIME header."""
    if not header_value:
        return ""
    try:
        decoded_slices = email.header.decode_header(header_value)
        pieces = []
        for content, encoding in decoded_slices:
            if isinstance(content, bytes):
                enc = encoding or "utf-8"
                try:
                    pieces.append(content.decode(enc, errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    pieces.append(content.decode("latin-1", errors="replace"))
            else:
                pieces.append(str(content))
        return " ".join("".join(pieces).split())
    except Exception:
        return str(header_value).strip()


def extract_sender_domain(from_header: Optional[str]) -> Tuple[str, str, int]:
    """Extract sender address, clean domain, and missing-sender flag."""
    if not from_header or not str(from_header).strip():
        return "", "<missing>", 1

    try:
        _, email_addr = email.utils.parseaddr(from_header)
        email_addr = email_addr.strip().lower()
        if "@" in email_addr:
            domain = email_addr.split("@")[-1].strip()
            # Clean domain of trailing/leading brackets or punctuation
            domain = re.sub(r"[^\w.-]", "", domain)
            if domain and "." in domain:
                return email_addr, domain, 0
        return email_addr or "<missing>", "<missing>", 1
    except Exception:
        return "", "<missing>", 1


def extract_message_body_and_structure(
    msg: email.message.Message,
) -> Tuple[str, int, int]:
    """Extract human-readable body text, has_html flag, and attachment count.

    Prefers plain text; falls back to visible text from HTML.
    Avoids double-counting multipart/alternative parts.
    Counts attachments while ignoring their contents.
    """
    body = ""
    has_html = 0
    attachment_count = 0

    if msg.is_multipart():
        plain_parts: List[str] = []
        html_parts: List[str] = []

        for part in msg.walk():
            ctype = str(part.get_content_type() or "").lower()
            cdisp = str(part.get_content_disposition() or "").lower()
            fname = part.get_filename()

            # Identify attachment
            if cdisp == "attachment" or (fname is not None and cdisp != "inline"):
                attachment_count += 1
                continue

            # Skip multipart containers
            if part.is_multipart():
                continue

            payload = part.get_payload(decode=True)
            if not payload:
                continue

            charset = part.get_content_charset() or "utf-8"
            try:
                decoded_text = payload.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                decoded_text = payload.decode("latin-1", errors="replace")

            if ctype == "text/plain":
                plain_parts.append(decoded_text)
            elif ctype == "text/html":
                has_html = 1
                html_parts.append(decoded_text)

        # Prioritize plain text over HTML to avoid duplicating content
        if plain_parts:
            body = "\n".join(plain_parts)
        elif html_parts:
            soup = BeautifulSoup("\n".join(html_parts), "html.parser")
            href_urls = [
                str(a.get("href")) for a in soup.find_all("a", href=True)
                if a.get("href") and isinstance(a.get("href"), str)
            ]
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            body = soup.get_text(separator=" ")
            if href_urls:
                body = body + " " + " ".join(href_urls)
    else:
        ctype = str(msg.get_content_type() or "").lower()
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                text = payload.decode("latin-1", errors="replace")

            if ctype == "text/html":
                has_html = 1
                soup = BeautifulSoup(text, "html.parser")
                href_urls = [
                    str(a.get("href")) for a in soup.find_all("a", href=True)
                    if a.get("href") and isinstance(a.get("href"), str)
                ]
                for s in soup(["script", "style"]):
                    s.decompose()
                body = soup.get_text(separator=" ")
                if href_urls:
                    body = body + " " + " ".join(href_urls)
            else:
                body = text

    # Normalize body whitespace
    body = re.sub(r"\s+", " ", body).strip()
    return body, has_html, attachment_count


def parse_raw_email(raw_bytes: bytes, label: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Parse raw RFC email bytes into structured feature dictionary.

    Spam-filter headers (e.g., X-Spam-Status) and folder names are explicitly
    excluded to prevent label leakage.
    """
    try:
        msg = email.message_from_bytes(raw_bytes)
    except Exception as e:
        logger.debug(f"Failed to parse email bytes: {e}")
        return None

    raw_subject = msg.get("Subject")
    subject = decode_mime_header(raw_subject)

    from_header = msg.get("From")
    sender_addr, sender_domain, is_sender_missing = extract_sender_domain(from_header)

    body, has_html, attachment_count = extract_message_body_and_structure(msg)

    # Compute message-level structural characteristics
    subject_len = len(subject)
    body_len = len(body)
    url_count = len(URL_REGEX.findall(body))
    digit_count = sum(c.isdigit() for c in body)
    digit_ratio = digit_count / max(body_len, 1)
    alpha_count = sum(c.isalpha() for c in body)
    upper_count = sum(c.isupper() for c in body)
    uppercase_ratio = upper_count / max(alpha_count, 1)
    exclamation_count = (subject + " " + body).count("!")

    # Check for empty content
    is_subject_missing = 1 if not subject.strip() else 0
    is_body_missing = 1 if not body.strip() else 0

    record = {
        "subject": subject,
        "body": body,
        "sender_address": sender_addr,
        "sender_domain": sender_domain,
        "is_sender_missing": is_sender_missing,
        "is_subject_missing": is_subject_missing,
        "is_body_missing": is_body_missing,
        "subject_len": subject_len,
        "body_len": body_len,
        "url_count": url_count,
        "attachment_count": attachment_count,
        "digit_ratio": round(digit_ratio, 4),
        "uppercase_ratio": round(uppercase_ratio, 4),
        "exclamation_count": exclamation_count,
        "has_html": has_html,
    }

    if label is not None:
        record["label"] = int(label)

    return record


def compute_content_hash(subject: str, body: str) -> str:
    """Exact content hash for deduplication."""
    norm = (subject.strip().lower() + "|||" + body.strip().lower()).encode("utf-8")
    return hashlib.sha256(norm).hexdigest()


def compute_near_duplicate_cluster(subject: str, body: str) -> str:
    """Compute near-duplicate cluster ID based on normalized tokens."""
    norm_subj = re.sub(r"\W+", " ", subject.lower()).strip()
    norm_body = re.sub(r"\W+", " ", body.lower()[:300]).strip()
    norm = f"{norm_subj}###{norm_body}".encode("utf-8")
    return hashlib.md5(norm).hexdigest()[:16]


def download_spamassassin_corpus(raw_dir: Path) -> None:
    """Download and extract SpamAssassin public corpus archives."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for category, archives in DEFAULT_ARCHIVES.items():
        category_dir = raw_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for archive_name in archives:
            archive_path = raw_dir / archive_name
            if not archive_path.exists():
                url = SPAMASSASSIN_BASE_URL + archive_name
                logger.info(f"Downloading {archive_name} from {url}...")
                try:
                    urllib.request.urlretrieve(url, archive_path)
                    logger.info(f"Downloaded {archive_name} ({archive_path.stat().st_size} bytes)")
                except Exception as e:
                    logger.error(f"Error downloading {archive_name}: {e}")
                    raise

            # Extract archive
            stem = archive_name.replace(".tar.bz2", "")
            target_extract = category_dir / stem
            if not target_extract.exists():
                logger.info(f"Extracting {archive_name} to {target_extract}...")
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(path=target_extract)
                logger.info(f"Extracted {archive_name}")


def parse_raw_corpus(raw_dir: Path) -> pd.DataFrame:
    """Walk raw directory, parse all emails, and return raw DataFrame."""
    records: List[Dict[str, Any]] = []
    parsing_failures = 0

    category_mapping = {
        "ham": 0,
        "spam": 1,
    }

    for category, label_val in category_mapping.items():
        cat_root = raw_dir / category
        if not cat_root.exists():
            continue

        for root, _, files in os.walk(cat_root):
            for fname in files:
                # Skip archive files or hidden OS files
                if fname.endswith((".bz2", ".gz", ".tar", ".zip", ".DS_Store")):
                    continue

                fpath = Path(root) / fname
                try:
                    with open(fpath, "rb") as f:
                        raw_bytes = f.read()
                    rec = parse_raw_email(raw_bytes, label=label_val)
                    if rec is not None:
                        records.append(rec)
                    else:
                        parsing_failures += 1
                except Exception:
                    parsing_failures += 1

    logger.info(f"Parsed {len(records)} emails ({parsing_failures} failures)")
    df = pd.DataFrame(records)
    return df


def clean_and_deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean records, resolve label conflicts, and deduplicate."""
    initial_count = len(df)
    if initial_count == 0:
        return df, {"initial_count": 0, "cleaned_count": 0}

    # Compute exact hashes
    df["content_hash"] = df.apply(
        lambda r: compute_content_hash(str(r["subject"]), str(r["body"])), axis=1
    )

    # Investigate conflicting labels
    label_counts_per_hash = df.groupby("content_hash")["label"].nunique()
    conflicting_hashes = set(label_counts_per_hash[label_counts_per_hash > 1].index)
    conflicting_count = len(conflicting_hashes)

    if conflicting_count > 0:
        logger.warning(
            f"Found {conflicting_count} content hashes with conflicting labels! Removing them to maintain dataset validity."
        )
        df = df[~df["content_hash"].isin(conflicting_hashes)].copy()

    # Drop exact duplicates, keeping first
    exact_duplicates_removed = len(df) - df["content_hash"].nunique()
    df = df.drop_duplicates(subset=["content_hash"], keep="first").copy()

    # Assign near-duplicate cluster IDs
    df["duplicate_group_id"] = df.apply(
        lambda r: compute_near_duplicate_cluster(str(r["subject"]), str(r["body"])), axis=1
    )

    cleaned_count = len(df)
    class_dist = df["label"].value_counts().to_dict()
    ham_count = class_dist.get(0, 0)
    spam_count = class_dist.get(1, 0)

    stats = {
        "initial_count": initial_count,
        "cleaned_count": cleaned_count,
        "conflicting_duplicates_removed": conflicting_count,
        "exact_duplicates_removed": exact_duplicates_removed,
        "ham_count": int(ham_count),
        "spam_count": int(spam_count),
        "spam_percentage": round(100.0 * spam_count / max(cleaned_count, 1), 2),
        "missing_subject_count": int(df["is_subject_missing"].sum()),
        "missing_body_count": int(df["is_body_missing"].sum()),
        "missing_sender_count": int(df["is_sender_missing"].sum()),
        "num_near_duplicate_groups": int(df["duplicate_group_id"].nunique()),
    }

    return df, stats


def create_synthetic_fixture(output_dir: Path) -> Path:
    """Create synthetic fixture dataset for offline testing and smoke tests.

    Produces both raw RFC 822 emails and a tabular CSV dataset.
    """
    raw_fixture_dir = output_dir / "synthetic_raw"
    raw_fixture_dir.mkdir(parents=True, exist_ok=True)
    ham_dir = raw_fixture_dir / "ham"
    spam_dir = raw_fixture_dir / "spam"
    ham_dir.mkdir(parents=True, exist_ok=True)
    spam_dir.mkdir(parents=True, exist_ok=True)

    synthetic_samples = [
        # Legitimate (Ham)
        (
            0,
            "ham_01.eml",
            "Meeting minutes from quarterly review",
            "alice@engineering.corp",
            "Hi team, please find the summary of our quarterly review attached. Let me know if you have feedback.",
            0,
            1,
        ),
        (
            0,
            "ham_02.eml",
            "Project status update and deployment roadmap",
            "david@techcorp.io",
            "We have completed phase 1 testing. Deployment to staging is scheduled for tomorrow at 10 AM.",
            0,
            0,
        ),
        (
            0,
            "ham_03.eml",
            "Lunch tomorrow?",
            "sarah@friends.net",
            "Hey! Are you free for lunch tomorrow around 12:30 at the deli downtown?",
            0,
            0,
        ),
        (
            0,
            "ham_04.eml",
            "Invoice #4829 from Cloud Services",
            "billing@cloudservices.com",
            "Your monthly cloud services invoice is now ready for review. Total amount due is $142.50.",
            0,
            1,
        ),
        (
            0,
            "ham_05.eml",
            "Bug report #1029: NullPointerException in parser",
            "dev-list@opensource.org",
            "The parser throws NPE when encountering truncated bytes at line 42. Patch attached.",
            0,
            1,
        ),
        (
            0,
            "ham_06.eml",
            "Invitation: Research Seminar on Machine Learning",
            "events@university.edu",
            "Join us this Friday for a talk on deep learning generalization by Prof. Johnson.",
            0,
            0,
        ),
        (
            0,
            "ham_07.eml",
            "Flight booking confirmation #AB9281",
            "reservations@airline.com",
            "Thank you for booking with us. Your flight departs at 08:15 AM on Oct 12.",
            0,
            0,
        ),
        (
            0,
            "ham_08.eml",
            "Code review request: PR #340",
            "alex@github-corp.com",
            "Could you please review PR 340 adding the new authentication middleware?",
            0,
            0,
        ),
        (
            0,
            "ham_09.eml",
            "",  # Missing subject edge case
            "coworker@company.org",
            "Just sent you the spreadsheet via drive.",
            0,
            0,
        ),
        (
            0,
            "ham_10.eml",
            "Quick question about documentation",
            "",  # Missing sender edge case
            "Where can I find the API documentation for the export endpoint?",
            0,
            0,
        ),
        # Spam
        (
            1,
            "spam_01.eml",
            "URGENT: Claim your $1,000,000 lottery prize NOW!!!",
            "winner@prize-claim-free.biz",
            "CONGRATULATIONS! You have been selected as our grand prize winner! Click http://claim-prize-now.biz/win to collect cash instantly!!! Call 1-800-555-CASH now!",
            1,
            0,
        ),
        (
            1,
            "spam_02.eml",
            "Refinance your mortgage with 0% interest guaranteed!",
            "lowrates@loans-direct-fast.info",
            "LOWEST RATES EVER! Save thousands each month. Fast approval, no credit check required! Apply at http://instant-mortgage-quote.info today!",
            1,
            0,
        ),
        (
            1,
            "spam_03.eml",
            "Exclusive Luxury Watches - 90% DISCOUNT TODAY ONLY",
            "deals@discount-store-vip.xyz",
            "Rolex, Omega, and Breitling replicas starting from $49! Free worldwide delivery! Buy now at http://replica-luxury-watches.xyz!",
            1,
            0,
        ),
        (
            1,
            "spam_04.eml",
            "INCREASE YOUR WEB TRAFFIC BY 500% IN 24 HOURS",
            "marketing@seo-boosters-unlimited.net",
            "Guaranteed first page Google rankings! Best SEO packages starting at just $29! Visit www.rank-one-fast.net today!",
            1,
            0,
        ),
        (
            1,
            "spam_05.eml",
            "Security Alert: Your bank account has been suspended",
            "security@bank-verify-alert.com",
            "URGENT: Unauthorized access detected. You must verify your credentials immediately at http://secure-banking-verification.com/login or account will be closed.",
            1,
            0,
        ),
        (
            1,
            "spam_06.eml",
            "Lose 30 pounds in 30 days with secret miracle pill!",
            "diet@miracle-weightloss-cure.org",
            "Doctors are stunned! Burn fat while you sleep without exercise or diet! Order your risk-free trial at www.miracle-pill-now.org!",
            1,
            0,
        ),
        (
            1,
            "spam_07.eml",
            "Act Now! Pre-approved credit card with $25,000 limit",
            "offers@card-rewards-direct.cc",
            "You are pre-approved! No annual fee, 0% APR for 18 months! Click here to accept your offer immediately!",
            1,
            0,
        ),
        (
            1,
            "spam_08.eml",
            "ATTENTION: Work from home and earn $5,000/week!!!",
            "careers@fast-cash-income.net",
            "No experience needed! Work only 2 hours a day from your computer! Limited spots available! Sign up today at www.work-home-cash.net!",
            1,
            0,
        ),
        (
            1,
            "spam_09.eml",
            "Get rich trading crypto - 1000% ROI guaranteed",
            "",  # Missing sender edge case
            "Automated bitcoin robot makes you wealthy while you sleep! Sign up now at http://crypto-robot-rich.biz/start!",
            1,
            0,
        ),
        (
            1,
            "spam_10.eml",
            "",  # Missing subject edge case
            "cheapmeds@online-pharmacy-rx.net",
            "Buy cheap prescription drugs online without doctor consultation! Fast discreet overnight shipping!",
            1,
            0,
        ),
    ]

    records = []
    for label, fname, subject, sender, body, has_html_flag, att_count in synthetic_samples:
        target_dir = spam_dir if label == 1 else ham_dir
        email_path = target_dir / fname

        # Construct raw RFC email string
        raw_lines = [
            f"From: {sender}" if sender else "From: ",
            f"Subject: {subject}" if subject else "Subject: ",
            "MIME-Version: 1.0",
        ]
        if has_html_flag:
            raw_lines.append("Content-Type: text/html; charset=utf-8")
            raw_lines.append("")
            raw_lines.append(f"<html><body><p>{body}</p></body></html>")
        else:
            raw_lines.append("Content-Type: text/plain; charset=utf-8")
            raw_lines.append("")
            raw_lines.append(body)

        raw_content = "\r\n".join(raw_lines).encode("utf-8")
        with open(email_path, "wb") as f:
            f.write(raw_content)

        parsed = parse_raw_email(raw_content, label=label)
        if parsed:
            records.append(parsed)

    fixture_df = pd.DataFrame(records)
    fixture_df["content_hash"] = fixture_df.apply(
        lambda r: compute_content_hash(str(r["subject"]), str(r["body"])), axis=1
    )
    fixture_df["duplicate_group_id"] = fixture_df.apply(
        lambda r: compute_near_duplicate_cluster(str(r["subject"]), str(r["body"])), axis=1
    )

    csv_path = output_dir / "synthetic_emails.csv"
    for c in ["subject", "body", "sender_address", "sender_domain"]:
        if c in fixture_df.columns:
            fixture_df[c] = fixture_df[c].astype(str).str.replace("\x00", "", regex=False)
    fixture_df.to_csv(csv_path, index=False, escapechar="\\")
    logger.info(f"Created synthetic fixture with {len(fixture_df)} emails at {csv_path}")
    return csv_path


def prepare_dataset(
    raw_dir: Path,
    output_path: Path,
    auto_download: bool = False,
    create_fixtures: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Complete dataset preparation pipeline."""
    if auto_download:
        download_spamassassin_corpus(raw_dir)

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        if create_fixtures:
            logger.warning(
                f"Raw directory '{raw_dir}' is empty or missing. Falling back to synthetic fixture..."
            )
            fixtures_dir = raw_dir.parent / "fixtures"
            fixture_csv = create_synthetic_fixture(fixtures_dir)
            df = pd.read_csv(fixture_csv)
            df, stats = clean_and_deduplicate(df)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            return df, stats
        else:
            raise FileNotFoundError(
                f"Raw directory '{raw_dir}' is empty. Pass --download to fetch the SpamAssassin corpus."
            )

    logger.info(f"Parsing raw email corpus in {raw_dir}...")
    raw_df = parse_raw_corpus(raw_dir)

    logger.info("Deduplicating and analyzing corpus integrity...")
    cleaned_df, stats = clean_and_deduplicate(raw_df)

    # Clean null characters and formatting quirks from string columns
    str_cols = ["subject", "body", "sender_address", "sender_domain"]
    for c in str_cols:
        if c in cleaned_df.columns:
            cleaned_df[c] = cleaned_df[c].astype(str).str.replace("\x00", "", regex=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False, escapechar="\\")
    logger.info(f"Saved {len(cleaned_df)} cleaned records to {output_path}")

    # Save stats summary
    stats_file = output_path.parent / "dataset_summary.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    if create_fixtures:
        fixtures_dir = raw_dir.parent / "fixtures"
        create_synthetic_fixture(fixtures_dir)

    return cleaned_df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SpamAssassin Email Dataset")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw email directory containing ham/ and spam/ subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/emails_cleaned.csv",
        help="Path to output cleaned CSV dataset",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Apache SpamAssassin public corpus archives",
    )
    parser.add_argument(
        "--create-fixture",
        action="store_true",
        help="Generate synthetic offline fixture dataset in data/fixtures",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    out_path = Path(args.output)

    if args.create_fixture:
        fixtures_dir = raw_path.parent / "fixtures"
        create_synthetic_fixture(fixtures_dir)
        return

    df, stats = prepare_dataset(
        raw_dir=raw_path,
        output_path=out_path,
        auto_download=args.download,
        create_fixtures=True,
    )

    print("\n" + "=" * 50)
    print("DATASET PREPARATION REPORT")
    print("=" * 50)
    print(f"Original record count        : {stats['initial_count']}")
    print(f"Cleaned record count         : {stats['cleaned_count']}")
    print(f"Ham (Class 0) count          : {stats['ham_count']}")
    print(f"Spam (Class 1) count         : {stats['spam_count']}")
    print(f"Spam percentage              : {stats['spam_percentage']}%")
    print(f"Exact duplicates removed     : {stats['exact_duplicates_removed']}")
    print(f"Conflicting labels removed   : {stats['conflicting_duplicates_removed']}")
    print(f"Near-duplicate groups        : {stats['num_near_duplicate_groups']}")
    print(f"Missing subject count        : {stats['missing_subject_count']}")
    print(f"Missing body count           : {stats['missing_body_count']}")
    print(f"Missing sender domain count  : {stats['missing_sender_count']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
