__all__ = [
    "count_images_in_accepted_dir",
    "save_stats_to_json",
    "async_download_urls",
    "async_fetch_and_download",
    "async_fetch_image_urls",
    "fetch_and_download",
    "fetch_image_urls",
    "make_session",
    "build_prompt",
    "fallback_distribution",
    "parse_count_analyse_output",
    "run_count_analyse",
    "scale_count_needs",
    "file_md5",
    "file_sha1",
    "hamming",
    "HashDeduplicator",
    "image_phash",
    "phash_dedup",
    "OCR",
    "MultiProcessOCRPool",
    "ocr_worker_process",
    "QualityAssessor",
    "generate_qwen_images",
    "SemanticChecker",
    "TextConsistencyChecker",
    "check_alamy",
]


def __getattr__(name: str):
    """Lazy-import on first access (PEP 562)."""
    _MODULE_MAP = {
        "count_images_in_accepted_dir": "src.tools.accepted_count",
        "save_stats_to_json": "src.tools.accepted_count",
        "async_download_urls": "src.tools.bing_noapi",
        "async_fetch_and_download": "src.tools.bing_noapi",
        "async_fetch_image_urls": "src.tools.bing_noapi",
        "fetch_and_download": "src.tools.bing_noapi",
        "fetch_image_urls": "src.tools.bing_noapi",
        "make_session": "src.tools.bing_noapi",
        "build_prompt": "src.tools.count_analyse",
        "fallback_distribution": "src.tools.count_analyse",
        "parse_count_analyse_output": "src.tools.count_analyse",
        "run_count_analyse": "src.tools.count_analyse",
        "scale_count_needs": "src.tools.count_analyse",
        "file_md5": "src.tools.hashing",
        "file_sha1": "src.tools.hashing",
        "hamming": "src.tools.hashing",
        "HashDeduplicator": "src.tools.hashing",
        "image_phash": "src.tools.hashing",
        "phash_dedup": "src.tools.hashing",
        "OCR": "src.tools.ocr",
        "MultiProcessOCRPool": "src.tools.ocr_multiprocess",
        "ocr_worker_process": "src.tools.ocr_multiprocess",
        "QualityAssessor": "src.tools.quality",
        "generate_qwen_images": "src.tools.qwen_image",
        "SemanticChecker": "src.tools.semantic_checker",
        "TextConsistencyChecker": "src.tools.text_consistency",
        "check_alamy": "src.tools.watermark_detector",
    }

    if name in _MODULE_MAP:
        import importlib

        mod = importlib.import_module(_MODULE_MAP[name])
        attr = getattr(mod, name)

        globals()[name] = attr
        return attr

    raise AttributeError(f"module 'src.tools' has no attribute {name!r}")
