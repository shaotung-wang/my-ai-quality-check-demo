"""共享核心库：高光定位、图像纠偏、ROI 提取。"""
from lib.roi import (
    manual_deskew_image,
    auto_search_rotate_angle,
    extract_roi_from_image,
    find_highlight_for_test_image,
    preview_highlight,
)

__all__ = [
    "manual_deskew_image",
    "auto_search_rotate_angle",
    "extract_roi_from_image",
    "find_highlight_for_test_image",
    "preview_highlight",
]