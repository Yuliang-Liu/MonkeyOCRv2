"""BBox mapping and crop clamping (normalized 0–1000 space and pixel space)."""

from __future__ import annotations

from PIL import Image

from core_runner import _build_page_tasks, _map_bbox_to_image


def test_full_frame_maps_to_image_size():
    assert _map_bbox_to_image([0, 0, 1000, 1000], 100, 200) == [0, 0, 100, 200]


def test_out_of_range_coordinates_are_clamped():
    assert _map_bbox_to_image([-50, -50, 2000, 2000], 100, 200) == [0, 0, 100, 200]


def test_negative_coordinates_clamp_to_origin_with_nonzero_area():
    # x2/y2 are also negative after scaling; the mapper still emits a 1px box.
    assert _map_bbox_to_image([-10, 10, 20, -20], 50, 50) == [0, 0, 1, 1]


def test_zero_area_box_is_expanded_to_one_pixel():
    assert _map_bbox_to_image([500, 500, 500, 500], 100, 100) == [50, 50, 51, 51]


def test_swapped_corners_are_ordered():
    assert _map_bbox_to_image([800, 200, 100, 900], 100, 100) == [10, 20, 80, 90]


def test_build_page_tasks_clamps_pixel_bboxes():
    image = Image.new("RGB", (10, 10), color=(0, 0, 0))
    tasks = _build_page_tasks(
        0,
        image,
        [
            {"bbox": [-10, -10, 100, 100], "label": "Text"},
            {"bbox": [5, 5, 5, 5], "label": "Title"},
        ],
    )
    assert tasks[0]["bbox"] == [0, 0, 10, 10]
    assert tasks[1]["bbox"] == [5, 5, 6, 6]
    assert tasks[0]["image"].size == (10, 10)
    assert tasks[1]["image"].size == (1, 1)
