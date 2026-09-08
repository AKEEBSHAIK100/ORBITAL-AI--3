import math
from typing import List, Tuple, Dict, Any

def generate_tiles(width: int, height: int, tile_size: int = 640, overlap: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """
    Generate overlapping tile coordinates (x1, y1, x2, y2) covering the full image.
    Ensures complete coverage without downscaling small structures.
    """
    if width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]

    stride = max(1, int(tile_size * (1.0 - overlap)))

    x_starts = []
    curr_x = 0
    while curr_x < width:
        x_starts.append(curr_x)
        if curr_x + tile_size >= width:
            break
        curr_x += stride
    # Ensure right edge is covered
    if x_starts[-1] + tile_size < width:
        x_starts.append(width - tile_size)

    y_starts = []
    curr_y = 0
    while curr_y < height:
        y_starts.append(curr_y)
        if curr_y + tile_size >= height:
            break
        curr_y += stride
    # Ensure bottom edge is covered
    if y_starts[-1] + tile_size < height:
        y_starts.append(height - tile_size)

    tiles = []
    for ys in y_starts:
        for xs in x_starts:
            x1 = max(0, min(xs, width - tile_size) if width >= tile_size else 0)
            y1 = max(0, min(ys, height - tile_size) if height >= tile_size else 0)
            x2 = min(x1 + tile_size, width)
            y2 = min(y1 + tile_size, height)
            tiles.append((x1, y1, x2, y2))

    # Remove any duplicates while preserving order
    unique_tiles = []
    seen = set()
    for t in tiles:
        if t not in seen:
            seen.add(t)
            unique_tiles.append(t)

    return unique_tiles

def translate_detection_to_global(
    tile_coords: Tuple[int, int, int, int],
    bbox: List[float],
    polygon: List[List[float]],
    img_width: int,
    img_height: int
) -> Dict[str, Any]:
    """
    Translate tile-local bounding box and polygon coordinates to full image coordinates,
    and compute normalized coordinates (percentages 0-100).
    """
    tx1, ty1, tx2, ty2 = tile_coords
    gx1 = bbox[0] + tx1
    gy1 = bbox[1] + ty1
    gx2 = bbox[2] + tx1
    gy2 = bbox[3] + ty1

    # Clamp to image boundaries
    gx1 = max(0.0, min(float(gx1), float(img_width)))
    gy1 = max(0.0, min(float(gy1), float(img_height)))
    gx2 = max(0.0, min(float(gx2), float(img_width)))
    gy2 = max(0.0, min(float(gy2), float(img_height)))

    global_polygon = []
    for pt in polygon:
        px = max(0.0, min(float(pt[0] + tx1), float(img_width)))
        py = max(0.0, min(float(pt[1] + ty1), float(img_height)))
        global_polygon.append([round(px, 1), round(py, 1)])

    # Normalized percentages for frontend SVG rendering
    norm_polygon = [
        [round((pt[0] / img_width) * 100.0, 3), round((pt[1] / img_height) * 100.0, 3)]
        for pt in global_polygon
    ]

    centroid = [
        round((gx1 + gx2) / 2.0, 1),
        round((gy1 + gy2) / 2.0, 1)
    ]
    centroid_pct = [
        round((centroid[0] / img_width) * 100.0, 3),
        round((centroid[1] / img_height) * 100.0, 3)
    ]

    return {
        "bbox": [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)],
        "bbox_pct": [
            round((gx1 / img_width) * 100.0, 3),
            round((gy1 / img_height) * 100.0, 3),
            round(((gx2 - gx1) / img_width) * 100.0, 3),
            round(((gy2 - gy1) / img_height) * 100.0, 3),
        ],
        "polygon": global_polygon,
        "polygon_pct": norm_polygon,
        "centroid": centroid,
        "centroid_pct": centroid_pct,
    }
