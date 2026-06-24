import cv2
import numpy as np


def order_points(pts):
    """
    Order 4 points as: TL, TR, BR, BL
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   
    rect[2] = pts[np.argmax(s)]   
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]  
    return rect


def is_near_edge(x, y, w, h, img_w, img_h, margin=0.25):

    edge_margin_w = img_w * margin
    edge_margin_h = img_h * margin

    near_left   = x < edge_margin_w
    near_right  = (x + w) > (img_w - edge_margin_w)
    near_top    = y < edge_margin_h
    near_bottom = (y + h) > (img_h - edge_margin_h)

    return (near_left or near_right) and (near_top or near_bottom)


def find_paper(thresh, debug_image=None):

    # 1. PADDING — prevent markers touching image border from being lost
    padding = 20
    padded_thresh = cv2.copyMakeBorder(
        thresh, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=0
    )

    if debug_image is not None:
        padded_debug = cv2.copyMakeBorder(
            debug_image, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

    # 2. Find contours
    # RETR_LIST (not RETR_EXTERNAL) — finds ALL contours, including ones
    # nested inside larger shapes (e.g., marker square inside hand shadow)
    contours, _ = cv2.findContours(
        padded_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = padded_thresh.shape[:2]
    mid_x = w_img // 2
    mid_y = h_img // 2

    # Quadrant-based marker storage (lists of candidates)
    quadrants = {"TL": [], "TR": [], "BL": [], "BR": []}

    candidate_count = 0

    # Image area for relative size filtering
    img_area = h_img * w_img

    for c in contours:
        area = cv2.contourArea(c)

        # --- AREA FILTER ---
        # Min: 50px (catch small markers in high-res photos)
        # Max: 3.5% of image (generous for close-up photos)
        if area < 50 or area > (img_area * 0.035):
            continue

        # --- SHAPE FILTER (polygon approximation) ---
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if not (3 <= len(approx) <= 8):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0 or w == 0:
            continue
        aspect_ratio = w / float(h)

        # Aspect ratio: markers are roughly square (0.4 – 2.5 for perspective distortion)
        if not (0.4 <= aspect_ratio <= 2.5):
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area

        if solidity < 0.7:
            continue  # Reject non-solid shapes (shadows, hand outlines)

        # --- EDGE PROXIMITY FILTER ---
        if not is_near_edge(x, y, w, h, w_img, h_img, margin=0.28):
            continue  # Must be in a corner region

        # --- DETERMINE QUADRANT ---
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        zone = ""
        if cx < mid_x and cy < mid_y:
            zone = "TL"
        elif cx >= mid_x and cy < mid_y:
            zone = "TR"
        elif cx < mid_x and cy >= mid_y:
            zone = "BL"
        elif cx >= mid_x and cy >= mid_y:
            zone = "BR"

        candidate_count += 1
        print(f"  Candidate {candidate_count}: Zone={zone}, Area={area:.0f}, "
              f"AR={aspect_ratio:.2f}, Solidity={solidity:.2f}, Pos=({x},{y})")

        quadrants[zone].append({"approx": approx, "area": area, "cx": cx, "cy": cy, "bbox": (x, y, w, h)})

        if debug_image is not None:
            cv2.drawContours(padded_debug, [approx], -1, (0, 255, 255), 2)

    # 3. VERIFY & SELECT BEST COMBINATION — need all 4 zones
    final_markers = []
    missing_zones = []

    for zone in ["TL", "TR", "BR", "BL"]:
        if len(quadrants[zone]) == 0:
            missing_zones.append(zone)
        else:
            # Sort by area descending and keep top 5 candidates
            quadrants[zone] = sorted(quadrants[zone], key=lambda c: c["area"], reverse=True)[:5]

    if len(missing_zones) == 0:
        valid_combos = []

        # Find all valid geometric combinations with similar areas
        for tl in quadrants["TL"]:
            for tr in quadrants["TR"]:
                for bl in quadrants["BL"]:
                    for br in quadrants["BR"]:
                        # 1. Geometric validity check (TL left of TR, TL above BL, etc)
                        if tl["cx"] >= tr["cx"] or bl["cx"] >= br["cx"]: continue
                        if tl["cy"] >= bl["cy"] or tr["cy"] >= br["cy"]: continue

                        # 2. Area consistency check (markers should be similar in size)
                        areas = [tl["area"], tr["area"], bl["area"], br["area"]]
                        area_ratio = max(areas) / min(areas)

                        if area_ratio <= 1.5:
                            valid_combos.append({
                                "combo": {"TL": tl, "TR": tr, "BL": bl, "BR": br},
                                "avg_area": sum(areas)/4
                            })

        if len(valid_combos) > 0:

            best_match = max(valid_combos, key=lambda x: x["avg_area"])
            
            for zone in ["TL", "TR", "BR", "BL"]:
                cand = best_match["combo"][zone]
                final_markers.append(cand["approx"])
                
                if debug_image is not None:
                    cv2.drawContours(padded_debug, [cand["approx"]], -1, (0, 255, 0), 4)
                    x, y, w, h = cand["bbox"]
                    cv2.putText(padded_debug, zone, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(padded_debug, f"{cand['area']:.0f}px", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            missing_zones = ["Valid Marker Combo (Area/Geometry Mismatch)"]

    if len(missing_zones) > 0:
        print(f"\n GAGAL: Marker tidak ditemukan di zona: {missing_zones}")
        print(f"   Total kandidat: {candidate_count}")
        print(f"   Saran:")
        print(f"   - Cek apakah marker memang ada di keempat sudut gambar")
        print(f"   - Pastikan marker cukup kontras (hitam pada putih)")
        print(f"   - Mungkin ada shadow/noise yang menutupi marker")

        if debug_image is not None:
            debug_image[:] = padded_debug[padding:-padding, padding:-padding]
        return None

    print(f"\n SUKSES: 4 Marker Zona Ditemukan!")

    # 4. Update debug image
    if debug_image is not None:
        debug_image[:] = padded_debug[padding:-padding, padding:-padding]

    # 5. COMPUTE PERSPECTIVE WARP
    centers = []
    for m in final_markers:
        M = cv2.moments(m)
        centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    rect_pts = np.array(centers, dtype="float32")
    rect_pts -= padding  # Remove padding offset

    rect = order_points(rect_pts)

    # Output: canonical A4-ish canvas
    dst = np.array([
        [0, 0], [1000, 0], [1000, 1414], [0, 1414]
    ], dtype="float32")

    M_warp = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(thresh, M_warp, (1000, 1414))

    return (warped, M_warp)