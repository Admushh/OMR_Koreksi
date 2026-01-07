import cv2
import numpy as np

def order_points(pts):
    """
    Mengurutkan titik agar selalu: TL, TR, BR, BL
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-Left (Jumlah x+y terkecil)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    
    # Bottom-Right (Jumlah x+y terbesar)
    rect[2] = pts[np.argmax(s)]
    
    # Top-Right (Selisih y-x terkecil) & Bottom-Left (Selisih y-x terbesar)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def is_near_edge(x, y, w, h, img_w, img_h, margin=0.15):
    """
    Cek apakah titik (x,y) dekat dengan tepi gambar.
    margin: persentase jarak dari tepi (0.15 = 15% dari lebar/tinggi)
    """
    edge_margin_w = img_w * margin
    edge_margin_h = img_h * margin
    
    # Cek apakah kotak dekat tepi kiri, kanan, atas, atau bawah
    near_left = x < edge_margin_w
    near_right = (x + w) > (img_w - edge_margin_w)
    near_top = y < edge_margin_h
    near_bottom = (y + h) > (img_h - edge_margin_h)
    
    # Marker harus di sudut, jadi harus dekat dengan 2 sisi sekaligus
    return (near_left or near_right) and (near_top or near_bottom)

def find_paper(thresh, debug_image=None):
    # 1. PADDING (Tetap dipakai biar aman dari border)
    padding = 20
    padded_thresh = cv2.copyMakeBorder(thresh, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    
    if debug_image is not None:
        padded_debug = cv2.copyMakeBorder(debug_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # 2. CARI KONTUR - GUNAKAN RETR_EXTERNAL untuk hindari nested contours
    contours, _ = cv2.findContours(padded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = padded_thresh.shape[:2]
    
    # Titik tengah gambar untuk membagi kuadran
    mid_x = w_img // 2
    mid_y = h_img // 2
    
    # Variabel penampung marker terbaik per kuadran
    quadrants = {
        "TL": None,  # Top-Left
        "TR": None,  # Top-Right
        "BL": None,  # Bottom-Left
        "BR": None   # Bottom-Right
    }

    print(f"\n--- SCANNING ZONES (Image: {w_img}x{h_img}) ---")
    print(f"Total contours found: {len(contours)}")

    candidate_count = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # Filter Luas - DIPERLONGGAR:
        # Min: Hindari noise kecil
        # Max: Naikkan ke 0.35 untuk toleransi marker lebih besar
        if area < 150 or area > (h_img * w_img * 0.35): 
            continue
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Harus Kotak (4-6 sudut toleransi)
        if 4 <= len(approx) <= 6:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Filter Rasio Kotak - DIPERLONGGAR: 0.6-1.5
            if 0.6 <= aspect_ratio <= 1.5:
                # CEK PROXIMITY KE TEPI - INI YANG PENTING!
                if not is_near_edge(x, y, w, h, w_img, h_img):
                    continue  # Skip jika tidak di sudut
                
                # Hitung Titik Pusat Kontur
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Tentukan kuadran
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
                print(f"  Candidate {candidate_count}: Zone={zone}, Area={area:.0f}, Aspect={aspect_ratio:.2f}, Pos=({x},{y})")
                
                # Cek apakah ini kandidat terbaik (terbesar) di zonanya
                if quadrants[zone] is None or area > quadrants[zone][1]:
                    quadrants[zone] = [approx, area]
                    
                    # Visualisasi kandidat sementara (Kuning Tipis)
                    if debug_image is not None:
                        cv2.drawContours(padded_debug, [approx], -1, (0, 255, 255), 2)

    # 3. VERIFIKASI HASIL
    final_markers = []
    missing_zones = []
    
    for zone in ["TL", "TR", "BR", "BL"]:
        if quadrants[zone] is not None:
            final_markers.append(quadrants[zone][0])
            # Visualisasi Final (Hijau Tebal) + Label Zona
            if debug_image is not None:
                approx = quadrants[zone][0]
                cv2.drawContours(padded_debug, [approx], -1, (0, 255, 0), 4)
                
                # Ambil koordinat untuk label text
                x, y, w, h = cv2.boundingRect(approx)
                cv2.putText(padded_debug, zone, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Tambahkan info area
                area_text = f"{quadrants[zone][1]:.0f}px"
                cv2.putText(padded_debug, area_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            missing_zones.append(zone)

    if len(missing_zones) > 0:
        print(f"\n❌ GAGAL: Marker tidak ditemukan di zona: {missing_zones}")
        print(f"   Total kandidat ditemukan: {candidate_count}")
        print(f"   Saran:")
        print(f"   - Cek apakah marker memang ada di keempat sudut gambar")
        print(f"   - Pastikan marker cukup kontras (hitam pada putih)")
        print(f"   - Coba sesuaikan threshold preprocessing")
        
        # Kembalikan gambar asli tanpa padding jika debug tersedia
        if debug_image is not None:
            debug_image[:] = padded_debug[padding:-padding, padding:-padding]
        return thresh

    print(f"\n✅ SUKSES: 4 Marker Zona Ditemukan!")

    # 4. KEMBALIKAN GAMBAR DEBUG
    if debug_image is not None:
        debug_image[:] = padded_debug[padding:-padding, padding:-padding]

    # 5. PROSES WARPING
    centers = []
    for m in final_markers:
        M = cv2.moments(m)
        centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    rect_pts = np.array(centers, dtype="float32")
    rect_pts -= padding  # Hapus efek padding
    
    rect = order_points(rect_pts)

    # Output A4 Size
    dst = np.array([
        [0, 0], [1000, 0], [1000, 1414], [0, 1414]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(thresh, M, (1000, 1414))

    return warped