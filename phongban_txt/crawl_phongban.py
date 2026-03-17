import requests
from bs4 import BeautifulSoup
import os
import re

# Danh sách link khoa (viết tay)
khoa_links = [
    "https://iuh.edu.vn/vi/phong-to-chuc-hanh-chinh.html",
    "https://iuh.edu.vn/vi/phong-tai-chinh-ke-toan.html",
    "https://iuh.edu.vn/vi/phong-ke-hoach-dau-tu.html",
    "https://iuh.edu.vn/vi/phong-dao-tao.html",
    "https://iuh.edu.vn/vi/phong-quan-ly-khoa-hoc-va-hop-tac-quoc-te.html",
    "https://iuh.edu.vn/vi/phong-cong-tac-chinh-tri-va-ho-tro-sinh-vien.html",
    "https://iuh.edu.vn/vi/phong-khao-thi-va-dam-bao-chat-luong.html",
    "https://iuh.edu.vn/vi/phong-quan-tri.html",
    "https://iuh.edu.vn/vi/phong-dich-vu.html",
    "https://iuh.edu.vn/vi/phong-quan-ly-ky-tuc-xa.html",
    "https://iuh.edu.vn/vi/tap-chi-khoa-hoc-va-cong-nghe-iuh.html",
    "https://iuh.edu.vn/vi/nha-xuat-ban-dai-hoc-cong-nghiep-tp-ho-chi-minh.html"
]

# Tạo thư mục lưu file
if not os.path.exists("phongban_txt"):
    os.makedirs("phongban_txt")

# Duyệt từng khoa
for link in khoa_links:
    print("Đang xử lý:", link)

    try:
        r = requests.get(link)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser')

        content = soup.find('div', {'class': 'iuhArticleContent'})
        if not content:
            print("Không tìm thấy nội dung")
            continue

        # Lấy text sạch
        text = content.get_text(separator="\n", strip=True)

        # Lấy tên khoa từ URL
        ten_khoa = link.split("/")[-1].replace(".html", "")
        ten_khoa = re.sub(r'[\\/*?:"<>|]', "_", ten_khoa)

        file_path = os.path.join("phongban_txt", ten_khoa + ".txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print("→ Đã lưu:", ten_khoa + ".txt")

    except Exception as e:
        print("Lỗi:", e)

print("Hoàn thành!")