import requests
from bs4 import BeautifulSoup
import os
import re

# Danh sách link khoa (viết tay)
khoa_links = [
    "https://iuh.edu.vn/vi/trung-tam-quan-tri-he-thong.html",
    "https://iuh.edu.vn/vi/trung-tam-thong-tin-truyen-thong.html",
    "https://iuh.edu.vn/vi/trung-tam-ngoai-ngu.html",
    "https://iuh.edu.vn/vi/trung-tam-giao-duc-quoc-phong-va-the-chat.html",
    "https://iuh.edu.vn/vi/trung-tam-thu-vien.html",
    "https://iuh.edu.vn/vi/trung-tam-nghien-cuu-va-phat-trien-cong-nghe-may-cong-nghiep.html"

]

# Tạo thư mục lưu file
if not os.path.exists("trungtam_txt"):
    os.makedirs("trungtam_txt")

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

        file_path = os.path.join("trungtam_txt", ten_khoa + ".txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print("→ Đã lưu:", ten_khoa + ".txt")

    except Exception as e:
        print("Lỗi:", e)

print("Hoàn thành!")