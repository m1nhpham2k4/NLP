import requests
from bs4 import BeautifulSoup
import csv

base_url = "https://iuh.edu.vn/vi/thong-bao.html/p={}"

data = []
stt = 1

for page in range(1,31):

    url = base_url.format(page)
    print("Page:", page)

    r = requests.get(url, verify=False)

    r.encoding = r.apparent_encoding

    soup = BeautifulSoup(r.text,"html.parser")

    titles = soup.select("h3 a")

    for t in titles:
        title = t.get_text(strip=True)

        data.append([stt, title, "Thong_Tin_Truong"])
        stt += 1


with open("title_tin_tuc_truong.csv", "a", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    # chỉ ghi header nếu file trống
    if f.tell() == 0:
        writer.writerow(["STT","Content","Label"])

    writer.writerows(data)

print("Done:",stt-1)