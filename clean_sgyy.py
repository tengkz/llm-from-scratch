"""
清洗 sgyy2.txt:
  1. GB18030 → UTF-8
  2. 移除广告行
  3. 移除文件头尾的垃圾内容
"""

import re
import os

# 1. 读取原始文件（GB18030 编码）
with open("sgyy2.txt", "rb") as f:
    raw = f.read()
text = raw.decode("gb18030")

print(f"原始文件: {len(text)} 字符, {len(text.splitlines())} 行")

# 2. 去除文件头部垃圾（在 "第一回" 之前的所有内容）
match = re.search(r"第一回\s", text)
if match:
    text = text[match.start():]
    print(f"去除头部垃圾，从「第一回」开始")

# 3. 去除尾部广告（最后一个章回之后的 URL 广告）
text = re.sub(
    r"\n本小说来自：.*$",
    "",
    text,
    flags=re.DOTALL,
)

# 4. 按行清洗：移除广告行
lines = text.splitlines()
clean_lines = []
ad_count = 0

for line in lines:
    stripped = line.strip()

    # 跳过空行
    if not stripped:
        clean_lines.append(line)
        continue

    # 跳过包含 URL 的广告行
    if re.search(r"(www\.|http:|七七手机网|欧度网|3GP|本节字数)", stripped):
        ad_count += 1
        continue

    # 跳过纯 ASCII art / 特殊字符构成的垃圾行
    # (大量 ★、(、)、~ 等符号构成的图案)
    special_ratio = sum(1 for c in stripped if ord(c) in range(0x20, 0x7F))
    if len(stripped) > 5 and special_ratio / len(stripped) > 0.8 and not re.search(r"[一-鿿]", stripped):
        ad_count += 1
        continue

    clean_lines.append(line)

text = "\n".join(clean_lines)
print(f"移除 {ad_count} 行广告")

# 5. 合并多余的空行（3个以上连续空行 → 2 个空行）
text = re.sub(r"\n{4,}", "\n\n\n", text)

# 6. 保存为 UTF-8
output = "sgyy_clean.txt"
with open(output, "w", encoding="utf-8") as f:
    f.write(text)

print(f"清洗完成: {output}")
print(f"最终: {len(text)} 字符, {len(text.splitlines())} 行")

# 统计各回
chapters = re.findall(r"第[一二三四五六七八九十百]+回\s", text)
print(f"检测到 {len(chapters)} 回:")
for ch in chapters[:5]:
    print(f"  {ch.strip()}")
print(f"  ...")
for ch in chapters[-3:]:
    print(f"  {ch.strip()}")
