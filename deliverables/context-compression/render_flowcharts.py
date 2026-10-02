"""Render the inspected context-compression flow as PNG and editable SVG."""

from __future__ import annotations

from html import escape
from math import atan2, cos, sin, pi
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
FONT = "C:/Windows/Fonts/msyh.ttc"
SCALE = 2
INK = "#253248"
MUTED = "#5D6B80"
LINE = "#8793A4"
PALETTE = {
    "plain": ("#F4F6F9", "#CFD6DF"),
    "blue": ("#EAF2FC", "#BED1EA"),
    "amber": ("#FFF5DB", "#E6CF91"),
    "purple": ("#F1EDFA", "#D3C5E8"),
    "green": ("#EAF5EF", "#BBDACA"),
    "red": ("#FCEFEB", "#E6C3B9"),
}


class Diagram:
    def __init__(self, width: int, height: int, title: str):
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width * SCALE, height * SCALE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f"<title>{escape(title)}</title>",
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        self.text(60, 63, title, size=43, align="left")

    def text(self, x, y, value, *, size=28, color=INK, align="center", max_width=None):
        font = ImageFont.truetype(FONT, round(size * SCALE))
        box = font.getbbox(value)
        width = font.getlength(value) / SCALE
        if max_width is not None:
            assert width <= max_width, (value, width, max_width)
        left = x - width / 2 if align == "center" else x
        assert left >= 0 and left + width <= self.width, (value, left, width)
        top = y * SCALE - (box[3] - box[1]) / 2 - box[1]
        self.draw.text((left * SCALE, top), value, font=font, fill=color)
        anchor = "middle" if align == "center" else "start"
        self.svg.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="central" '
            f'font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="{size}" '
            f'fill="{color}">{escape(value)}</text>'
        )

    def rect(self, x, y, w, h, kind="plain", radius=18):
        fill, stroke = PALETTE[kind]
        self.draw.rounded_rectangle(
            (x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE),
            radius=radius * SCALE, fill=fill, outline=stroke, width=2 * SCALE,
        )
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    def node(self, x, y, w, h, title, details=(), kind="plain", size=29, detail_size=24):
        self.rect(x, y, w, h, kind)
        lines = [(title, size, INK), *((d, detail_size, MUTED) for d in details)]
        gap = 37
        for i, (value, font_size, color) in enumerate(lines):
            cy = y + h / 2 + (i - (len(lines) - 1) / 2) * gap
            self.text(x + w / 2, cy, value, size=font_size, color=color, max_width=w - 30)

    def decision(self, cx, cy, w, h, title, detail=None):
        pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
        fill, stroke = PALETTE["amber"]
        self.draw.polygon([(int(x * SCALE), int(y * SCALE)) for x, y in pts], fill=fill)
        self.draw.line([(int(x * SCALE), int(y * SCALE)) for x, y in [*pts, pts[0]]], fill=stroke, width=2 * SCALE)
        encoded = " ".join(f"{x},{y}" for x, y in pts)
        self.svg.append(f'<polygon points="{encoded}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        self.text(cx, cy - 14 if detail else cy, title, size=28, max_width=w * .73)
        if detail:
            self.text(cx, cy + 23, detail, size=22, color=MUTED, max_width=w * .73)

    def line(self, pts, *, arrow=True, color=LINE, width=2):
        scaled = [(round(x * SCALE), round(y * SCALE)) for x, y in pts]
        self.draw.line(scaled, fill=color, width=width * SCALE, joint="curve")
        encoded = " ".join(f"{x},{y}" for x, y in pts)
        self.svg.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
        if arrow:
            x, y = pts[-1]
            px, py = pts[-2]
            angle = atan2(y - py, x - px)
            head = [(x, y), (x - 12 * cos(angle - pi / 6), y - 12 * sin(angle - pi / 6)), (x - 12 * cos(angle + pi / 6), y - 12 * sin(angle + pi / 6))]
            self.draw.polygon([(round(a * SCALE), round(b * SCALE)) for a, b in head], fill=color)
            encoded = " ".join(f"{a},{b}" for a, b in head)
            self.svg.append(f'<polygon points="{encoded}" fill="{color}"/>')

    def label(self, x, y, value, size=23):
        font = ImageFont.truetype(FONT, size * SCALE)
        half = font.getlength(value) / SCALE / 2 + 7
        bounds = (x - half, y - size / 2 - 5, x + half, y + size / 2 + 5)
        self.draw.rectangle(tuple(round(v * SCALE) for v in bounds), fill="white")
        self.svg.append(
            f'<rect x="{bounds[0]}" y="{bounds[1]}" width="{half * 2}" height="{size + 10}" fill="white"/>'
        )
        self.text(x, y, value, size=size, color=MUTED)

    def save(self, stem):
        self.image.save(OUT / f"{stem}.png", optimize=True)
        (OUT / f"{stem}.svg").write_text("\n".join([*self.svg, "</svg>"]), encoding="utf-8")
        print(f"{stem}: {self.image.width} x {self.image.height} PNG + SVG")


def main_flow():
    d = Diagram(1200, 1640, "上下文压缩 · 自动压缩主流程")
    d.text(60, 122, "每次主模型调用前执行；沿中间向下看，右侧为跳过压缩的分支。", size=25, color=MUTED, align="left")

    for a, b in [(264, 305), (419, 460), (574, 600), (730, 775), (869, 950), (1080, 1135), (1265, 1330)]:
        d.line([(600, a), (600, b)])
    d.line([(860, 665), (1080, 665), (1080, 1380), (860, 1380)])
    d.line([(860, 822), (1080, 822)], arrow=False)
    d.line([(340, 1380), (310, 1380)])

    d.node(340, 180, 520, 84, "进入 ReAct 循环", kind="blue")
    d.node(340, 305, 520, 114, "① 裁剪过长工具结果", ["snip · 保留头尾"], "blue")
    d.node(340, 460, 520, 114, "② 清理较旧工具结果", ["microcompact · 保留调用 / 结果配对"], "blue")
    d.decision(600, 665, 520, 130, "达到压缩阈值？", "且自动压缩未熔断")
    d.node(340, 775, 520, 94, "PreCompact(auto)", ["Hook 可阻止本次自动压缩"], "plain")
    d.node(340, 950, 520, 130, "③ 历史折叠", ["复跑廉价阶段，仍超限才尝试 fold", "摘要生成过程见图 2 左侧"], "purple")
    d.node(340, 1135, 520, 130, "压缩后处理", ["有新摘要 → 保存 snapshot（若启用）", "按压缩事件触发 PostCompact"], "purple")
    d.node(340, 1330, 520, 100, "调用主模型", ["成功：更新 usage，继续或结束任务"], "green")
    d.node(60, 1330, 250, 100, "超限恢复", ["见图 2 右侧"], "red", size=28)

    d.label(655, 752, "是")
    d.label(962, 633, "否 / 已熔断")
    d.label(965, 792, "Hook 阻止")
    d.label(680, 909, "未阻止")
    d.label(213, 1296, "上下文超限报错")

    d.line([(60, 1480), (1140, 1480)], arrow=False, color="#DCE1E8")
    d.text(60, 1524, "恢复会话：读取最后一个有效 snapshot 及后续消息；旧记录仍保留在磁盘。", size=25, align="left", max_width=1080)
    d.text(60, 1576, "实现：react.py · compression.py · tokens.py · transcript.py", size=23, color=MUTED, align="left")
    d.save("context-compression-main")


def detail_flow():
    d = Diagram(1400, 1890, "上下文压缩 · 摘要生成与超限恢复")
    d.text(60, 122, "左侧解释历史如何压缩；右侧解释模型拒绝过长上下文后如何继续。", size=26, color=MUTED, align="left")
    d.line([(700, 166), (700, 1670)], arrow=False, color="#DCE1E8")
    d.text(60, 191, "A / 摘要生成", size=33, align="left")
    d.text(755, 191, "B / 超限恢复", size=33, align="left")

    # Summary generation: a local failure route and a separate success bypass.
    d.line([(360, 360), (360, 410)])
    d.line([(360, 538), (360, 570)])
    d.line([(360, 700), (360, 750)])
    d.line([(140, 635), (40, 635), (40, 1030), (80, 1030)])
    d.line([(360, 876), (360, 975)])
    d.line([(640, 813), (671, 813), (671, 1212), (640, 1212)])
    d.line([(360, 1085), (360, 1160)])
    d.line([(360, 1264), (360, 1340)])

    d.node(80, 240, 560, 120, "划分消息区域", ["system / pinned 保留；控制消息后置"], "purple", detail_size=24)
    d.node(80, 410, 560, 128, "按完整 round 切分", ["保留 recent，旧 prefix 待压缩", "历史不足 / prefix 为空 → 跳过折叠"], "purple")
    d.decision(360, 635, 440, 130, "LLM 摘要可用？")
    d.node(80, 750, 560, 126, "Track A · LLM 总结", ["单块直接总结；多块先摘要后合成", "无工具调用 · 受输入 / 输出预算限制"], "purple")
    d.node(80, 975, 560, 110, "Track B · 确定性摘要", ["按 role 拼接短片段，限制总长度"], "plain")
    d.node(80, 1160, 560, 104, "一条 USER 摘要", ["清理框架保留标签；旧摘要可再次折叠"], "purple")
    d.node(80, 1340, 560, 168, "上下文重组", ["保留区 → 摘要 → recent", "→ 最近文件附件 → 临时控制消息"], "green")
    d.label(414, 723, "是")
    d.label(86, 607, "否")
    d.label(360, 925, "异常 / 超时 / 空摘要 → 降级")
    d.label(614, 904, "成功")
    d.text(80, 1571, "切分时不拆散工具调用与对应结果。", size=24, color=MUTED, align="left", max_width=570)
    d.text(80, 1618, "只有真实 fold 才追加最近文件附件。", size=24, color=MUTED, align="left")

    # Recovery: up to five additional provider calls, progressively reducing input.
    d.line([(1050, 352), (1050, 405)])
    d.line([(1050, 531), (1050, 580)])
    d.line([(1050, 690), (1050, 770)])
    d.line([(1050, 864), (1050, 920)])
    d.line([(1050, 1050), (1050, 1095)])
    d.line([(1270, 985), (1350, 985), (1350, 1275), (1320, 1275)])
    d.line([(1050, 1330), (1050, 1400)])
    d.line([(1050, 1510), (1050, 1560)])
    d.line([(780, 1275), (735, 1275), (735, 817), (810, 817)])
    d.line([(780, 1455), (735, 1455), (735, 1275)], arrow=False)

    d.node(780, 240, 540, 112, "模型报错：上下文超限", ["LLMContextTooLongError"], "red")
    d.node(780, 405, 540, 126, "强制 aggressive 压缩", ["PreCompact 的 block 被忽略", "snip → microcompact → fold"], "red")
    d.node(780, 580, 540, 110, "压缩后处理", ["有摘要则存边界；按事件运行 hook"], "plain")
    d.node(810, 770, 480, 94, "重试主模型", ["最多 5 次追加调用"], "blue")
    d.decision(1050, 985, 440, 130, "调用成功？")
    d.node(860, 1095, 380, 80, "继续原任务", kind="green")
    d.node(780, 1220, 540, 110, "仍超限：丢弃最旧完整 rounds", ["按 token gap；未知则按比例"], "red", size=27)
    d.node(780, 1400, 540, 110, "无 round 可丢：裁剪大消息", ["非 preserved 内容：保留头尾 + 标记"], "red", size=27)
    d.node(780, 1560, 540, 100, "无法缩减 / 重试耗尽", ["向上传播超限异常"], "red")
    d.label(1105, 1071, "是")
    d.label(1304, 950, "仍超限")
    d.label(923, 1366, "无安全 round 可丢")
    d.label(912, 1535, "无法再缩减")
    d.label(824, 718, "删除 / 裁剪后回到重试")

    d.line([(60, 1710), (1340, 1710)], arrow=False, color="#DCE1E8")
    d.text(60, 1755, "手动入口 /compact → 强制三阶段 → 更新当前 history", size=28, align="left")
    d.text(60, 1804, "该入口本身不触发 Pre/PostCompact、最近文件回注或 compaction snapshot 提交。", size=25, color=MUTED, align="left", max_width=1280)
    d.text(60, 1851, "实现：compression.py · compression_summary.py · react.py（MAX_PTL_RETRIES = 5）", size=23, color=MUTED, align="left")
    d.save("context-compression-details")


if __name__ == "__main__":
    main_flow()
    detail_flow()
