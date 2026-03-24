import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import pandas as pd
import os
import io
import base64
from datetime import datetime

# --- 1. MARD 221 色标库 ---
MARD_221 = {
    "A1": (250, 245, 205), "A2": (252, 254, 214), "A3": (252, 255, 146), "A4": (247, 236, 92), "A5": (240, 216, 58), "A6": (253, 169, 81), "A7": (250, 140, 79), "A8": (251, 218, 77), "A9": (247, 157, 95), "A10": (244, 126, 56), "A11": (254, 219, 153), "A12": (253, 162, 118), "A13": (254, 198, 103), "A14": (247, 88, 66), "A15": (251, 246, 94), "A16": (254, 255, 151), "A17": (253, 225, 115), "A18": (252, 191, 128), "A19": (253, 126, 119), "A20": (249, 214, 102), "A21": (250, 227, 147), "A22": (237, 248, 120), "A23": (228, 200, 186), "A24": (243, 246, 169), "A25": (253, 247, 133), "A26": (255, 199, 52),
    "B1": (223, 241, 59), "B2": (100, 243, 67), "B3": (161, 245, 134), "B4": (95, 223, 52), "B5": (57, 225, 88), "B6": (100, 224, 164), "B7": (62, 174, 124), "B8": (29, 155, 84), "B9": (42, 80, 55), "B10": (154, 209, 186), "B11": (98, 112, 50), "B12": (26, 110, 61), "B13": (200, 232, 125), "B14": (171, 232, 79), "B15": (48, 83, 53), "B16": (192, 237, 156), "B17": (158, 179, 62), "B18": (230, 237, 79), "B19": (38, 183, 142), "B20": (203, 236, 207), "B21": (24, 97, 106), "B22": (10, 66, 65), "B23": (52, 59, 26), "B24": (232, 250, 166), "B25": (78, 132, 109), "B26": (144, 124, 53), "B27": (208, 224, 175), "B28": (158, 229, 187), "B29": (198, 223, 95), "B30": (227, 251, 177), "B31": (180, 230, 145), "B32": (146, 173, 96),
    "C1": (240, 254, 228), "C2": (171, 248, 254), "C3": (162, 224, 247), "C4": (68, 205, 251), "C5": (6, 170, 223), "C6": (84, 167, 233), "C7": (57, 119, 202), "C8": (15, 82, 189), "C9": (51, 73, 195), "C10": (60, 188, 227), "C11": (42, 222, 211), "C12": (30, 51, 78), "C13": (205, 231, 254), "C14": (213, 252, 247), "C15": (33, 197, 196), "C16": (24, 88, 162), "C17": (2, 209, 243), "C18": (33, 50, 68), "C19": (24, 134, 157), "C20": (26, 112, 169), "C21": (188, 221, 252), "C22": (107, 177, 187), "C23": (200, 226, 253), "C24": (126, 197, 249), "C25": (169, 232, 224), "C26": (66, 173, 207), "C27": (208, 222, 249), "C28": (189, 206, 232), "C29": (54, 74, 137),
    "D1": (172, 183, 239), "D2": (134, 141, 211), "D3": (53, 84, 175), "D4": (22, 45, 123), "D5": (179, 78, 198), "D6": (179, 123, 220), "D7": (135, 88, 169), "D8": (227, 210, 254), "D9": (213, 185, 244), "D10": (48, 26, 73), "D11": (190, 185, 226), "D12": (220, 153, 206), "D13": (181, 3, 141), "D14": (134, 41, 147), "D15": (47, 31, 140), "D16": (226, 228, 240), "D17": (199, 211, 249), "D18": (154, 100, 184), "D19": (216, 194, 217), "D20": (154, 53, 173), "D21": (148, 5, 149), "D22": (56, 56, 154), "D23": (234, 219, 248), "D24": (118, 138, 225), "D25": (73, 80, 194), "D26": (214, 198, 235),
    "E1": (246, 212, 203), "E2": (252, 193, 221), "E3": (246, 189, 232), "E4": (232, 100, 158), "E5": (240, 86, 159), "E6": (235, 65, 114), "E7": (197, 54, 116), "E8": (253, 219, 233), "E9": (227, 118, 199), "E10": (209, 59, 149), "E11": (247, 218, 212), "E12": (246, 147, 191), "E13": (181, 2, 106), "E14": (250, 212, 191), "E15": (245, 201, 202), "E16": (251, 244, 236), "E17": (247, 227, 236), "E18": (249, 200, 219), "E19": (246, 187, 209), "E20": (215, 198, 206), "E21": (192, 157, 164), "E22": (179, 140, 159), "E23": (147, 125, 138), "E24": (222, 190, 229),
    "F1": (254, 147, 129), "F2": (246, 61, 75), "F3": (238, 78, 62), "F4": (251, 42, 64), "F5": (225, 3, 40), "F6": (145, 54, 53), "F7": (145, 25, 50), "F8": (187, 1, 38), "F9": (224, 103, 122), "F10": (135, 70, 40), "F11": (89, 35, 35), "F12": (243, 83, 107), "F13": (244, 92, 69), "F14": (252, 173, 178), "F15": (213, 5, 39), "F16": (248, 192, 169), "F17": (232, 155, 125), "F18": (208, 127, 74), "F19": (190, 69, 74), "F20": (198, 148, 149), "F21": (242, 184, 198), "F22": (247, 195, 208), "F23": (237, 128, 108), "F24": (224, 157, 175), "F25": (232, 72, 84),
    "G1": (255, 228, 211), "G2": (252, 198, 172), "G3": (241, 196, 165), "G4": (220, 179, 135), "G5": (231, 179, 78), "G6": (227, 160, 20), "G7": (152, 92, 58), "G8": (113, 61, 47), "G9": (228, 182, 133), "G10": (218, 140, 66), "G11": (218, 200, 152), "G12": (254, 201, 147), "G13": (178, 113, 75), "G14": (139, 104, 76), "G15": (246, 248, 227), "G16": (242, 216, 193), "G17": (119, 84, 78), "G18": (255, 227, 213), "G19": (221, 125, 65), "G20": (165, 69, 47), "G21": (179, 133, 97),
    "H1": (255, 255, 255), "H2": (251, 251, 251), "H3": (180, 180, 180), "H4": (135, 135, 135), "H5": (70, 70, 72), "H6": (44, 44, 44), "H7": (1, 1, 1), "H8": (231, 214, 220), "H9": (239, 237, 238), "H10": (235, 235, 235), "H11": (205, 205, 205), "H12": (253, 246, 238), "H13": (244, 237, 241), "H14": (206, 215, 212), "H15": (154, 166, 166), "H16": (27, 18, 19), "H17": (240, 238, 239), "H18": (252, 255, 246), "H19": (242, 238, 229), "H20": (150, 160, 159), "H21": (248, 251, 230), "H22": (202, 202, 210), "H23": (155, 156, 148),
    "M1": (187, 198, 182), "M2": (144, 153, 148), "M3": (105, 126, 129), "M4": (224, 212, 188), "M5": (209, 204, 175), "M6": (176, 170, 134), "M7": (176, 167, 150), "M8": (174, 128, 130), "M9": (166, 136, 98), "M10": (196, 179, 187), "M11": (157, 118, 147), "M12": (100, 75, 81), "M13": (199, 146, 102), "M14": (194, 117, 99), "M15": (116, 125, 122)
}

# --- 2. 算法逻辑 ---
def get_visual_dist(c1, c2):
    return ((c1[0]-c2[0])*0.3)**2 + ((c1[1]-c2[1])*0.59)**2 + ((c1[2]-c2[2])*0.11)**2

@st.cache_data
def find_best_mard(rgb):
    min_dist, best = float('inf'), ("None", rgb)
    for mid, mrgb in MARD_221.items():
        d = get_visual_dist(rgb, mrgb)
        if d < min_dist:
            min_dist, best = d, (mid, mrgb)
    return best

def get_color_square(rgb):
    """为 Dataframe 生成色块图片的 Base64 数据"""
    img = Image.new("RGB", (40, 25), rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# --- 3. Streamlit 页面配置 ---
st.set_page_config(page_title="MARD 221 拼豆专家系统", layout="wide")
st.title("🎨 MARD 221 拼豆图纸专家系统")

# 🔴 页面状态池
if "modified_pixels" not in st.session_state:
    st.session_state.modified_pixels = {}

with st.sidebar:
    st.header("⚙️ 核心参数设置")
    uploaded_file = st.file_uploader("1. 上传图片", type=["jpg", "png", "jpeg"], help="支持常用JPG、PNG格式。上传后系统会自动开始进行拼豆化转换。")
    st.divider()
    
    # 🌟 将大段文字说明，写进了 help="弹出提示" 中 🌟
    st.subheader("2. 作品规格")
    preset = st.radio("快速选择规格：", ["自定义", "钥匙扣 (29px)", "摆件 (50px)", "挂画 (80px)", "精细模型 (120px)"], index=2, help="规格决定了最后拼豆板的物理大小。颗粒数越多，细节越丰富，但也越耗费精力和材料。")
    preset_map = {"钥匙扣 (29px)": 29, "摆件 (50px)": 50, "挂画 (80px)": 80, "精细模型 (120px)": 120}
    default_w = preset_map.get(preset, 50)
    target_w = st.slider("目标宽度 (像素/颗数)", 10, 200, default_w, help="横向一共由多少颗拼豆组成。高度会按图片原比例自动计算。")
    st.divider()
    
    st.subheader("3. 色彩处理")
    color_limit = st.slider("色彩精简度 (色数)", 2, 60, 25, help="将原图合并压缩成多少种主色调。现实色彩千千万，但实物拼豆颜色有限。数字越小画面越简洁、越容易采购豆子。")
    saturation = st.slider("色彩饱和度增强", 0.5, 2.0, 1.2, help="拼豆实物颜色通常比电脑屏幕鲜艳。建议稍微拉高（>1.0）使成品更生动好看。")
    st.divider()
    
    st.subheader("4. 视图辅助")
    show_grid = st.checkbox("显示辅助网格", value=True, help="在画布上每隔1颗和5颗渲染灰线。")
    show_labels = st.checkbox("显示色号标注", value=False, help="直接在预览图的豆子上写上对应的色号。小尺寸图纸建议开启。")

# --- 4. 图像处理与渲染 ---
if uploaded_file is not None:
    img_orig = Image.open(uploaded_file).convert('RGB')
    img = ImageEnhance.Color(img_orig).enhance(saturation)
    w, h = img.size
    target_h = int(target_w * (h / w))
    img_small = img.resize((target_w, target_h), resample=Image.LANCZOS)
    img_quant = img_small.quantize(colors=color_limit, method=Image.MAXCOVERAGE).convert('RGB')
    px = img_quant.load()
    inventory = {}
    label_map = {} 
    
    for y in range(target_h):
        for x in range(target_w):
            if (x, y) in st.session_state.modified_pixels:
                mid = st.session_state.modified_pixels[(x, y)]
                mrgb = MARD_221[mid]
            else:
                mid, mrgb = find_best_mard(px[x, y])
            px[x, y] = mrgb
            inventory[mid] = inventory.get(mid, 0) + 1
            label_map[(x, y)] = mid
            
    scale = 20 
    res = img_quant.resize((target_w * scale, target_h * scale), resample=Image.NEAREST)
    draw = ImageDraw.Draw(res)
    
    if show_grid or show_labels:
        try: font = ImageFont.load_default()
        except: font = None
        for y in range(target_h):
            for x in range(target_w):
                if show_labels:
                    mid = label_map[(x, y)]
                    bg_rgb = px[x, y]
                    brightness = sum(bg_rgb) / 3
                    text_color = (0,0,0) if brightness > 128 else (255,255,255)
                    draw.text((x * scale + 2, y * scale + 2), mid, fill=text_color, font=font)
        if show_grid:
            for x in range(0, res.width + 1, scale):
                width = 2 if (x//scale) % 5 == 0 else 1
                draw.line([(x, 0), (x, res.height)], fill=(120,120,120), width=width)
            for y in range(0, res.height + 1, scale):
                width = 2 if (y//scale) % 5 == 0 else 1
                draw.line([(0, y), (res.width, y)], fill=(120,120,120), width=width)

    # --- 整理清单数据 ---
    beads_data = []
    for mid in sorted(inventory, key=inventory.get, reverse=True):
        rgb = MARD_221[mid]
        beads_data.append({
            "预览": get_color_square(rgb),
            "色号": mid,
            "数量 (颗)": inventory[mid],
            "rgb": rgb
        })
    df = pd.DataFrame(beads_data)

    col1, col2 = st.columns([2.5, 1])
    with col1:
        st.subheader("🖼️ 拼豆图纸预览")
        st.image(res, use_container_width=True)
        
        st.divider()
        with st.expander("🛠️ 展开/收起 像素级色彩微调工具"):
            st.subheader("🎨 局部放大寻点微调器")
            col_slice1, col_slice2, col_slice3 = st.columns([1.2, 1.2, 1.2])
            with col_slice1: active_y = st.slider("🔍 定位行数 (Y 轴)", 0, target_h - 1, 0)
            with col_slice2: active_x = st.slider(f"🔍 定位列数 (X 轴)", 0, target_w - 1, 0)
            with col_slice3: view_range = st.slider("👀 局部寻位视野半径", 1, 15, 3, help="控制目标周围看多少颗豆子。默认看周围各3颗豆子。")

            start_x, end_x = max(0, active_x - view_range), min(target_w - 1, active_x + view_range)
            start_y, end_y = max(0, active_y - view_range), min(target_h - 1, active_y + view_range)
            w_box, h_box = end_x - start_x + 1, end_y - start_y + 1
            
            st.markdown(f"**👁️ 锁定坐标附近范围排布 ($X={active_x}, Y={active_y}$)：**")
            box_img = Image.new("RGB", (w_box * 30, h_box * 30), (255, 255, 255))
            box_draw = ImageDraw.Draw(box_img)
            for local_y, y in enumerate(range(start_y, end_y + 1)):
                for local_x, x in enumerate(range(start_x, end_x + 1)):
                    color = px[x, y]
                    is_center = (x == active_x and y == active_y)
                    border = (255, 0, 0) if is_center else (180, 180, 180)
                    box_draw.rectangle([local_x * 30, local_y * 30, (local_x + 1) * 30 - 1, (local_y + 1) * 30 - 1], fill=color, outline=border, width=3 if is_center else 1)
            st.image(box_img)
            
            st.markdown("---")
            color_scope = st.radio("选用色彩库作用域", ["当前画布已用色", "MARD 221 全库"], horizontal=True)
            target_mid_list = [item['色号'] for item in beads_data] if color_scope == "当前画布已用色" else list(MARD_221.keys())

            cols_per_row = 8 
            for i in range(0, len(target_mid_list), cols_per_row):
                batch = target_mid_list[i:i + cols_per_row]
                btn_cols = st.columns(cols_per_row)
                for j, mid in enumerate(batch):
                    rgb = MARD_221[mid]
                    color_hex = '#%02x%02x%02x' % rgb
                    with btn_cols[j]:
                        st.markdown(f'<div style="background-color:{color_hex}; width:24px; height:24px; border-radius:4px; border:1.5px solid #666; display:inline-block; vertical-align:middle;"></div> <span style="font-size:14px; font-weight:bold;">{mid}</span>', unsafe_allow_html=True)
                        if st.button(f"替换", key=f"btn_{mid}_{i}_{j}"):
                            st.session_state.modified_pixels[(active_x, active_y)] = mid
                            st.rerun()

            if st.session_state.modified_pixels:
                if st.button("🧹 一键重置所有手动修改"):
                    st.session_state.modified_pixels = {}
                    st.rerun()
        
        st.divider()
        
        # --- 🚀 内存级导出功能（支持手机直接下载到相册/系统） ---
        list_height = 80 + (len(beads_data) // 4 + 1) * 40
        final_export = Image.new("RGB", (res.width, res.height + list_height), (255, 255, 255))
        final_export.paste(res, (0, 0))
        draw_ex = ImageDraw.Draw(final_export)
        try: font_ex = ImageFont.load_default()
        except: font_ex = None
        
        y_start = res.height + 20
        draw_ex.text((20, y_start), "Material List / 拼豆清单:", fill=(0,0,0), font=font_ex)
        for i, item in enumerate(beads_data):
            row, col_idx = i // 4, i % 4
            x_pos, y_pos = 20 + col_idx * 200, y_start + 40 + row * 30
            draw_ex.rectangle([x_pos, y_pos, x_pos + 15, y_pos + 15], fill=item['rgb'], outline=(100,100,100))
            draw_ex.text((x_pos + 25, y_pos), f"{item['色号']}: {item['数量 (颗)']} pcs", fill=(0,0,0), font=font_ex)
        
        img_buffer = io.BytesIO()
        final_export.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        st.download_button(
            label="🚀 点击这里下载完整资料图（带清单）",
            data=img_bytes,
            file_name=f"Beads_Pattern_{datetime.now().strftime('%m%d_%H%M')}.png",
            mime="image/png"
        )
        
    with col2:
        st.subheader("🛒 拼豆材料清单")
        st.dataframe(
            df[["预览", "色号", "数量 (颗)"]], 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "预览": st.column_config.ImageColumn("预览", width="small"),
                "色号": "色号",
                "数量 (颗)": "数量"
            }
        )
        st.metric("总计消耗拼豆", f"{sum(inventory.values())} 颗")

else:
    st.info("👋 欢迎使用 MARD 221 拼豆专家系统！请先在左侧上传图片。")
st.divider()
st.caption("MARD 221 Expert System v2.6 | 网页/手机适配下载版")
