"""
Maison POS — Streamlit app
Nhận diện món ăn từ ảnh bằng model Keras (EfficientNetB0 fine-tuned, 224x224, 11 lớp)
và tính tiền hoá đơn tự động.

Chạy:
    streamlit run app.py
"""

import io
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────
# CẤU HÌNH CHUNG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maison POS",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "model_nhan_dang_mon_an.keras"
IMG_SIZE = (224, 224)

# ──────────────────────────────────────────────────────────────────────────
# ⚠️ QUAN TRỌNG — SỬA PHẦN NÀY CHO ĐÚNG VỚI DỮ LIỆU LÚC BẠN TRAIN MODEL
#
# CLASS_NAMES phải đúng THỨ TỰ với train_generator.class_indices lúc train
# (thường là thứ tự alphabet theo tên thư mục ảnh). Nếu sai thứ tự, model
# vẫn chạy nhưng tên món hiển thị sẽ SAI.
#
# Mỗi class tương ứng với 1 món trong MENU bên dưới (cùng index).
# ──────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "truffle_mushroom_risotto",
    "uc_vit_ap_chao",
    "ca_hoi_dai_tay_duong",
    "wagyu_beef_burger",
    "burrata_ca_chua",
    "muc_chien_gion",
    "sup_hanh_tay_phap",
    "charcuterie_board",
    "creme_brulee",
    "chocolate_fondant",
    "sup_lo_nuong",
]

# index khớp với CLASS_NAMES ở trên — sửa tên hiển thị / giá / mô tả tại đây
MENU = [
    {"name": "Truffle Mushroom Risotto", "category": "Món chính", "price": 28.50,
     "desc": "Arborio, nấm rừng, dầu truffle, parmesan"},
    {"name": "Ức vịt áp chảo", "category": "Món chính", "price": 38.00,
     "desc": "Sốt cherry, rau củ nướng"},
    {"name": "Cá hồi Đại Tây Dương", "category": "Món chính", "price": 34.00,
     "desc": "Beurre blanc chanh, măng tây, caper"},
    {"name": "Wagyu Beef Burger", "category": "Món chính", "price": 32.00,
     "desc": "Brioche, cheddar, truffle aioli, khoai tây chiên"},
    {"name": "Burrata & Cà chua", "category": "Khai vị", "price": 16.50,
     "desc": "Dầu olive, húng quế, bánh mì nướng"},
    {"name": "Mực chiên giòn", "category": "Khai vị", "price": 15.00,
     "desc": "Aioli chanh, ớt, salad rocket"},
    {"name": "Súp hành tây Pháp", "category": "Khai vị", "price": 14.00,
     "desc": "Gruyère crouton, nước dùng hành caramel"},
    {"name": "Charcuterie Board", "category": "Khai vị", "price": 22.00,
     "desc": "Prosciutto, manchego, cornichons, mật ong"},
    {"name": "Crème Brûlée", "category": "Tráng miệng", "price": 12.00,
     "desc": "Vani Madagascar, đường caramen giòn"},
    {"name": "Chocolate Fondant", "category": "Tráng miệng", "price": 13.50,
     "desc": "Sốt chocolate tan chảy, kem vani"},
    {"name": "Súp lơ nướng", "category": "Món chính", "price": 24.00,
     "desc": "Sốt romesco, hạnh nhân rang"},
]
assert len(CLASS_NAMES) == len(MENU), "CLASS_NAMES và MENU phải có cùng số lượng phần tử!"

ACCENT = "#f5a623"
BG = "#0d0d0d"
CARD = "#161616"
BORDER = "#2a2a2a"

# ──────────────────────────────────────────────────────────────────────────
# CSS — phối màu tối + cam giống bản gốc
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; color: #eaeaea; }}
section[data-testid="stSidebar"] {{ background-color: {CARD}; }}

h1, h2, h3 {{ color: #ffffff; }}

div[data-testid="stMetric"] {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 18px;
}}
div[data-testid="stMetricValue"] {{ color: #ffffff; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background-color: #1a1a1a;
    padding: 6px;
    border-radius: 10px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 8px;
    color: #b5b5b5;
    padding: 8px 18px;
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    background-color: {ACCENT} !important;
    color: #1a1a1a !important;
}}

.menu-card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}
.menu-row {{
    display: grid;
    grid-template-columns: 1.3fr 0.7fr 2fr 0.6fr;
    padding: 14px 18px;
    border-bottom: 1px solid {BORDER};
    align-items: center;
}}
.menu-row.head {{
    color: #8a8a8a;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER};
}}
.menu-name {{ color: #ffffff; font-weight: 600; }}
.menu-cat {{ color: #c9c9c9; }}
.menu-desc {{ color: #9a9a9a; font-size: 14px; }}
.menu-price {{ color: {ACCENT}; font-weight: 700; text-align: right; }}

.pill-btn {{
    display:inline-block; padding:6px 14px; border-radius:8px;
    background:#1a1a1a; border:1px solid {BORDER}; color:#cfcfcf;
    font-size:13px; margin-right:6px;
}}

.upload-box {{
    border: 2px dashed {BORDER};
    border-radius: 12px;
    padding: 50px 20px;
    text-align: center;
    background-color: #111111;
}}

.bill-item {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.conf-tag {{
    font-size: 12px;
    color: #8a8a8a;
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_dish(pil_image: Image.Image):
    model = load_model()
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)  # EfficientNetB0 trong model đã có sẵn lớp Rescaling/Normalization
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    return idx, confidence, preds


# ──────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────
if "current_bill" not in st.session_state:
    st.session_state.current_bill = []  # list of dict: idx, qty
if "transactions" not in st.session_state:
    st.session_state.transactions = []  # list of dict: timestamp, idx, qty, price


def add_to_bill(idx):
    for it in st.session_state.current_bill:
        if it["idx"] == idx:
            it["qty"] += 1
            return
    st.session_state.current_bill.append({"idx": idx, "qty": 1})


# ──────────────────────────────────────────────────────────────────────────
# HEADER / NAV
# ──────────────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([5, 1])
with top_l:
    st.markdown(f"### 🍽️ **Maison POS**")
with top_r:
    st.markdown(
        f"<div style='text-align:right; padding-top:10px;'>"
        f"<span style='color:#3ddc6f'>●</span> Đang phục vụ</div>",
        unsafe_allow_html=True,
    )

tab_menu, tab_bill, tab_stats = st.tabs(["📋 Menu", "🧾 Tính tiền", "📊 Thống kê"])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — MENU
# ──────────────────────────────────────────────────────────────────────────
with tab_menu:
    st.markdown("## Menu")
    st.caption("Danh sách món ăn & đồ uống")

    categories = ["Tất cả"] + sorted(set(m["category"] for m in MENU))
    c1, c2 = st.columns([3, 1.4])
    with c1:
        chosen_cat = st.radio("Danh mục", categories, horizontal=True, label_visibility="collapsed")
    with c2:
        search = st.text_input("Tìm món...", label_visibility="collapsed", placeholder="🔍 Tìm món...")

    rows = MENU
    if chosen_cat != "Tất cả":
        rows = [m for m in rows if m["category"] == chosen_cat]
    if search:
        rows = [m for m in rows if search.lower() in m["name"].lower()]

    html = ['<div class="menu-card">']
    html.append(
        '<div class="menu-row head"><div>Tên món</div><div>Danh mục</div><div>Mô tả</div><div style="text-align:right">Giá</div></div>'
    )
    for m in rows:
        html.append(
            f'<div class="menu-row">'
            f'<div class="menu-name">{m["name"]}</div>'
            f'<div class="menu-cat">{m["category"]}</div>'
            f'<div class="menu-desc">{m["desc"]}</div>'
            f'<div class="menu-price">${m["price"]:.2f}</div>'
            f'</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — TÍNH TIỀN
# ──────────────────────────────────────────────────────────────────────────
with tab_bill:
    st.markdown("## Tính tiền")
    st.caption("Tải ảnh món ăn để nhận diện và tính tổng")

    col_l, col_r = st.columns([1, 1.1])

    with col_l:
        files = st.file_uploader(
            "Tải ảnh món ăn lên",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Có thể chọn nhiều ảnh cùng lúc — mỗi ảnh là 1 món.",
        )
        if files:
            run = st.button("🔎 Nhận diện món", type="primary", use_container_width=True)
            if run:
                with st.spinner("Đang nhận diện..."):
                    for f in files:
                        img = Image.open(io.BytesIO(f.read()))
                        idx, conf, _ = predict_dish(img)
                        add_to_bill(idx)
                st.success(f"Đã nhận diện {len(files)} ảnh.")
        else:
            st.markdown(
                '<div class="upload-box">📷<br><br><b>Tải ảnh món ăn lên</b><br>'
                '<span style="color:#888">Kéo thả hoặc nhấn để chọn file</span></div>',
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown("**Các món đã nhận diện**")
        if not st.session_state.current_bill:
            st.markdown(
                '<div style="color:#888; padding:30px 0;">Chưa có ảnh. Vui lòng tải ảnh lên.</div>',
                unsafe_allow_html=True,
            )
        else:
            total = 0.0
            for i, item in enumerate(st.session_state.current_bill):
                m = MENU[item["idx"]]
                line_total = m["price"] * item["qty"]
                total += line_total
                bc1, bc2, bc3, bc4 = st.columns([3, 1.2, 1, 0.6])
                with bc1:
                    st.markdown(f"**{m['name']}**  \n<span class='conf-tag'>{m['category']}</span>",
                                unsafe_allow_html=True)
                with bc2:
                    new_qty = st.number_input(
                        "SL", min_value=1, value=item["qty"], step=1,
                        key=f"qty_{i}", label_visibility="collapsed",
                    )
                    st.session_state.current_bill[i]["qty"] = new_qty
                with bc3:
                    st.markdown(f"<div style='padding-top:8px;color:{ACCENT};font-weight:700'>${line_total:.2f}</div>",
                                unsafe_allow_html=True)
                with bc4:
                    if st.button("✕", key=f"rm_{i}"):
                        st.session_state.current_bill.pop(i)
                        st.rerun()

            st.divider()
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:22px;'>"
                f"<b>Tổng cộng</b><b style='color:{ACCENT}'>${total:,.2f}</b></div>",
                unsafe_allow_html=True,
            )

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("🗑️ Xoá hoá đơn", use_container_width=True):
                    st.session_state.current_bill = []
                    st.rerun()
            with bcol2:
                if st.button("✅ Lưu hoá đơn & thanh toán", type="primary", use_container_width=True):
                    now = dt.datetime.now()
                    for item in st.session_state.current_bill:
                        m = MENU[item["idx"]]
                        st.session_state.transactions.append({
                            "time": now,
                            "idx": item["idx"],
                            "name": m["name"],
                            "category": m["category"],
                            "qty": item["qty"],
                            "price": m["price"],
                            "total": m["price"] * item["qty"],
                        })
                    st.session_state.current_bill = []
                    st.success("Đã lưu hoá đơn vào thống kê!")
                    st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — THỐNG KÊ
# ──────────────────────────────────────────────────────────────────────────
with tab_stats:
    today = dt.date.today().strftime("%d/%m/%Y")
    st.markdown("## Thống kê hôm nay")
    st.caption(f"Phân tích lượt gọi món & doanh thu — {today}")

    tx = st.session_state.transactions

    if not tx:
        st.info("Chưa có dữ liệu. Hãy nhận diện và lưu vài hoá đơn ở tab **Tính tiền** trước.")
    else:
        df = pd.DataFrame(tx)
        total_orders = int(df["qty"].sum())
        revenue = df["total"].sum()
        avg_order = revenue / df["total"].count() if df["total"].count() else 0
        best_seller = df.groupby("name")["qty"].sum().idxmax()
        best_qty = df.groupby("name")["qty"].sum().max()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TỔNG LƯỢT GỌI", f"{total_orders}", "hôm nay")
        m2.metric("DOANH THU", f"${revenue:,.0f}", "hôm nay")
        m3.metric("TB / LƯỢT", f"${avg_order:,.2f}", "mỗi đơn")
        m4.metric("MÓN BÁN CHẠY", best_seller, f"{best_qty} lượt")

        st.markdown("### Hiệu suất từng món")
        plt.style.use("dark_background")
        perf = df.groupby("name")["qty"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        colors = [ACCENT if v == perf.max() else "#6b4f22" for v in perf.values]
        ax.bar(perf.index, perf.values, color=colors, width=0.55)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Theo danh mục")
            cat = df.groupby("category")["qty"].sum()
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            fig2.patch.set_facecolor(BG)
            pal = ["#f5a623", "#3ddc6f", "#4aa8ff", "#d36bd9"]
            ax2.pie(
                cat.values, labels=None, colors=pal[: len(cat)],
                wedgeprops=dict(width=0.45, edgecolor=BG),
            )
            st.pyplot(fig2, use_container_width=True)
            for name, val, col in zip(cat.index, cat.values, pal):
                st.markdown(
                    f"<span style='color:{col}'>●</span> {name} &nbsp; <b>{int(val)}</b>",
                    unsafe_allow_html=True,
                )

        with c2:
            st.markdown("### Khách theo giờ")
            df["hour"] = df["time"].apply(lambda t: t.hour)
            by_hour = df.groupby("hour")["qty"].sum()
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            fig3.patch.set_facecolor(BG)
            ax3.set_facecolor(BG)
            ax3.bar(by_hour.index.astype(str) + "h", by_hour.values, color="#3ddc6f", width=0.5)
            ax3.spines[["top", "right", "left"]].set_visible(False)
            ax3.tick_params(colors="#aaaaaa", labelsize=9)
            st.pyplot(fig3, use_container_width=True)

        st.markdown("### Bảng xếp hạng món")
        rank = (
            df.groupby("name")
            .agg(luot=("qty", "sum"), doanh_thu=("total", "sum"))
            .sort_values("luot", ascending=False)
            .reset_index()
        )
        max_luot = rank["luot"].max()
        for i, r in rank.iterrows():
            pct = r["luot"] / max_luot * 100
            st.markdown(
                f"<div style='margin-bottom:10px;'>"
                f"<div style='display:flex;justify-content:space-between;font-size:14px;'>"
                f"<span>{i+1}. {r['name']}</span>"
                f"<span style='color:#999'>{int(r['luot'])} lượt · ${r['doanh_thu']:,.0f}</span></div>"
                f"<div style='background:#222;border-radius:6px;height:8px;margin-top:4px;'>"
                f"<div style='background:{ACCENT};width:{pct}%;height:8px;border-radius:6px;'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
