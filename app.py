import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import plotly.graph_objects as go

# 1. Cấu hình giao diện chuẩn Dashboard rộng rãi
st.set_page_config(
    page_title="Food AI Insights - Nhận Diện Ẩm Thực",
    page_icon="🍲",
    layout="wide"
)

# 2. Định nghĩa CSS trực tiếp trong code để không bị lỗi trên Streamlit Cloud
st.markdown("""
    <style>
    /* Tổng thể ứng dụng */
    .main {
        background-color: #fdfdfd;
    }

    /* Thẻ Card hiển thị món ăn chính dạng Premium */
    .premium-card {
        background: linear-gradient(135deg, #0f9b0f 0%, #006400 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,100,0,0.12);
        margin-bottom: 25px;
        text-align: center;
    }

    .premium-card h2 {
        color: white !important;
        margin: 0 !important;
        font-size: 38px !important;
        font-weight: 800;
    }

    .premium-card p {
        margin: 5px 0 0 0 !important;
        opacity: 0.9;
        font-size: 16px;
        letter-spacing: 1px;
    }

    /* Khung hiển thị thông tin dinh dưỡng */
    .macro-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        margin-top: 10px;
    }

    .macro-title {
        font-weight: bold;
        color: #1b5e20;
        font-size: 18px;
        border-bottom: 2px solid #a5d6a7;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_html=True)

# 3. Cấu hình Dữ liệu: Thứ tự nhãn và Thông tin dinh dưỡng mở rộng
# (Bạn cần sắp xếp danh sách CLASS_NAMES chính xác theo thứ tự khi train model)
CLASS_NAMES = ["Bánh mì", "Bánh xèo", "Bún chả", "Bún bò Huế", "Phở bò"]

FOOD_DETAILS = {
    "Bánh mì": {"calo": "380 kcal", "carb": "50g", "protein": "12g", "fat": "15g", "vitamin": "B1, B6, Sắt", "desc": "Món ăn đường phố biểu tượng. Gồm pate, chả giò, thịt nguội hòa quyện cùng rau dưa chua ngọt."},
    "Bánh xèo": {"calo": "420 kcal", "carb": "45g", "protein": "14g", "fat": "22g", "vitamin": "C, Chất xơ, Kẽm", "desc": "Vỏ bánh giòn rụm từ bột gạo nghệ, nhân tôm thịt giá đỗ đượm vị. Ăn kèm nước mắm tỏi ớt."},
    "Bún chả": {"calo": "540 kcal", "carb": "65g", "protein": "24g", "fat": "18g", "vitamin": "B3, B12, Sắt", "desc": "Thịt ba chỉ và chả viên nướng than hoa đậm đà, thả trong bát nước chấm ấm nóng đặc trưng Hà Nội."},
    "Bún bò Huế": {"calo": "580 kcal", "carb": "60g", "protein": "28g", "fat": "20g", "vitamin": "B6, Sắt, Canxi", "desc": "Nước dùng nồng nàn hương mắm ruốc và sả tươi, kết hợp sợi bún to cùng miếng giò heo, thịt bò chín mượt."},
    "Phở bò": {"calo": "450 kcal", "carb": "55g", "protein": "26g", "fat": "12g", "vitamin": "B12, Kẽm, Kali", "desc": "Tinh túy ẩm thực Việt Nam. Nước dùng thanh trong ninh từ xương ống kèm thảo mộc quế hồi và bánh phở mềm dẻo."}
}

# 4. Tải model bằng cơ chế lưu bộ nhớ đệm (Cache) tối ưu hiệu năng
@st.cache_resource
def load_food_deep_model():
    return tf.keras.models.load_model('model_nhan_dang_mon_an.keras')

try:
    model = load_food_deep_model()
except Exception as e:
    st.error(f"❌ Hệ thống chưa tìm thấy file 'model_nhan_dang_mon_an.keras'. Chi tiết: {e}")

# --- GIAO DIỆN CHÍNH ---
st.title("🍲 Hệ Thống Định Danh & Phân Tích Ẩm Thực AI")
st.caption("Phiên bản v3.0 - Tích hợp phân tích vi chất & cấu trúc dinh dưỡng đa lượng (Macronutrients)")
st.markdown("---")

# --- BẢNG ĐIỀU KHIỂN SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Cấu Hình Mô Hình")
    confidence_threshold = st.slider("Ngưỡng xác thực tối thiểu (%)", min_value=10, max_value=95, value=40, step=5)
    
    st.markdown("---")
    st.markdown("### 🖼️ Ảnh Khảo Sát Nhanh (Test Samples)")
    st.write("Nếu chưa chuẩn bị sẵn file, bạn có thể chọn ảnh mẫu online để thử nghiệm ngay lập tức:")
    sample_choice = st.selectbox("Chọn ảnh mẫu:", ["Không chọn", "Mẫu Bánh Mì", "Mẫu Phở Bò"])
    
    st.markdown("---")
    st.markdown("### 📊 Trạng Thái Hệ Thống")
    st.success("Core AI: 🟢 Hoạt động tốt")
    st.info("Kích thước ảnh: 224x224x3")

# Xử lý nguồn ảnh kiểm thử online nếu người dùng chọn ảnh mẫu
img_source = None
if sample_choice == "Mẫu Bánh Mì":
    img_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Banh_mi_vietnamien.JPG/640px-Banh_mi_vietnamien.JPG"
elif sample_choice == "Mẫu Phở Bò":
    img_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Pho_Bo_Hanoi.JPG/640px-Pho_Bo_Hanoi.JPG"

# Chia bố cục làm 2 cột chính cân xứng
col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.markdown("### 📸 Tải Ảnh Đầu Vào")
    uploaded_file = st.file_uploader("Kéo thả hoặc nhấp chọn tệp hình ảnh món ăn (jpg, png, jpeg)...", type=["jpg", "png", "jpeg"])
    
    active_image = None
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
    elif img_source is not None:
        import requests
        from io import BytesIO
        try:
            res = requests.get(img_source)
            active_image = Image.open(BytesIO(res.content))
        except:
            st.error("Không thể tải ảnh mẫu từ internet. Vui lòng tải file thủ công.")

    if active_image is not None:
        st.image(active_image, caption="Hình ảnh thực tế đưa vào xử lý", use_container_width=True)

with col_right:
    st.markdown("### 📊 Kết Quả Trực Quan Từ AI")
    
    if active_image is not None:
        t_start = time.time()
        
        with st.spinner("🧠 Đang biến đổi ma trận ảnh & thực hiện dự đoán..."):
            img_ready = active_image.resize((224, 224))
            img_arr = tf.keras.preprocessing.image.img_to_array(img_ready)
            img_arr = np.expand_dims(img_arr, axis=0)
            
            raw_preds = model.predict(img_arr)
            probs = tf.nn.softmax(raw_preds[0]).numpy() * 100
            
            top_idx = np.argmax(probs)
            top_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else "Không xác định"
            top_prob = probs[top_idx]
            
            t_inference = time.time() - t_start

        if top_prob >= confidence_threshold:
            st.markdown(f"""
                <div class='premium-card'>
                    <p>MÓN ĂN ĐƯỢC DỰ ĐOÁN CHÍNH XÁC NHẤT</p>
                    <h2>{top_label}</h2>
                    <p>ĐỘ TIN CẬY: {top_prob:.1f}%</p>
                </div>
            """, unsafe_html=True)
            
            if top_label in FOOD_DETAILS:
                food_info = FOOD_DETAILS[top_label]
                st.markdown("<div class='macro-box'>", unsafe_html=True)
                st.markdown("<div class='macro-title'>🥗 Thành Phần Cấu Trúc Dinh Dưỡng</div>", unsafe_html=True)
                st.write(f"**Mô tả:** {food_info['desc']}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🔥 Năng lượng", food_info['calo'])
                m2.metric("🍞 Tinh bột", food_info['carb'])
                m3.metric("🥩 Chất đạm", food_info['protein'])
                m4.metric("🥑 Chất béo", food_info['fat'])
                
                st.caption(f"💡 **Khoáng chất & Vitamin nổi bật:** {food_info['vitamin']}")
                st.markdown("</div>", unsafe_html=True)
        else:
            st.warning(f"⚠️ Điểm tin cậy tối đa ({top_prob:.1f}%) thấp hơn ngưỡng thiết lập ({confidence_threshold}%). Vui lòng đổi góc chụp hoặc cung cấp ảnh rõ nét hơn.")

        st.write("<br>**📈 Phân bổ phần trăm xác suất giữa các lớp:**", unsafe_html=True)
        
        sort_order = np.argsort(probs)
        lbls_sorted = [CLASS_NAMES[i] for i in sort_order]
        prbs_sorted = [probs[i] for i in sort_order]

        fig = go.Figure(go.Bar(
            x=prbs_sorted,
            y=lbls_sorted,
            orientation='h',
            marker=dict(
                color=prbs_sorted,
                colorscale=[[0, '#e8f5e9'], [0.5, '#4caf50'], [1, '#1b5e20']], 
                line=dict(color='#1b5e20', width=1.2)
            ),
            text=[f" {p:.1f}%" for p in prbs_sorted],
            textposition='outside'
        ))

        fig.update_layout(
            margin=dict(l=10, r=40, t=10, b=10),
            height=250,
            xaxis=dict(title='Tỷ lệ xác suất (%)', range=[0, 120], showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.caption(f"⚡ Tốc độ tính toán của mạng nơ-ron: {t_inference:.4f} giây")
    else:
        st.info("👋 Chào mừng bạn đến với hệ thống! Hãy tải tệp ảnh lên ở vùng bên trái hoặc kích hoạt nhanh bằng các 'Ảnh mẫu' để xem phân tích đồ thị tự động.")
