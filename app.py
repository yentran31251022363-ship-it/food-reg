import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import plotly.graph_objects as go

# 1. Cấu hình giao diện chuẩn Dashboard cao cấp
st.set_page_config(
    page_title="Food AI Insights - Nhận Diện Ẩm Thực",
    page_icon="🍲",
    layout="wide"
)

# 2. Tùy biến giao diện (CSS Custom Theme) phối màu Emerald & Slate cực sang
st.markdown("""
    <style>
    /* Chỉnh toàn bộ phông chữ và nền */
    .reportview-container { background: #fdfdfd; }
    
    /* Thiết kế thẻ Card cho kết quả chính */
    .premium-card {
        background: linear-gradient(135deg, #0f9b0f 0%, #006400 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,100,0,0.15);
        margin-bottom: 25px;
        text-align: center;
    }
    .premium-card h2 { color: white !important; margin: 0; font-size: 36px; font-weight: 800; }
    .premium-card p { margin: 5px 0 0 0; opacity: 0.9; font-size: 18px; }
    
    /* Khung Dinh Dưỡng */
    .macro-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .macro-title {
        font-weight: bold;
        color: #1b5e20;
        font-size: 18px;
        border-bottom: 2px solid #a5d6a7;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    
    /* Custom sidebar */
    .sidebar .sidebar-content { background-image: linear-gradient(#f1f8e9, #ffffff); }
    </style>
""", unsafe_html=True)

# 3. Data cấu hình: Nhãn món ăn và Data Dinh dưỡng chuyên sâu
# (Đảm bảo sắp xếp CLASS_NAMES đúng thứ tự lúc bạn train model)
CLASS_NAMES = ["Bánh mì", "Bánh xèo", "Bún chả", "Bún bò Huế", "Phở bò"]

FOOD_DETAILS = {
    "Bánh mì": {"calo": "380 kcal", "carb": "50g", "protein": "12g", "fat": "15g", "vitamin": "B1, B6, Sắt", "desc": "Món ăn đường phố biểu tượng. Thường gồm pate, chả giò, thịt nguội và rau dưa ngâm chua."},
    "Bánh xèo": {"calo": "420 kcal", "carb": "45g", "protein": "14g", "fat": "22g", "vitamin": "C, Chất xơ, Kẽm", "desc": "Vỏ bánh giòn rụm từ bột gạo và nghệ, nhân tôm thịt giá đỗ. Ăn kèm nước mắm chua ngọt và rổ rau rừng."},
    "Bún chả": {"calo": "540 kcal", "carb": "65g", "protein": "24g", "fat": "18g", "vitamin": "B3, B12, Sắt", "desc": "Thịt ba chỉ và chả viên nướng than hoa, thả trong bát nước mắm ấm vị đu đủ xanh, ăn cùng bún tươi."},
    "Bún bò Huế": {"calo": "580 kcal", "carb": "60g", "protein": "28g", "fat": "20g", "vitamin": "B6, Sắt, Canxi", "desc": "Hương vị đậm đà từ nước dùng xương ống ninh kèm mắm ruốc đặc trưng, sả tươi, sợi bún to dai mịn."},
    "Phở bò": {"calo": "450 kcal", "carb": "55g", "protein": "26g", "fat": "12g", "vitamin": "B12, Kẽm, Kali", "desc": "Tinh túy ẩm thực Việt. Nước dùng thanh trong được hầm từ xương bò và các loại thảo mộc (quế, hồi, thảo quả)."}
}

# 4. Hàm load model tối ưu bằng Cache
@st.cache_resource
def load_deep_model():
    return model = tf.keras.models.load_model('model_nhan_dang_mon_an.keras')
try:
    model = load_deep_model()
except Exception as e:
    st.error(f"❌ Cảnh báo hệ thống: Chưa tìm thấy file 'model_nhan_dang_mon_an.keras' trong cùng thư mục app.py! Lỗi: {e}")

# --- HEADER TỔNG THỂ ---
st.title("🍲 Hệ Thống Định Danh & Phân Tích Ẩm Thực Thực Tế")
st.caption("Phiên bản v3.0 nâng cấp - Tích hợp phân tích cấu trúc dinh dưỡng đa lượng (Macronutrients)")
st.markdown("---")

# --- SIDEBAR: ĐIỀU KHIỂN NÂNG CAO ---
with st.sidebar:
    st.markdown("### ⚙️ Bảng Điều Khiển AI")
    confidence_threshold = st.slider("Ngưỡng xác thực (%)", min_value=20, max_value=95, value=50, step=5)
    
    st.markdown("---")
    st.markdown("### 🖼️ Bộ Ảnh Mẫu Thử Nghiệm (Test Samples)")
    st.write("Nếu bạn không có sẵn ảnh, hãy chọn một ảnh mẫu bên dưới để kiểm tra nhanh:")
    
    # Tính năng ảnh test nhanh để khách hàng/thầy cô xem app chạy ngay lập tức
    sample_choice = st.selectbox("Chọn ảnh mẫu:", ["Không chọn", "Mẫu Bánh Mì", "Mẫu Phở Bò"])
    
    st.markdown("---")
    st.markdown("### 📊 Trạng thái Core")
    st.success("Tình trạng: 🟢 Hoạt động ổn định")
    st.info("Input Size: 224x224x3 (RGB)")

# --- XỬ LÝ NGUỒN ẢNH ĐẦU VÀO ---
img_source = None

# Nếu chọn ảnh mẫu
if sample_choice == "Mẫu Bánh Mì":
    # Link ảnh tượng trưng online để chạy thử nếu không có file
    img_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Banh_mi_vietnamien.JPG/640px-Banh_mi_vietnamien.JPG"
elif sample_choice == "Mẫu Phở Bò":
    img_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Pho_Bo_Hanoi.JPG/640px-Pho_Bo_Hanoi.JPG"

# Tạo layout hai cột chính
col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.markdown("### 📸 Cửa Sổ Nhập Liệu Ảnh")
    uploaded_file = st.file_uploader("Tải tệp ảnh món ăn lên hệ thống...", type=["jpg", "png", "jpeg"])
    
    # Ưu tiên ảnh do người dùng upload, nếu không thì lấy ảnh mẫu
    active_image = None
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
    elif img_source is not None:
        import requests
        from io import BytesIO
        try:
            response = requests.get(img_source)
            active_image = Image.open(BytesIO(response.content))
        except:
            st.error("Không thể tải ảnh mẫu từ internet. Vui lòng upload file thủ công.")

    if active_image is not None:
        st.image(active_image, caption="Hình ảnh đang được đưa vào mạng nơ-ron", use_container_width=True)

with col_right:
    st.markdown("### 📊 Kết Quả Trực Quan Từ Hệ Thống")
    
    if active_image is not None:
        # Bắt đầu tính toán
        t_start = time.time()
        
        with st.spinner("🧠 Khởi tạo ma trận điểm ảnh & chạy Inference..."):
            # Tiền xử lý dữ liệu ảnh cho khớp model функциональ_1 (224, 224, 3)
            img_ready = active_image.resize((224, 224))
            img_arr = tf.keras.preprocessing.image.img_to_array(img_ready)
            img_arr = np.expand_dims(img_arr, axis=0)
            
            # Dự đoán
            raw_preds = model.predict(img_arr)
            probs = tf.nn.softmax(raw_preds[0]).numpy() * 100
            
            top_idx = np.argmax(probs)
            top_label = CLASS_NAMES[top_idx]
            top_prob = probs[top_idx]
            
            t_inference = time.time() - t_start

        # Hiển thị kết quả dạng Premium Card
        if top_prob >= confidence_threshold:
            st.markdown(f"""
                <div class='premium-card'>
                    <p>MÓN ĂN NHẬN DIỆN ĐƯỢC</p>
                    <h2>{top_label}</h2>
                    <p>Độ tin cậy: {top_prob:.1f}%</p>
                </div>
            """, unsafe_html=True)
            
            # Hiển thị thông tin chi tiết dinh dưỡng đa lượng (Macronutrients) dạng bảng/cột
            if top_label in FOOD_DETAILS:
                food_info = FOOD_DETAILS[top_label]
                
                st.markdown("<div class='macro-box'>", unsafe_html=True)
                st.markdown(f"<div class='macro-title'>🥗 Thông Tin Thành Phần & Dinh Dưỡng</div>", unsafe_html=True)
                st.write(f"**Giới thiệu:** {food_info['desc']}")
                
                # Chia 4 cột nhỏ hiển thị chỉ số dinh dưỡng bằng Widget Metric
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("🔥 Năng lượng", food_info['calo'])
                m_col2.metric("🍞 Tinh bột", food_info['carb'])
                m_col3.metric("🥩 Đạm (Protein)", food_info['protein'])
                m_col4.metric("🥑 Chất béo", food_info['fat'])
                
                st.caption(f"💡 **Vi chất nổi bật:** {food_info['vitamin']}")
                st.markdown("</div>", unsafe_html=True)
        else:
            st.warning(f"⚠️ Kết quả nhận diện cao nhất ({top_prob:.1f}%) nằm dưới ngưỡng chấp nhận ({confidence_threshold}%). Không thể xuất báo cáo.")

        # BIỂU ĐỒ HOÀN HẢO ĐỒNG BỘ MÀU SẮC (Sử dụng thang màu Emerald/Green đồng bộ với App)
        st.write("<br>**📈 Phân bổ phần trăm xác suất giữa các lớp:**", unsafe_html=True)
        
        # Sắp xếp để món cao nhất nằm trên cùng của bảng thanh ngang
        sort_order = np.argsort(probs)
        lbls_sorted = [CLASS_NAMES[i] for i in sort_order]
        prbs_sorted = [probs[i] for i in sort_order]

        fig = go.Figure(go.Bar(
            x=prbs_sorted,
            y=lbls_sorted,
            orientation='h',
            marker=dict(
                color=prbs_sorted,
                colorscale=[[0, '#e8f5e9'], [0.5, '#4caf50'], [1, '#1b5e20']], # Đổ màu gradient từ xanh nhạt tới xanh đậm quý phái
                line=dict(color='#1b5e20', width=1.5)
            ),
            text=[f" {p:.1f}%" for p in prbs_sorted],
            textposition='outside'
        ))

        fig.update_layout(
            margin=dict(l=10, r=40, t=10, b=10),
            height=260,
            xaxis=dict(title='Xác suất cấu thành (%)', range=[0, 120], showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.caption(f"⚡ Thời gian phản hồi của kiến trúc mạng: {t_inference:.4f} giây")

    else:
        st.info("👋 Chào mừng bạn! Vui lòng tải một tấm ảnh ở vùng bên trái hoặc chọn nhanh 'Ảnh mẫu' ở thanh menu dọc bên trái để kiểm thử độ chính xác.")
