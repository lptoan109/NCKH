import os
from flask import Flask, url_for, session, redirect, render_template
from authlib.integrations.flask_client import OAuth

# --- 1. Cài đặt và Cấu hình ---

app = Flask(__name__)

# !!! THAY THẾ BẰNG MÃ BÍ MẬT CỦA BẠN
app.secret_key = 'your_super_secret_key_change_this' 

# !!! DÁN CLIENT ID VÀ CLIENT SECRET CỦA BẠN VÀO ĐÂY
GOOGLE_CLIENT_ID = '564904327189-4gsii5kfkht070218tsjqu8amnstc7o1.apps.googleusercontent.com'
GOOGLE_CLIENT_SECRET = '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'

oauth = OAuth(app)

# Đăng ký dịch vụ OAuth của Google
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)


# --- 2. Định nghĩa các trang (Routes) ---

@app.route('/')
def homepage():
    """Trang chủ - Hiển thị các nút đăng nhập/đăng ký."""
    # Kiểm tra xem người dùng đã đăng nhập chưa
    if 'user' in session:
        # Nếu đã đăng nhập, hiển thị avatar
        user = session.get('user')
        return render_template('index.html', user_is_logged_in=True, user_avatar=user.get('picture'))
    else:
        # Nếu chưa đăng nhập, hiển thị các nút
        return render_template('index.html', user_is_logged_in=False)

@app.route('/login')
def login():
    """Trang hiển thị form đăng nhập."""
    return render_template('login.html')

@app.route('/login/google')
def login_google():
    """Chuyển hướng người dùng đến trang đăng nhập của Google."""
    # URI mà Google sẽ trả người dùng về sau khi đăng nhập thành công
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/register')
def register():
    """Trang hiển thị form đăng ký."""
    return render_template('register.html')

@app.route('/authorize')
def authorize():
    """Trang nhận phản hồi từ Google sau khi người dùng đăng nhập."""
    # Lấy token và thông tin người dùng
    token = google.authorize_access_token()
    user_info = google.userinfo()
    
    # Lưu thông tin người dùng vào session
    session['user'] = user_info
    
    # Chuyển hướng về trang chủ
    return redirect('/')

@app.route('/logout')
def logout():
    """Xóa session và đăng xuất người dùng."""
    session.pop('user', None)
    return redirect('/')

# --- 3. Chạy ứng dụng ---

if __name__ == '__main__':
    # Chạy máy chủ ở chế độ debug để dễ dàng theo dõi lỗi
    app.run(debug=True)