from flask import Flask
from flask_cors import CORS
from app.routes import chat

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # 注册蓝图
    app.register_blueprint(chat.chat_bp)
    # 注册其他蓝图...
    
    return app