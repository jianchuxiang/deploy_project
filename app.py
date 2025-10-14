import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
import os

# 初始化Flask应用
app = Flask(__name__)

# --- 核心功能：加载模型和预处理器 ---
# 定义模型和文件的路径
MODEL_PATH = 'xgb_model.joblib'
SCALER_PATH = 'scaler.joblib'
# 更新为新脚本生成的通用文件名
FEATURES_PATH = 'top_features.joblib' 

# 在应用启动时，加载模型、预处理器和特征列表
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # 加载特征名称列表和数值特征列表
    top_features, numerical_features_final = joblib.load(FEATURES_PATH) # 更新变量名
    print("--- 模型和相关文件加载成功! ---")
    print(f"模型期望的 {len(top_features)} 个特征: {top_features}")
    print(f"其中需要标准化的数值特征: {numerical_features_final}")
except FileNotFoundError as e:
    print(f"错误：找不到必要的模型文件: {e}")
    print(f"请确保 xgb_model.joblib, scaler.joblib, 和 {FEATURES_PATH} 文件与 app.py 在同一目录下。")
    model, scaler, top_features, numerical_features_final = None, None, [], []
except Exception as e:
    print(f"加载模型或预处理器时发生未知错误: {e}")
    model, scaler, top_features, numerical_features_final = None, None, [], []


# --- 网页路由定义 ---

# 路由 1: 主页
@app.route('/')
def home():
    """
    渲染主页 (index.html)，并把特征列表和特征数量传递给前端页面。
    """
    if not top_features:
        return "错误：特征列表未加载，无法渲染页面。请检查服务器日志。", 500
    # 渲染 templates/index.html 文件，并传递两个变量
    return render_template('index.html', features=top_features, num_features=len(top_features))

# 路由 2: 预测 API
@app.route('/predict', methods=['POST'])
def predict():
    """
    接收来自前端的POST请求，处理数据并返回预测结果。
    """
    # 检查模型是否成功加载
    if not all([model, scaler, top_features]):
        return jsonify({'error': '模型或相关文件未成功加载，无法进行预测。请检查服务器日志。'}), 500

    try:
        # 1. 从POST请求中获取JSON格式的数据
        data = request.get_json(force=True)
        print(f"接收到的原始输入数据: {data}")

        # 2. 将收到的数据转换为Pandas DataFrame，并确保列的顺序与训练时完全一致
        input_df = pd.DataFrame([data])
        input_df = input_df[top_features]

        # 3. 使用加载好的Scaler对象来转换数值特征
        if numerical_features_final:
            input_df_scaled = input_df.copy()
            input_df_scaled[numerical_features_final] = scaler.transform(input_df[numerical_features_final])
            print(f"标准化后的数据:\n{input_df_scaled}")
        else:
            input_df_scaled = input_df
            print("没有需要标准化的数值特征。")


        # 4. 使用加载好的模型进行预测
        prediction_proba = model.predict_proba(input_df_scaled)[:, 1]
        prediction = model.predict(input_df_scaled)

        # 5. 格式化要返回给前端的JSON结果
        result = {
            'prediction_text': '预测结果: 是 类别 2/3 (Positive for Class 2/3)' if prediction[0] == 1 else '预测结果: 不是 类别 2/3 (Negative for Class 2/3)',
            'probability': f'{prediction_proba[0] * 100:.2f}%'
        }
        print(f"发送到前端的预测结果: {result}")

        return jsonify(result)

    except KeyError as e:
        # 如果前端发送的数据缺少某个特征，会触发这个错误
        error_msg = f"输入数据缺少必要特征: {e}"
        print(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        # 捕获其他所有可能的错误
        error_msg = f"预测过程中发生未知错误: {e}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

# --- 启动应用 ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

