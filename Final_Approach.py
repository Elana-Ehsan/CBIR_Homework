import os
import json
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import torch
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# Streamlit 页面设置
st.set_page_config(page_title="基于图像内容的智能搜索工具", layout="wide")
st.title("基于图像内容的智能搜索工具")
st.sidebar.header("功能选项")



#-----------------------------------------内核部分-----------------------------------------
#---------利用分类标签比对的方法间接实现代码----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#使用GPU进行计算

# 加载 YOLO 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#使用GPU进行计算
model_YOLO = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# 1. 利用Yolo检测图像中的目标 输出标签、检测框、置信度
def detect_objects_Tag(image_path):
    results = model_YOLO(image_path)
    objects = []
    for _, row in results.pandas().xyxy[0].iterrows():
        objects.append({
            "label": row['name'],  # 目标类别
            "bbox": [row['xmin'], row['ymin'], row['xmax'], row['ymax']],  # 检测框
            "confidence": row['confidence']  # 置信度
        })
    # 示例返回值：{"label": "object", "box": [x_min, y_min, x_max, y_max]}
    return objects

# 2. 计算 IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # 避免除零错误
    if union_area == 0:
        return 0
    
    return inter_area / union_area

database = {}
# 3. 构建数据库
def build_database_Tag(image_folder):
    database = {}
    image_paths = glob.glob(f"{image_folder}/*.jpg")
    for image_path in image_paths:
        objects = detect_objects_Tag(image_path)
        database[image_path] = objects
    return database

# 4. 图像检索
def match_images_Tag(query_objects, database):
    results = []
    for image_path, db_objects in database.items():
        total_similarity = 0
        for q_obj in query_objects:
            for db_obj in db_objects:
                if q_obj['label'] == db_obj['label']: # 标签匹配
                    iou = calculate_iou(q_obj['bbox'], db_obj['bbox'])
                    total_similarity += iou # 加入 IoU 相似度（交集占比）
        results.append((image_path, total_similarity))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results

# 5. 显示图像及检测框
def display_image_with_boxes(image_path, objects):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    for obj in objects:
        bbox = obj['bbox']
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 10, f"{obj['label']} ({obj['confidence']:.2f})",
                color='red', fontsize=12, backgroundcolor="white")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


#---------利用CNN图像特征提取的方法直接实现----------



# 图像预处理和模型加载
model_CNN = models.resnet50(pretrained=True)
model_CNN = torch.nn.Sequential(*list(model_CNN.children())[:-1])  # 移除分类层
model_CNN = model_CNN.to(device)  # 将模型迁移到 GPU
model_CNN.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 提取图像特征
def extract_features(img_path_or_object):
    if isinstance(img_path_or_object, str):
        img = Image.open(img_path_or_object).convert("RGB")
    elif isinstance(img_path_or_object, Image.Image):  # 如果是 PIL 图像
        img = img_path_or_object.convert("RGB")
    else:
        raise ValueError("输入必须是文件路径或 PIL 图像对象")
    img_tensor = transform(img).unsqueeze(0).to(device)  # 数据迁移到 GPU
    with torch.no_grad():
        features = model_CNN(img_tensor).squeeze()
    return features.cpu().numpy()

# 构建数据库
def build_database_CNN(image_dir):
    features, paths = [], []
    database = {}
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        features.append(extract_features(img_path))
        paths.append(img_path)
    return np.array(features), np.array(paths)
        #database[img_path] = extract_features(img_path).tolist()#使用字典存储
    #return database临时替换

# 相似度计算、图像检索、显示查询结果
def display_results_CNN(query_path_or_object, db_features, db_paths,top_k=5):
    
    query_features = extract_features(query_path_or_object)
    print(type(query_path_or_object))
    # 计算相似度
    similarities = cosine_similarity(query_features.reshape(1, -1), db_features)
    top_indices = similarities.argsort()[0][-top_k:][::-1]

    # 显示查询图像
    st.subheader("查询图像")
    if isinstance(query_path_or_object, str):
        st.image(query_path_or_object, caption="查询图像", width=150)
    else:
        st.image(query_path_or_object, caption="查询图像", width=150)
    
    # 显示相似图像
    st.subheader("最相似的图像")
    cols = st.columns(5)
    for rank, idx in enumerate(top_indices, start=1):
        similar_img_path = db_paths[idx]
        similarity_score = similarities[0, idx]
        with cols[(rank-1) % 5]:#按列显示
            st.image(similar_img_path, caption=f"Rank {rank} - 相似度: {similarity_score:.4f}", width=150)


#-----------------------------------------UI部分-----------------------------------------
st.sidebar.header("GPU是否成功启用")
st.sidebar.subheader(torch.cuda.is_available())
# 用户上传的查询图像
query_image = st.sidebar.file_uploader("上传查询图片", type=["jpg","jpeg", "png"])

# 数据库路径设置
image_folder = st.sidebar.text_input("数据库图片文件夹路径", "D:\picture\Training")
st.sidebar.text("更换模型无需重新构建数据库")
st.sidebar.text("更换数据集需重新构建数据库")

# 构建数据库按钮
if st.sidebar.button("构建Tag数据库"):
    if image_folder:
        st.sidebar.write("正在构建数据库，请稍候...")
        database_Tag = build_database_Tag(image_folder)
        with open("database_Tag.json", "w") as f:
            json.dump(database_Tag, f)#将处理好的数据存储到文件中
        st.sidebar.success("Tag数据库已构建完成！")
    else:
        st.sidebar.error("请指定有效的图片文件夹路径。")
if st.sidebar.button("构建CNN数据库"):
    if image_folder:
        st.sidebar.write("正在构建数据库，请稍候...")
        database_CNN = build_database_CNN(image_folder)#以np数组的形式返回
        np.save("features.npy", database_CNN[0])  # 保存特征数组
        np.save("paths.npy", database_CNN[1])    # 保存路径数组
        st.sidebar.success("CNN数据库已构建完成！")
    else:
        st.sidebar.error("请指定有效的图片文件夹路径。")

method = st.sidebar.radio("选择检索方式", ["标签比对", "CNN特征向量提取"])
st.write(f"您选择了:{method}")

# 检索并展示结果
if query_image and st.sidebar.button("开始检索"):
    #查询图片
    query_image_pil = Image.open(query_image)
    query_image_path = "query_image.jpg"
    query_image_pil.save(query_image_path)

    if method == "标签比对":
        #打开加载数据库
        database_Tag = {}
        try:
            with open("database_Tag.json", "r") as f:
                database_Tag = json.load(f)
            st.sidebar.success("数据库加载成功！")
        except FileNotFoundError:
            st.sidebar.error("未找到数据库文件，请先构建数据库。")
        #运行查询程序
        st.subheader("基于Tag(使用YOLO模型)检索结果")
        # 检测查询图像目标
        query_objects = detect_objects_Tag(query_image_path)
        st.image(query_image_pil, caption="查询图像", width=150)
        # 检索相似图像
        matches = match_images_Tag(query_objects, database_Tag)  # Yolo比对
        # 显示检索结果
        st.subheader("最相似的图像")
        cols = st.columns(5)
        for idx, match in enumerate(matches[:5]):  # 显示前 5 个匹配结果
            matched_image_path, similarity = match
            buf = display_image_with_boxes(matched_image_path, database_Tag[matched_image_path])
            with cols[idx% 5]:
                st.image(buf, caption=f"rank: {idx+1} - 相似度: {similarity:.2f}", width=200)

    elif method == "CNN特征向量提取":
        database_CNN = {}
        try:
            features = np.load("features.npy")
            paths = np.load("paths.npy", allow_pickle=True)
            st.sidebar.success("数据库加载成功！")
        except FileNotFoundError:
            st.sidebar.error("未找到数据库文件，请先构建数据库。")
        # 使用 CNN 提取特征并检索
        #运行查询程序
        st.subheader("基于CNN提取特征检索结果")
        display_results_CNN(query_image_pil,features,paths,top_k=5)#CNN这边封装起来了