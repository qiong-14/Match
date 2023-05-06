import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from pathlib import Path
import math
import numpy as np
import matplotlib.cm as cm
import torch
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
torch.set_grad_enabled(False)

# superglue 模型初始化，权重选用outdoor
# Load the SuperPoint and SuperGlue models.
device = 'cpu'
# print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
input_dir = Path("")
# print('Looking for data in directory \"{}\"'.format(input_dir))
# 输出路径
output_dir = Path("glue_result")
output_dir.mkdir(exist_ok=True, parents=True)
# print('Will write matches to directory \"{}\"'.format(output_dir))
matching = Matching(config).eval().to(device)

HEIGHT = 200
WIDTH = 300
# 定义文本样式（字体类型、颜色和大小）
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)   # 文本颜色为绿色
font_size = 0.6
delta_y = 20
start = 15
thick = 2
# 误差计算函数，用来计算特征点的重投影误差
def compute_error(image1, image2, keypoints1, keypoints2, good_matches=None):
    # 利用匹配点对计算单应变换矩阵H
    if good_matches == None:
        src_pts = np.float32(keypoints1).reshape(-1, 1, 2)
        dst_pts = np.float32(keypoints2).reshape(-1, 1, 2)
    else:
        if len(good_matches) > 10:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        else:
            result_image = []
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    result_image = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0] + 50))
    alpha = 0.5
    result_image[0:image2.shape[0], 0:image2.shape[1]] = alpha*image2
    result_image_1 = np.zeros((result_image.shape[0]+30, result_image.shape[1] + 30, 3), dtype=np.uint8)
    result_image_1[25:result_image.shape[0]+25, 25:result_image.shape[1]+25] = result_image
    # 利用单应变换映射源图像中的特征点
    src_pts_mapped = cv2.perspectiveTransform(src_pts, H)
    # 计算匹配点对之间的欧氏距离
    reproj_error = np.mean(np.linalg.norm(src_pts_mapped - dst_pts, axis=2))
    # 输出匹配误差
    # print("重投影误差为：%.2f" % reproj_error)
    return reproj_error, result_image_1



class ImageMatch(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.img1 = None
        self.img2 = None
        self.img1_path = None
        self.img2_path = None
        self.height = HEIGHT
        self.width = WIDTH
        
        # 创建左侧Fram，用于放置按钮Frame和图片显示Frame
        left_frame = tk.Frame(self)
        left_frame.pack(side='left', padx=10, pady=10)

        # 创建一个button_frame作为容器，用于放置选择图片按钮
        button_frame = tk.Frame(left_frame)
        button_frame.pack(side='top', padx=10, pady=10)
        self.button1 = tk.Button(button_frame, text="选择输入图片1", command=self.select_image1)
        self.button1.pack(side='left', padx=10, pady=10)
        self.button2 = tk.Button(button_frame, text="选择输入图片2", command=self.select_image2)
        self.button2.pack(side='left', padx=10, pady=10)
        
        # 创建一个left_picture_frame作为容器，用于放置左侧的矩形框显示图片
        left_picture_frame = tk.Frame(left_frame)
        left_picture_frame.pack(side='left', padx=10, pady=10)
        
        # 在左侧容器中创建矩形框1，并添加标题
        canvas_left_frame_1 = tk.Frame(left_picture_frame)
        canvas_left_frame_1.pack(padx=10, pady=10)
        self.canvas1_title = tk.Label(canvas_left_frame_1, text='输入图片1')
        self.canvas1_title.pack(side='top')
        self.canvas1 = tk.Canvas(canvas_left_frame_1, width=self.width, height=self.height, bd=1, relief='solid')
        self.canvas1.pack(side='bottom')

        # 在左侧容器中创建矩形框2，并添加标题
        canvas_frame2 = tk.Frame(left_picture_frame)
        canvas_frame2.pack(padx=10, pady=10)
        self.canvas2_title = tk.Label(canvas_frame2, text='输入图片2')
        self.canvas2_title.pack(side='top')
        self.canvas2 = tk.Canvas(canvas_frame2, width=self.width, height=self.height, bd=1, relief='solid')
        self.canvas2.pack(side='bottom')
        
        
        # 创建一个right_frame作为容器，用于放置右侧的算法选择和匹配按钮，以及结果图片框
        right_frame = tk.Frame(self)
        right_frame.pack(side='left', padx=10, pady=10)
        
        # 创建right_frame_top作为容器，放置算法选择和开始匹配按钮
        right_frame_top = tk.Frame(right_frame)
        right_frame_top.pack(side='top')

        # 在右侧容器中创建一个算法选择按钮
        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set("SIFT")
        algorithm_choices = ["SIFT", "ORB", "PCA-SIFT","SuperGlue"]
        self.algorithm_menu = tk.OptionMenu(right_frame_top, self.algorithm_var, *algorithm_choices)
        self.algorithm_menu.pack(side='left',padx=10, pady=10)

        # 在右侧容器中创建一个匹配按钮
        self.match_button = tk.Button(right_frame_top, text="开始匹配", command=self.match_images)
        self.match_button.pack(side='left', padx=10, pady=10)
        
        # 在右侧容器中创建一个result_frame作为容器，用于放置匹配误差显示框和结果图片框
        result_frame = tk.Frame(right_frame)
        result_frame.pack(side='top')

        # 在error_frame中创建一个误差显示框，并添加标题
        error_frame = tk.Frame(result_frame)
        error_frame.pack(side='top')
        self.error_title = tk.Label(error_frame, text='重投影误差')
        self.error_title.pack(side="left",padx=10, pady=10)
        self.error_label = tk.Label(error_frame, text='0')
        self.error_label.pack(side="left",padx=10, pady=10)

        result_frame_1 = tk.Frame(result_frame)
        result_frame_1.pack(side='top')

        result_frame_11 = tk.Frame(result_frame_1)
        result_frame_11.pack(side='top')

        self.result_title = tk.Label(result_frame_11, text='匹配特征')
        self.result_title.pack(side="top")
        self.result_canvas = tk.Canvas(result_frame_11,width=2*self.width + 10, height=self.height + 10, bd=1, relief='solid')
        self.result_canvas.create_rectangle(5, 5, 2*self.width + 10, self.height + 10,outline='blue', width=5)
        self.result_canvas.pack(side="top")

        result_frame_12 = tk.Frame(result_frame_1)
        result_frame_12.pack(side='top',padx=10, pady=10)
        self.result_title1 = tk.Label(result_frame_12, text='匹配图片')
        self.result_title1.pack(side="top")
        self.result_canvas1 = tk.Canvas(result_frame_12,width=2*self.width + 10, height= self.height + 110, bd=1, relief='solid')
        self.result_canvas1.create_rectangle(5, 5, 2*self.width + 10, self.height + 110, outline='red', width=5)
        self.result_canvas1.pack(side="top")

    def select_image1(self):
        # 选择图片
        file_path = filedialog.askopenfilename(initialdir="./", title="Select file",
                                               filetypes=(("image files", ["*.jpg","*.png","*.jpeg","*.bmp"]), ("all files", "*.*")))
        self.img1_path = file_path
        if file_path:
            # 读取图像并resize
            self.img1 = cv2.imread(self.img1_path)
            self.img1 = cv2.resize(self.img1, (self.width, self.height))
            # 加载图片
            image = Image.fromarray(cv2.cvtColor(self.img1,cv2.COLOR_BGR2RGB))
            # 将图片转换为PhotoImage对象
            photo = ImageTk.PhotoImage(image)
            # 将图片显示到画布上
            self.canvas1.create_image(0, 0, image=photo, anchor='nw')
            # 保存photo对象的引用，以防止被垃圾回收
            self.canvas1.image = photo

    def select_image2(self):
        # 选择图片
        file_path = filedialog.askopenfilename(initialdir="./", title="Select file",
                                               filetypes=(("image files", ["*.jpg","*.png","*.jpeg","*.bmp"]), ("all files", "*.*")))
        self.img2_path = file_path

        if file_path:
           # 读取图像并resize
            self.img2 = cv2.imread(self.img2_path)
            self.img2 = cv2.resize(self.img2, (self.width, self.height))
            # 加载图片
            image = Image.fromarray(cv2.cvtColor(self.img2,cv2.COLOR_BGR2RGB))
            # 将图片转换为PhotoImage对象
            photo = ImageTk.PhotoImage(image)
            # 将图片显示到画布上
            self.canvas2.create_image(0, 0, image=photo, anchor='nw')
            # 保存photo对象的引用，以防止被垃圾回收
            self.canvas2.image = photo

    
    def match_images(self):
        # print(self.img1_path, self.img2_path)
        # 根据选择的算法进行匹配，并获取匹配结果图像
        algorithm = self.algorithm_var.get()
        if algorithm == "SIFT":
            img_match, reproj_error, wrap_image = self.sift_match()
        elif algorithm == "ORB":
            img_match, reproj_error, wrap_image = self.orb_match()
        elif algorithm == "PCA-SIFT":
            img_match, reproj_error, wrap_image = self.pcasift_match()
        elif algorithm == "SuperGlue":
            img_match, reproj_error, wrap_image = self.superGlueMatch(self.img1_path, self.img2_path)
        else:
            img_match = np.zeros((self.width, self.height, 3), dtype=np.uint8)
            wrap_image = np.zeros((self.width, self.height, 3), dtype=np.uint8)
            reproj_error = 0

        # 将匹配结果图像调整大小，并显示到结果图片框中
        img_match = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
        img_match = Image.fromarray(img_match)
        photo = ImageTk.PhotoImage(img_match)
        self.result_canvas.create_image(10, 10, image=photo, anchor='nw')
        self.result_canvas.image = photo
        serror = "{:.4f}".format(reproj_error)
        self.error_label.config(text=serror)

        # wrap_image = cv2.resize(wrap_image, (self.width * 2, self.height * 2))
        wrap_image = cv2.cvtColor(wrap_image, cv2.COLOR_BGR2RGB)
        wrap_image = Image.fromarray(wrap_image)
        photo = ImageTk.PhotoImage(wrap_image)
        self.result_canvas1.create_image(10, 10, image=photo, anchor='nw')
        self.result_canvas1.image = photo

    # sift算法
    def sift_match(self, threshold=0.8):
        img1 = self.img1.copy()
        img2 = self.img2.copy()
        # 初始化 SIFT 特征检测器
        sift = cv2.SIFT_create()

        # 检测图像中的关键点和描述符
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        # 使用 FLANN 匹配算法进行特征点匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # 提取最佳匹配结果
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)


        # 绘制匹配结果
        result_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # result_img = cv2.resize(result_img, (self.width * 2, self.height * 2))
        # 在图像上绘制文本，坐标为(10, 30)，字体类型为font，颜色为color，字体大小为font_size
        text1 = "SIFT"
        text2 = "KeyPoints: {}:{}".format(len(keypoints1), len(keypoints2))
        text3 = "Matches: {}".format(len(good_matches))

        cv2.putText(result_img, text1, (0, start), font, font_size, color, thickness=thick)
        cv2.putText(result_img, text2, (0, start + delta_y), font, font_size, color, thickness=thick)
        cv2.putText(result_img, text3, (0, start + 2 * delta_y), font, font_size, color, thickness=thick)

        # 计算重投影误差
        reproj_error, wrap_image = compute_error(img1, img2, keypoints1, keypoints2, good_matches)

        return result_img, reproj_error, wrap_image

    # orb算法
    def orb_match(self, threshold=0.8):
        img1 = self.img1.copy()
        img2 = self.img2.copy()
        # 初始化ORB检测器
        orb = cv2.ORB_create()

        # 使用ORB算法在两张图片中检测特征点和描述符
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # 创建BFMatcher匹配器对象
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 匹配两张图片中的特征点
        matches = bf.match(des1, des2)

        # # 按匹配程度从小到大排序
        # matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches

        # 绘制匹配结果，并显示
        result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # result_img = cv2.resize(result_img, (self.width * 2, self.height * 2))
        # 在图像上绘制文本，坐标为(10, 30)，字体类型为font，颜色为color，字体大小为font_size
        text1 = "ORB"
        text2 = "KeyPoints: {}:{}".format(len(kp1), len(kp2))
        text3 = "Matches: {}".format(len(good_matches))
        cv2.putText(result_img, text1, (0, start), font, font_size, color, thickness=thick)
        cv2.putText(result_img, text2, (0, start + delta_y), font, font_size, color, thickness=thick)
        cv2.putText(result_img, text3, (0, start + 2 * delta_y), font, font_size, color, thickness=thick)
        reproj_error, wrap_image = compute_error(img1, img2, kp1, kp2, good_matches)

        return result_img, reproj_error, wrap_image

    def pcasift_match(self, threshold=0.8):

        img1 = self.img1.copy()
        img2 = self.img2.copy()

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        pca = sklearnPCA(n_components=120)
        det = np.concatenate([des1, des2])
        pca.fit(det)
        des1_pca = pca.transform(des1)
        des2_pca = pca.transform(des2)

        # 创建flann匹配器
        flann = cv2.FlannBasedMatcher()

        # 使用knnMatch()匹配
        matches = flann.knnMatch(des1_pca, des2_pca, k=2)

        # 提取最佳匹配结果
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        # 绘制匹配结果
        result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # 在图像上绘制文本，坐标为(10, 30)，字体类型为font，颜色为color，字体大小为font_size
        text1 = "PCA_SIFT"
        text2 = "KeyPoints: {}:{}".format(len(kp1), len(kp2))
        text3 = "Matches: {}".format(len(good_matches))
        cv2.putText(result_img, text1, (0, start), font, font_size, color, thickness=thick)
        cv2.putText(result_img, text2, (0, start + delta_y), font, font_size, color, thickness=thick)
        cv2.putText(result_img, text3, (0, start + 2 * delta_y), font, font_size, color, thickness=thick)
        reproj_error, wrap_image = compute_error(img1, img2, kp1, kp2, good_matches)

        return result_img, reproj_error, wrap_image



    def superGlueMatch(self, img1_path, img2_path):
        name0, name1 = img1_path, img2_path
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        # matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, "png")
        rot0, rot1 = 0, 0
        # Load the image pair.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, (HEIGHT, WIDTH),rot0, False)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, (HEIGHT, WIDTH), rot1, False)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem0, stem1),
        ]

        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            [], viz_path, True,
            True, False, 'Matches',[])
        result_img = cv2.imread(str(viz_path))
        result_img = cv2.resize(result_img, (self.width * 2, self.height))
        text1 = "SuperGlue"
        text2 = "KeyPoints: {}:{}".format(len(kpts0), len(kpts1))
        text3 = "Matches: {}".format(len(mkpts0))
        cv2.putText(result_img, text1, (0, start), font, font_size, (0, 255,0), thickness=thick)
        cv2.putText(result_img, text2, (0, start + delta_y), font, font_size, (0, 255,0), thickness=thick)
        cv2.putText(result_img, text3, (0, start + 2 * delta_y), font, font_size, (0, 255,0), thickness=thick)
        reproj_error, wrap_image = compute_error(self.img1.copy(), self.img2.copy(), mkpts0, mkpts1)
        return result_img, reproj_error,wrap_image
if __name__ == "__main__":
    root = tk.Tk()
    root.title("图片匹配")
    app = ImageMatch(master=root)
    app.mainloop()