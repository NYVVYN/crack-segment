import sys
import os
import json
import time
import uuid
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel,
                             QFileDialog, QScrollArea, QFrame, QGraphicsView,
                             QGraphicsScene, QMessageBox, QMenu)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPainter, QAction
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread

MODERN_STYLE = """
QMainWindow { background-color: #F8FAFC; }
QFrame#Dashboard { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; }
QFrame#DropArea { background-color: #F1F5F9; border: 2px dashed #CBD5E1; border-radius: 16px; }
QFrame#DropArea:hover { border-color: #3B82F6; background-color: #EFF6FF; }

QPushButton#NavBtn {
    background-color: transparent; color: #64748B; font-weight: bold; font-size: 15px;
    border: none; border-bottom: 3px solid transparent; padding: 10px 20px;
}
QPushButton#NavBtn:hover { color: #3B82F6; }
QPushButton#NavBtn:checked { color: #3B82F6; border-bottom: 3px solid #3B82F6; }

QPushButton#PrimaryBtn { background-color: #3B82F6; color: white; border-radius: 8px; font-weight: bold; font-size: 14px;}
QPushButton#PrimaryBtn:hover { background-color: #2563EB; }
QPushButton#StopBtn { background-color: #EF4444; color: white; border-radius: 8px; font-weight: bold; font-size: 14px;}
QPushButton#StopBtn:hover { background-color: #DC2626; }
QPushButton#ExportBtn { background-color: #10B981; color: white; border-radius: 8px; font-weight: bold; font-size: 14px;}
QPushButton#ExportBtn:hover { background-color: #059669; }

QPushButton#BatchDelBtn { background-color: #EF4444; color: white; border-radius: 6px; padding: 5px 12px; font-weight: bold; }
QPushButton#BatchDelBtn:hover { background-color: #DC2626; }
QPushButton#BatchExpBtn { background-color: #F59E0B; color: white; border-radius: 6px; padding: 5px 12px; font-weight: bold; }
QPushButton#BatchExpBtn:hover { background-color: #D97706; }

QPushButton#HistoryBtn { background-color: transparent; border: 2px solid transparent; border-radius: 8px; }
QPushButton#HistoryBtn:hover { border: 2px solid #3B82F6; background-color: #E2E8F0; }

QMenu { background-color: white; border: 1px solid #D1D5DB; border-radius: 5px; padding: 5px; font-family: "Microsoft YaHei"; }
QMenu::item { padding: 8px 25px; border-radius: 4px; }
QMenu::item:selected { background-color: #FEE2E2; color: #DC2626; }

QPushButton#OverlayCheck {
    background-color: rgba(255, 255, 255, 0.5);
    border: 1px solid #94A3B8; border-radius: 4px;
    font-size: 12px; font-weight: bold;
}
QPushButton#OverlayCheck:hover { background-color: rgba(255, 255, 255, 0.8); }
QPushButton#OverlayCheck:checked {
    background-color: #3B82F6; border: 1px solid #2563EB; color: white;
}
"""


class VideoDetectionThread(QThread):
    frame_ready = pyqtSignal(QPixmap)
    stats_ready = pyqtSignal(dict)

    def __init__(self, model, source, conf=0.25):
        super().__init__()
        self.model = model
        self.source = source
        self.conf = conf
        self.running = True
        self.paused = False

    def run(self):
        cap = cv2.VideoCapture(self.source)
        while self.running and cap.isOpened():
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret: break

            results = self.model.predict(frame, conf=self.conf, verbose=False)
            for result in results:
                annotated_frame = result.plot()
                crack_count = len(result.masks) if result.masks else 0
                inference_time = round(sum(result.speed.values()), 1)
                h, w = result.orig_shape

                if result.boxes and len(result.boxes) > 0:
                    confs = result.boxes.conf.tolist()
                    conf_text = f"{sum(confs) / len(confs):.1%} (最高: {max(confs):.1%})"
                else:
                    conf_text = "N/A (未检出)"

                bytes_per_line = 3 * w
                cv_rgb_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                qt_img = QImage(cv_rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.frame_ready.emit(QPixmap.fromImage(qt_img))
                self.stats_ready.emit({
                    "count": crack_count, "time_ms": inference_time,
                    "resolution": f"{w} x {h}", "conf_text": conf_text
                })
            time.sleep(0.01)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class HistoryThumbnail(QWidget):
    clicked = pyqtSignal(str)
    rightClicked = pyqtSignal(object, str, object)
    selectionChanged = pyqtSignal()

    def __init__(self, file_path, icon_path, data_type, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.data_type = data_type
        self.setFixedSize(90, 90)

        self.img_btn = QPushButton(self)
        self.img_btn.setFixedSize(90, 90)
        self.img_btn.setIcon(QIcon(icon_path))
        self.img_btn.setIconSize(QSize(80, 80))
        prefix = "🎞️ 视频: " if data_type == "video" else "🖼️ 图片: "
        self.img_btn.setToolTip(prefix + file_path)
        self.img_btn.setObjectName("HistoryBtn")
        self.img_btn.clicked.connect(lambda: self.clicked.emit(self.file_path))

        self.img_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.img_btn.customContextMenuRequested.connect(
            lambda pos: self.rightClicked.emit(self.img_btn.mapToGlobal(pos), self.file_path, self)
        )

        self.check_btn = QPushButton(self)
        self.check_btn.setObjectName("OverlayCheck")
        self.check_btn.setCheckable(True)
        self.check_btn.setGeometry(65, 5, 20, 20)
        self.check_btn.toggled.connect(self.on_checked)

    def on_checked(self, is_checked):
        self.check_btn.setText("✔" if is_checked else "")
        self.selectionChanged.emit()

    def is_selected(self): return self.check_btn.isChecked()


class DragDropArea(QFrame):
    files_dropped = pyqtSignal(list)
    clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("DropArea")
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        self.icon_label = QLabel("📁")
        self.icon_label.setFont(QFont("Arial", 48))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label = QLabel("点击或拖拽上传文件")
        self.text_label.setFont(QFont("Microsoft YaHei", 14))
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("color: #64748B;")
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)

    def update_mode(self, mode):
        if mode == "image":
            self.icon_label.setText("🖼️")
            self.text_label.setText("图片分析模式\n点击选择图片或将图片拖拽至此")
        elif mode == "video":
            self.icon_label.setText("🎬")
            self.text_label.setText("视频分析模式\n点击选择或拖入本地 MP4/AVI 视频")
        elif mode == "camera":
            self.icon_label.setText("📷")
            self.text_label.setText("实时监控模式\n请点击此区域或右侧按钮唤醒摄像头")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if paths: self.files_dropped.emit(paths)


class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pixmap_item = None

    def set_image(self, pixmap):
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self.pixmap_item: return
        zoom_in_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(1 / zoom_in_factor, 1 / zoom_in_factor)


class CrackDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("裂隙检测系统")
        self.resize(1200, 850)
        self.setStyleSheet(MODERN_STYLE)

        self.inference_conf = 0.25
        self.current_mode = "image"
        self.video_thread = None
        self.current_image_path = None
        self.current_result_data = None
        self.history_items = []

        self.cache_dir = "history_cache"
        self.json_path = os.path.join(self.cache_dir, "records.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.history_cache = self.load_history_from_disk()

        print("正在加载 YOLO 模型，请稍候...")
        self.model = YOLO("best.pt")
        print("模型加载完成！")

        self.init_ui()
        self.restore_ui_from_history()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 10, 20, 20)

        # 1. 顶部导航
        self.nav_bar = QWidget()
        nav_layout = QHBoxLayout(self.nav_bar)
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.setSpacing(30)

        self.btn_mode_img = QPushButton("🖼️ 静态图片")
        self.btn_mode_vid = QPushButton("🎬 本地视频")
        self.btn_mode_cam = QPushButton("📷 实时监控")
        self.nav_btns = [self.btn_mode_img, self.btn_mode_vid, self.btn_mode_cam]
        for btn in self.nav_btns:
            btn.setObjectName("NavBtn")
            btn.setCheckable(True)
            nav_layout.addWidget(btn)

        self.btn_mode_img.clicked.connect(lambda: self.switch_mode("image"))
        self.btn_mode_vid.clicked.connect(lambda: self.switch_mode("video"))
        self.btn_mode_cam.clicked.connect(lambda: self.switch_mode("camera"))
        self.main_layout.addWidget(self.nav_bar)

        self.batch_panel = QWidget()
        batch_layout = QHBoxLayout(self.batch_panel)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_info_lbl = QLabel("已选中 0 项")
        self.batch_info_lbl.setStyleSheet("color: #3B82F6; font-weight: bold;")
        self.batch_del_btn = QPushButton("批量删除");
        self.batch_del_btn.setObjectName("BatchDelBtn")
        self.batch_del_btn.clicked.connect(self.batch_delete)
        self.batch_exp_btn = QPushButton("批量导出");
        self.batch_exp_btn.setObjectName("BatchExpBtn")
        self.batch_exp_btn.clicked.connect(self.batch_export)
        batch_layout.addWidget(self.batch_info_lbl);
        batch_layout.addWidget(self.batch_del_btn);
        batch_layout.addWidget(self.batch_exp_btn);
        batch_layout.addStretch()
        self.batch_panel.hide()
        self.main_layout.addWidget(self.batch_panel)

        self.history_area = QScrollArea()
        self.history_area.setFixedHeight(115)
        self.history_widget = QWidget()
        self.history_layout = QHBoxLayout(self.history_widget)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.history_area.setWidget(self.history_widget)
        self.history_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.history_area)

        self.split_layout = QHBoxLayout()
        self.main_layout.addLayout(self.split_layout)

        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)

        self.drop_area = DragDropArea()
        self.drop_area.files_dropped.connect(self.handle_dropped_files)
        self.image_viewer = ImageViewer()
        self.image_viewer.hide()

        self.left_layout.addWidget(self.drop_area)
        self.left_layout.addWidget(self.image_viewer)
        self.split_layout.addWidget(self.left_panel, stretch=7)

        self.dashboard = QFrame()
        self.dashboard.setObjectName("Dashboard")
        self.dash_layout = QVBoxLayout(self.dashboard)
        self.dash_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.dash_layout.addWidget(QLabel("数据分析面板", font=QFont("Microsoft YaHei", 14, QFont.Weight.Bold)))
        self.status_label = QLabel("状态：就绪")
        self.dash_layout.addWidget(self.status_label)

        conf_container = QWidget();
        conf_layout = QVBoxLayout(conf_container);
        conf_layout.setContentsMargins(0, 5, 0, 5)
        conf_layout.addWidget(QLabel("当前帧检测置信度", font=QFont("Microsoft YaHei", 11, QFont.Weight.Bold)))
        self.conf_display_lbl = QLabel("-");
        self.conf_display_lbl.setFont(QFont("Arial", 16, QFont.Weight.Bold));
        self.conf_display_lbl.setStyleSheet("color: #3B82F6;")
        conf_layout.addWidget(self.conf_display_lbl)
        self.dash_layout.addWidget(conf_container)

        self.result_count_lbl = QLabel("裂隙数量：-")
        self.result_time_lbl = QLabel("推理耗时：-")
        self.result_res_lbl = QLabel("画面尺寸：-")
        for lbl in [self.result_count_lbl, self.result_time_lbl, self.result_res_lbl]:
            lbl.setFont(QFont("Microsoft YaHei", 11))
            self.dash_layout.addWidget(lbl)

        self.dash_layout.addStretch()

        self.action_btn_1 = QPushButton("操作 1");
        self.action_btn_1.setObjectName("PrimaryBtn");
        self.action_btn_1.setFixedHeight(45)
        self.action_btn_2 = QPushButton("操作 2");
        self.action_btn_2.setObjectName("StopBtn");
        self.action_btn_2.setFixedHeight(45)
        self.dash_layout.addWidget(self.action_btn_1);
        self.dash_layout.addWidget(self.action_btn_2)
        self.split_layout.addWidget(self.dashboard, stretch=3)

        self.switch_mode("image")

    def switch_mode(self, mode):
        self.stop_video_stream()

        self.current_mode = mode
        self.image_viewer.hide()
        self.drop_area.show()
        self.drop_area.update_mode(mode)

        self.btn_mode_img.setChecked(mode == "image")
        self.btn_mode_vid.setChecked(mode == "video")
        self.btn_mode_cam.setChecked(mode == "camera")

        try:
            self.action_btn_1.clicked.disconnect()
            self.action_btn_2.clicked.disconnect()
            self.drop_area.clicked.disconnect()
        except TypeError:
            pass

        if mode == "image":
            self.action_btn_1.setText("批量上传图片");
            self.action_btn_1.setObjectName("PrimaryBtn")
            self.action_btn_1.clicked.connect(self.action_open_local_images)
            self.drop_area.clicked.connect(self.action_open_local_images)

            self.action_btn_2.setText("导出当前图片报告");
            self.action_btn_2.setObjectName("ExportBtn")
            self.action_btn_2.setEnabled(False)
            self.action_btn_2.clicked.connect(self.export_single_report)

        elif mode == "video":
            self.action_btn_1.setText("选择本地视频播放");
            self.action_btn_1.setObjectName("PrimaryBtn")
            self.action_btn_1.clicked.connect(self.action_open_local_video)
            self.drop_area.clicked.connect(self.action_open_local_video)

            # 【修复点】：本地视频变成暂停控制
            self.action_btn_2.setText("暂停播放");
            self.action_btn_2.setObjectName("StopBtn")
            self.action_btn_2.setEnabled(False)
            self.action_btn_2.clicked.connect(self.toggle_video_pause)

        elif mode == "camera":
            self.action_btn_1.setText("唤醒摄像头");
            self.action_btn_1.setObjectName("PrimaryBtn")
            self.action_btn_1.clicked.connect(self.action_start_camera)
            self.drop_area.clicked.connect(self.action_start_camera)

            self.action_btn_2.setText("关闭摄像头");
            self.action_btn_2.setObjectName("StopBtn")
            self.action_btn_2.setEnabled(False)
            self.action_btn_2.clicked.connect(self.stop_video_stream)

        self.action_btn_1.style().unpolish(self.action_btn_1);
        self.action_btn_1.style().polish(self.action_btn_1)
        self.action_btn_2.style().unpolish(self.action_btn_2);
        self.action_btn_2.style().polish(self.action_btn_2)

    def action_open_local_images(self, checked=False):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "批量选择待检测的图片", "",
                                                     "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_paths: self.process_multiple_images(file_paths)

    def action_open_local_video(self, checked=False):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择本地视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if file_path: self.start_video_stream(file_path)

    def action_start_camera(self, checked=False):
        self.start_video_stream(0)

    def handle_dropped_files(self, file_paths):
        if self.current_mode == "image":
            valid = [p for p in file_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            self.process_multiple_images(valid)
        elif self.current_mode == "video":
            valid = [p for p in file_paths if p.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if valid: self.start_video_stream(valid[0])

    def process_multiple_images(self, file_paths):
        if not file_paths: return
        self.drop_area.hide();
        self.image_viewer.show()
        total = len(file_paths)
        for i, path in enumerate(file_paths):
            self.status_label.setText(f"正在批量分析图像 ({i + 1}/{total})...")
            QApplication.processEvents()
            is_last = (i == total - 1)
            self.process_single_image(path, update_main_ui=is_last)
        self.status_label.setText(f"批量分析完成 (共 {total} 张)");
        self.status_label.setStyleSheet("color: #059669; font-weight: bold;")

    def process_single_image(self, file_path, update_main_ui=True):
        if update_main_ui: self.current_image_path = file_path
        if file_path in self.history_cache and self.history_cache[file_path].get("type", "image") == "image":
            if update_main_ui: self.load_from_history(file_path)
            return

        results = self.model.predict(source=file_path, conf=self.inference_conf)
        for result in results:
            crack_count = len(result.masks) if result.masks else 0
            annotated_frame = result.plot()
            inference_time = round(sum(result.speed.values()), 1)
            h, w = result.orig_shape

            if result.boxes and len(result.boxes) > 0:
                confs = result.boxes.conf.tolist()
                conf_text = f"{sum(confs) / len(confs):.1%} (最高: {max(confs):.1%})"
            else:
                conf_text = "N/A (未检出)"

            cache_img_name = f"{uuid.uuid4().hex}.jpg"
            cache_img_path = os.path.join(self.cache_dir, cache_img_name)
            cv2.imencode('.jpg', annotated_frame)[1].tofile(cache_img_path)

            cache_data = {
                "type": "image", "annotated_path": cache_img_path,
                "count": crack_count, "time_ms": inference_time,
                "resolution": f"{w} x {h}", "conf_text": conf_text
            }
            self.history_cache[file_path] = cache_data
            self.save_history_to_disk()
            self.add_history_thumbnail(file_path, cache_img_path, "image")
            if update_main_ui: self.update_dashboard_stats(cache_data, cv_img=annotated_frame)

    def start_video_stream(self, source):
        self.drop_area.hide();
        self.image_viewer.show()
        self.status_label.setText("视频流分析中...");
        self.status_label.setStyleSheet("color: #059669; font-weight: bold;")

        self.action_btn_1.setEnabled(False)
        self.action_btn_2.setEnabled(True)
        if self.current_mode == "video":
            self.action_btn_2.setText("暂停播放")
            self.action_btn_2.setObjectName("StopBtn")
            self.action_btn_2.style().unpolish(self.action_btn_2);
            self.action_btn_2.style().polish(self.action_btn_2)

        self.video_thread = VideoDetectionThread(model=self.model, source=source, conf=self.inference_conf)
        self.video_thread.frame_ready.connect(self.image_viewer.set_image)
        self.video_thread.stats_ready.connect(self.update_dashboard_stats)
        self.video_thread.start()

        if isinstance(source, str) and source not in self.history_cache:
            cap = cv2.VideoCapture(source)
            ret, frame = cap.read()
            cap.release()
            if ret:
                thumb_name = f"vid_thumb_{uuid.uuid4().hex}.jpg"
                thumb_path = os.path.join(self.cache_dir, thumb_name)
                thumb_img = cv2.resize(frame, (180, 180))
                cv2.imencode('.jpg', thumb_img)[1].tofile(thumb_path)
                cache_data = {"type": "video", "thumb_path": thumb_path}
                self.history_cache[source] = cache_data
                self.save_history_to_disk()
                self.add_history_thumbnail(source, thumb_path, "video")

    def toggle_video_pause(self):
        if not self.video_thread: return

        if self.video_thread.paused:
            self.video_thread.resume()
            self.action_btn_2.setText("暂停播放")
            self.action_btn_2.setObjectName("StopBtn")
            self.status_label.setText("视频流分析中...")
            self.status_label.setStyleSheet("color: #059669; font-weight: bold;")
        else:
            self.video_thread.pause()
            self.action_btn_2.setText("继续播放")
            self.action_btn_2.setObjectName("ExportBtn")
            self.status_label.setText("视频已暂停")
            self.status_label.setStyleSheet("color: #D97706; font-weight: bold;")

        # 刷新样式
        self.action_btn_2.style().unpolish(self.action_btn_2)
        self.action_btn_2.style().polish(self.action_btn_2)

    def stop_video_stream(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        self.status_label.setText("监控已关闭");
        self.status_label.setStyleSheet("color: #DC2626; font-weight: bold;")
        self.action_btn_1.setEnabled(True);
        self.action_btn_2.setEnabled(False)

    def update_dashboard_stats(self, data, cv_img=None):
        if cv_img is not None:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            cv_rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            qt_img = QImage(cv_rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_viewer.set_image(QPixmap.fromImage(qt_img))

        self.conf_display_lbl.setText(data.get('conf_text', '-'))
        self.result_count_lbl.setText(
            f"裂隙数量：<span style='color:#DC2626; font-weight:bold;'>{data.get('count', '-')} 处</span>")
        self.result_time_lbl.setText(f"推理耗时：{data.get('time_ms', '-')} ms")
        self.result_res_lbl.setText(f"画面尺寸：{data.get('resolution', '-')}")

        if self.current_mode == "image":
            self.action_btn_2.setEnabled(True)
            self.current_result_data = {
                "source": self.current_image_path, "count": data.get('count', 0),
                "time": data.get('time_ms', 0), "conf_text": data.get('conf_text', '-')
            }

    def on_history_clicked(self, file_path):
        data = self.history_cache.get(file_path)
        if not data: return
        data_type = data.get("type", "image")
        if data_type == "image":
            if self.current_mode != "image": self.switch_mode("image")
            self.load_from_history(file_path)
        elif data_type == "video":
            if self.current_mode != "video": self.switch_mode("video")
            self.start_video_stream(file_path)

    def load_from_history(self, file_path):
        self.drop_area.hide();
        self.image_viewer.show()
        data = self.history_cache[file_path]
        self.current_image_path = file_path
        img_data = np.fromfile(data["annotated_path"], dtype=np.uint8)
        cv_img = cv2.imdecode(img_data, -1)
        self.update_dashboard_stats(data, cv_img=cv_img)
        self.status_label.setText("分析完成")

    def load_history_from_disk(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except:
                    return {}
        return {}

    def save_history_to_disk(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.history_cache, f, ensure_ascii=False, indent=4)

    def restore_ui_from_history(self):
        for path, data in self.history_cache.items():
            data_type = data.get("type", "image")
            icon_path = data.get("annotated_path") if data_type == "image" else data.get("thumb_path")
            if icon_path and os.path.exists(icon_path):
                self.add_history_thumbnail(path, icon_path, data_type)

    def add_history_thumbnail(self, file_path, icon_path, data_type):
        for item in self.history_items:
            if item.file_path == file_path: return
        thumb = HistoryThumbnail(file_path, icon_path, data_type)
        thumb.clicked.connect(self.on_history_clicked)
        thumb.rightClicked.connect(self.delete_single_history)
        thumb.selectionChanged.connect(self.handle_selection_change)
        self.history_items.append(thumb)
        self.history_layout.insertWidget(0, thumb)

    def handle_selection_change(self):
        selected_count = sum(1 for item in self.history_items if item.is_selected())
        if selected_count > 0:
            self.batch_info_lbl.setText(f"已选中 {selected_count} 项")
            self.batch_panel.show()
        else:
            self.batch_panel.hide()

    def delete_single_history(self, pos, file_path, thumb_widget):
        menu = QMenu(self)
        menu.addAction(QAction("彻底删除此项", self))
        if menu.exec(pos): self._execute_delete([thumb_widget])

    def batch_delete(self):
        selected = [item for item in self.history_items if item.is_selected()]
        if selected and QMessageBox.question(self, "批量删除",
                                             f"确定彻底删除 {len(selected)} 项记录吗？") == QMessageBox.StandardButton.Yes:
            self._execute_delete(selected)

    def _execute_delete(self, thumb_widgets):
        for thumb in thumb_widgets:
            path = thumb.file_path
            thumb.deleteLater()
            self.history_items.remove(thumb)
            if path in self.history_cache:
                data = self.history_cache[path]
                data_type = data.get("type", "image")
                img_to_del = data.get("annotated_path") if data_type == "image" else data.get("thumb_path")
                if img_to_del and os.path.exists(img_to_del): os.remove(img_to_del)
                del self.history_cache[path]

            if self.current_image_path == path:
                self.image_viewer.hide()
                self.drop_area.show()
                self.current_image_path = None
        self.save_history_to_disk()
        self.handle_selection_change()

    def export_single_report(self):
        if not self.current_result_data: return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存当前报告", "单次裂隙检测报告.txt", "Text Files (*.txt)")
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("=" * 40 + "\n      裂隙检测系统 - 单次分析报告      \n" + "=" * 40 + "\n\n")
                f.write(f"检测图片: {self.current_result_data['source']}\n")
                f.write(f"检测确信度: {self.current_result_data.get('conf_text', '-')}\n")
                f.write(f"发现裂隙: {self.current_result_data['count']} 处\n")
                f.write(f"算法耗时: {self.current_result_data['time']} ms\n")
            QMessageBox.information(self, "导出成功", f"报告已成功保存至：\n{save_path}")

    def batch_export(self):
        selected_items = [item for item in self.history_items if item.is_selected()]
        if not selected_items: return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存批量分析报告", "批量裂隙检测汇总.txt",
                                                   "Text Files (*.txt)")
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("=" * 50 + "\n         智能裂隙检测系统 - 批量分析汇总报告         \n" + "=" * 50 + "\n\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"处理批次: 共汇总 {len(selected_items)} 个文件\n\n")

                total_cracks = 0
                for idx, thumb in enumerate(selected_items, 1):
                    data = self.history_cache.get(thumb.file_path, {})
                    c_count = data.get('count', 0)
                    total_cracks += c_count

                    f.write(f"【记录 {idx}】 ({'视频' if data.get('type') == 'video' else '图片'})\n")
                    f.write(f"- 文件路径: {thumb.file_path}\n")
                    f.write(f"- 裂隙数量: {c_count} 处\n")
                    if data.get('type') == 'image':
                        f.write(f"- 检测确信度: {data.get('conf_text', '未知')}\n")
                        f.write(f"- 算法耗时: {data.get('time_ms', 0)} ms\n")
                    f.write("-" * 30 + "\n")

                f.write(f"\n[总体结论]\n本次批量选中的文件中，共计检测到 {total_cracks} 处特征。\n")

            QMessageBox.information(self, "批量导出成功", f"数据已汇总并保存至：\n{save_path}")
            for item in selected_items: item.check_btn.setChecked(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CrackDetectionApp()
    window.show()
    sys.exit(app.exec())
