from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton
)

from .config import LEFT_BAR_WIDTH, FONT_FAMILY


class FaceListWidget(QFrame):
    face_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(LEFT_BAR_WIDTH)
        self.setStyleSheet("background-color: #1A1A3A; border-radius: 8px;")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        title_label = QLabel("人脸ID列表")
        title_label.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        title_label.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(title_label)

        self.face_list = QListWidget()
        self.face_list.setFont(QFont(FONT_FAMILY, 12))
        self.face_list.setStyleSheet("""
            QListWidget {
                background-color: #0F0F25;
                border: none;
                border-radius: 6px;
                padding: 8px;
                color: #FFFFFF;
            }
            QListWidget::item {
                background-color: #252545;
                border-radius: 6px;
                padding: 12px;
                margin-bottom: 6px;
            }
            QListWidget::item:selected {
                background-color: #353560;
                border: 1px solid #6666CC;
            }
            QListWidget::item:hover {
                background-color: #2A2A50;
            }
        """)
        self.face_list.itemClicked.connect(self.on_face_clicked)
        layout.addWidget(self.face_list)

        hint_label = QLabel("点击左侧人脸ID查看分析记录")
        hint_label.setFont(QFont(FONT_FAMILY, 10))
        hint_label.setStyleSheet("color: #888888;")
        hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint_label)

    def load_face_ids(self, face_ids):
        self.face_list.clear()
        for face_id in face_ids:
            item = QListWidgetItem(f"👤 {face_id}")
            self.face_list.addItem(item)

    def on_face_clicked(self, item):
        face_id = item.text().replace("👤 ", "")
        self.face_selected.emit(face_id)
