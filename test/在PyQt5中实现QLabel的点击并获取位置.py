# -*- coding: utf-8 -*-
"""
@Time ： 12/25/2022 8:58 PM
@Auth ： YY
@File ：在PyQt5中实现QLabel的点击并获取位置.py
@IDE ：PyCharm
@state:
@Function：
"""
import sys
from PyQt5 import QtWidgets, QtCore

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # 创建 QLabel 对象
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Click me")
        self.label.installEventFilter(self)  # 安装事件过滤器

        # 设置布局
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.label)

    def eventFilter(self, obj, event):
        if obj == self.label and event.type() == QtCore.QEvent.MouseButtonPress:
            # 点击事件发生时，获取点击的坐标
            x = event.pos().x()
            y = event.pos().y()
            print(f"Clicked at ({x}, {y})")
            return True  # 事件已被处理，不再传递
        return super().eventFilter(obj, event)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())
