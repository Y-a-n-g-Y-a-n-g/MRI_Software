# -*- coding: utf-8 -*-
"""
@Time ： 12/24/2022 10:33 PM
@Auth ： YY
@File ：测试pyqt5表格.py
@IDE ：PyCharm
@state:
@Function：
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QTableView, QApplication, QAction, QMessageBox


class MetaTableView(QTableView):

    def __init__(self, parent=None):
        super(MetaTableView, self).__init__(parent)
        self.resize(800, 600)
        self.myModel = QStandardItemModel()  # model
        self.initHeader()  # 初始化表头
        self.setModel(self.myModel)
        self.initData()  # 初始化模拟数据



    def initHeader(self):
        self.myModel.setHorizontalHeaderItem(0, QStandardItem("Tags"))
        self.myModel.setHorizontalHeaderItem(1, QStandardItem("value"))
        self.myModel.setHorizontalHeaderItem(2, QStandardItem("VR"))
        self.myModel.setHorizontalHeaderItem(3, QStandardItem("Name"))
        self.myModel.setHorizontalHeaderItem(4, QStandardItem("State"))
        self.myModel.setHorizontalHeaderItem(5, QStandardItem("中文备注-will delete later"))
    def initData(self):
        for row in range(100):
            for col in range(5):
                self.myModel.setItem(
                    row, col, QStandardItem("row: {row},col: {col}".format(row=row + 1, col=col + 1)))


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("Dicom file MetaData")
    w = MetaTableView()
    w.show()
    sys.exit(app.exec_())