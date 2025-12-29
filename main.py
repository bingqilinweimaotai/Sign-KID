from LoginUI import *
from InterfaceUI import *
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import QUrl
from QD import main
from contextlib import contextmanager
import sys
import os
import psycopg2
import tkinter as tk
import threading

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_LoginWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QtCore.Qt.gray)
        self.ui.frame.setGraphicsEffect(self.shadow)
        self.ui.pushButton_Login.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_Register.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))

        self.ui.pushButton_L_sure.clicked.connect(self.Login_in)
        self.ui.pushButton_R_sure.clicked.connect(self.Register_in)
        self.show()

    def Login_in(self):
        account = self.ui.lineEdit_L_account.text()
        password = self.ui.lineEdit_L_password.text()
        account_list = []
        password_list = []
        conn = psycopg2.connect(database="Data_MY", user="postgres", password="123456789", host="127.0.0.1", port="5432")
        cur = conn.cursor()
        cur.execute("select * from users")
        rows = cur.fetchall()
        for row in rows:
            account_list.append(row[0])
            password_list.append(row[1])
        conn.commit()
        conn.close()
        for i in range(len(account_list)):
            if len(account) == 0 or len(password) == 0:
                self.ui.stackedWidget.setCurrentIndex(2)
            if account == account_list[i] and password == password_list[i]:
                self.win = InterfaceWindow()
                self.close()
            else:
                self.ui.stackedWidget.setCurrentIndex(1)

    def Register_in(self):
        account = self.ui.lineEdit_R_account.text()
        password = self.ui.lineEdit_R_password_1.text()
        re_password = self.ui.lineEdit_R_password_2.text()

        conn = psycopg2.connect(database="Data_MY", user="postgres", password="123456789", host="127.0.0.1",
                                port="5432")
        cur = conn.cursor()
        cur.execute(f"insert into users values('{account}', '{password}')")
        conn.commit()
        conn.close()

class InterfaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_InferfaceWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_production.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_train.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_production_video.clicked.connect(self.Produce_video)
        self.ui.pushButton_t_train.clicked.connect(self.start_train_thread)
        self.ui.pushButton_t_test.clicked.connect(self.Test)
        self.show()

    def Train(self):
        # 在这个方法中执行main函数
        main(mode="train", config_path="C:\Users\YCY\Downloads\代码包\代码包\Sign-KID\Configs\Base.yaml", ckpt=None, gpu_id="0")

    def start_train_thread(self):
        # 创建一个线程来执行Train方法
        train_thread = threading.Thread(target=self.Train)
        train_thread.start()

    def Test(self):
        main()

    def Produce_video(self):
        gloss = self.ui.lineEdit_GLOSS.text()
        if len(gloss) == 0:
            return
        else:
            gloss_list = []
            file_video_list = []
            # 创建视频播放器实例
            conn = psycopg2.connect(database="Data_MY", user="postgres", password="123456789", host="127.0.0.1",
                                    port="5432")
            cur = conn.cursor()
            cur.execute("select * from gloss_video")
            rows = cur.fetchall()
            for row in rows:
                gloss_list.append(row[0])
                file_video_list.append(row[1])
            conn.commit()
            conn.close()

            for i in range(len(gloss_list)):
                if gloss == gloss_list[i]:
                    file_video = file_video_list[i]


            self.player = QMediaPlayer()

            # 把视频播放器放入对应组件(PyQt5.QtMultimediaWidgets.QVideoWidget)
            self.player.setVideoOutput(self.ui.widget)

            # 获取视频的地址，这里需要Qurl类型getOpenFileUrl
            media_path = QUrl.fromLocalFile(file_video)

            self.player.setMedia(QMediaContent(media_path))

            # 播放视频
            self.player.play()

            self.player.stateChanged.connect(self.handle_state_changed)

    def handle_state_changed(self, state):
        # 当播放器状态改变时检查是否到达视频结束
        if state == QMediaPlayer.StoppedState:
            # 视频播放结束后重新开始播放
            self.player.setPosition(0)  # 将播放位置重置为开始
            self.player.play()  # 重新开始播放


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = InterfaceWindow()
    win.setWindowTitle('Sign-KID')
    sys.exit(app.exec_())