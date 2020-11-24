#must import image !!!

from PyQt5 import QtCore, QtWidgets,QtGui
import cv2


   

class Ui_MainWindow(object):
    def update(self):
        self.UiComponenets()

    def UiComponenets(self):
        self.rep_count.setObjectName(current_rep)
        #updating list
        for x in workouts:
                #this needs work to be able to add workouts to the list sequence index is not an int slice, or instance
                item = workouts[x]
                self.workout_list.addItem(item)

        

       #don't need video capture, just when new frames passed in (every update)  
        
        cap = cv2.VideoCapture('')
        while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                        break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pix = QPixmap.fromImage(img)
                pix = QPixmap.fromImage()
                self.ui.label_7.setPixmap(pix)   

                self.ui.frame = pix 

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        cap.release()

        

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1713, 883)
        MainWindow.setStyleSheet("QWidget\n"
        "{\n"
"    background-color: rgb(26, 28, 35);\n"
"border-radius: 10px;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.dropShadowFrame = QtWidgets.QFrame(self.centralwidget)
        self.dropShadowFrame.setGeometry(QtCore.QRect(30, 30, 1651, 831))
        self.dropShadowFrame.setStyleSheet("QFrame{\n"
"    background-color: rgb(108, 111, 147);\n"
"    border-radius: 10px;\n"
"}")
        self.dropShadowFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.dropShadowFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.dropShadowFrame.setObjectName("dropShadowFrame")
        self.rep_count_label = QtWidgets.QLabel(self.dropShadowFrame)
        self.rep_count_label.setGeometry(QtCore.QRect(1260, 100, 321, 71))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(40)
        self.rep_count_label.setFont(font)
        self.rep_count_label.setStyleSheet("color: rgb(46, 48, 62);")
        self.rep_count_label.setAlignment(QtCore.Qt.AlignCenter)
        self.rep_count_label.setObjectName("rep_count_label")
        self.rep_count = QtWidgets.QLabel(self.dropShadowFrame)
        self.rep_count.setGeometry(QtCore.QRect(1320, 180, 181, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(25)
        self.rep_count.setFont(font)
        self.rep_count.setStyleSheet("color: rgb(46, 48, 62);")
        self.rep_count.setAlignment(QtCore.Qt.AlignCenter)
        self.rep_count.setObjectName("rep_count")
        self.workout_list = QtWidgets.QListWidget(self.dropShadowFrame)
        self.workout_list.setGeometry(QtCore.QRect(1270, 290, 301, 511))
        self.workout_list.setStyleSheet("QListWidget{\n"
"    \n"
"    \n"
"    \n"
"    color: rgb(46, 48, 62);\n"
"}")
        self.workout_list.setObjectName("workout_list")
        self.label = QtWidgets.QLabel(self.dropShadowFrame)
        self.label.setGeometry(QtCore.QRect(20, 0, 111, 81))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../../../../Pictures/f2aback.jpg"))
        self.label.setObjectName("label")
        self.frame = QtWidgets.QFrame(self.dropShadowFrame)
        self.frame.setGeometry(QtCore.QRect(30, 100, 1211, 711))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.rep_count_label.setText(_translate("MainWindow", "<strong>Rep Count"))
        self.rep_count.setText(_translate("MainWindow", "1"))

                
        


        

  

workouts = ['workout1','workout2','workout3','workout4']
current_rep = 1
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    while True:
            ui.update()
            current_rep += 1
    sys.exit(app.exec_())
