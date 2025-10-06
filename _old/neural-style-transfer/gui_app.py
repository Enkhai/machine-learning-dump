import io
import torch
import sys
from functools import partial
from PIL import Image

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt

from helper import load_image
from train_style import train_image_style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingThread(QThread):
    image_ready = pyqtSignal(Image.Image, int)
    training_finished = pyqtSignal()

    def __init__(self, model, content, style, steps, show_every, alpha, beta, style_feature_weights):
        QThread.__init__(self)
        self.model = model
        self.content = content
        self.style = style
        self.steps = steps
        self.show_every = show_every
        self.alpha = alpha
        self.beta = beta
        self.style_feature_weights = style_feature_weights

    def run(self):
        content_tensor = load_image('', image=self.content).to(device)
        style_tensor = load_image('', image=self.style, shape=content_tensor.shape[-2:]).to(device)

        step = 0
        for image in train_image_style(self.model, content_tensor, style_tensor,
                                       steps=self.steps, show_every=self.show_every,
                                       alpha=self.alpha, beta=self.beta, style_weights=self.style_feature_weights):
            step += self.show_every
            self.image_ready.emit(image, step)

        self.training_finished.emit()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.content_image_label = QLabel()
        self.style_image_label = QLabel()

        self.steps_textbox = QLineEdit()
        self.update_every_textbox = QLineEdit()

        self.alpha_textbox = QLineEdit()
        self.beta_textbox = QLineEdit()

        self.style_feature1_weight = QLineEdit()
        self.style_feature2_weight = QLineEdit()
        self.style_feature3_weight = QLineEdit()
        self.style_feature4_weight = QLineEdit()
        self.style_feature5_weight = QLineEdit()

        self.loading_gif_movie = QMovie('app_images/loading.gif')
        self.loading_label = QLabel()

        self.error_label = QLabel()
        self.output_label = QLabel()

        self.threads = []
        self.model = torch.load('model.pth').to(device)

        self.init_window()

    def init_window(self):
        self.setWindowTitle("Image style transferor")
        self.setFixedSize(1200, 1000)
        self.setStyleSheet("QMainWindow {background-image: url('app_images/background.jpg'); "
                           "background-position: center;}")

        self.create_layout()

        self.show()

    def create_layout(self):
        grid_layout = QGridLayout()

        grid_layout.addWidget(self.top_widget(), 0, 0, Qt.AlignCenter)
        grid_layout.addWidget(self.mid_widget(), 1, 0, Qt.AlignCenter)
        grid_layout.addWidget(self.bot_widget(), 2, 0, Qt.AlignCenter)

        central_widget = QWidget()
        central_widget.setLayout(grid_layout)

        self.setCentralWidget(central_widget)

    def top_widget(self):

        h_box_layout = QHBoxLayout()

        content_layout = QVBoxLayout()
        self.content_image_label.setFixedSize(512, 384)
        self.content_image_label.setScaledContents(True)
        load_content_button = QPushButton("Load content image")
        load_content_button.clicked.connect(partial(self.load_image_file, self.content_image_label))

        content_layout.addWidget(load_content_button)
        content_layout.addWidget(self.content_image_label)

        content_widget = QWidget()
        content_widget.setLayout(content_layout)

        h_box_layout.addWidget(content_widget)

        style_layout = QVBoxLayout()
        self.style_image_label.setFixedSize(512, 384)
        self.style_image_label.setScaledContents(True)
        load_style_button = QPushButton("Load style image")
        load_style_button.clicked.connect(partial(self.load_image_file, self.style_image_label))

        style_layout.addWidget(load_style_button)
        style_layout.addWidget(self.style_image_label)

        style_widget = QWidget()
        style_widget.setLayout(style_layout)

        h_box_layout.addWidget(style_widget)

        widget = QWidget()
        widget.setLayout(h_box_layout)

        return widget

    def mid_widget(self):

        grid_layout = QGridLayout()

        steps_layout = QHBoxLayout()
        steps_label = QLabel("Steps")
        steps_label.setStyleSheet('color: white')
        steps_layout.addWidget(steps_label)
        self.steps_textbox.setText("2000")
        self.steps_textbox.setFixedSize(50, 20)
        steps_layout.addWidget(self.steps_textbox)
        steps_widget = QWidget()
        steps_widget.setLayout(steps_layout)
        grid_layout.addWidget(steps_widget, 0, 0)

        update_every_layout = QHBoxLayout()
        update_every_label = QLabel("Update target image every steps")
        update_every_label.setStyleSheet('color: white')
        update_every_layout.addWidget(update_every_label)
        self.update_every_textbox.setText("10")
        self.update_every_textbox.setFixedSize(50, 20)
        update_every_layout.addWidget(self.update_every_textbox)
        update_every_widget = QWidget()
        update_every_widget.setLayout(update_every_layout)
        grid_layout.addWidget(update_every_widget, 1, 0)

        alpha_layout = QHBoxLayout()
        alpha_label = QLabel("Content weight (alpha)")
        alpha_label.setStyleSheet('color: white')
        alpha_layout.addWidget(alpha_label)
        self.alpha_textbox.setText("1")
        self.alpha_textbox.setFixedSize(50, 20)
        alpha_layout.addWidget(self.alpha_textbox)
        alpha_widget = QWidget()
        alpha_widget.setLayout(alpha_layout)
        grid_layout.addWidget(alpha_widget, 0, 1)

        beta_layout = QHBoxLayout()
        beta_label = QLabel("Style weight (beta)")
        beta_label.setStyleSheet('color: white')
        beta_layout.addWidget(beta_label)
        self.beta_textbox.setText("1000000")
        self.beta_textbox.setFixedSize(50, 20)
        beta_layout.addWidget(self.beta_textbox)
        beta_widget = QWidget()
        beta_widget.setLayout(beta_layout)
        grid_layout.addWidget(beta_widget, 1, 1)

        feature1_layout = QHBoxLayout()
        feature1_label = QLabel("w1")
        feature1_label.setStyleSheet('color: white')
        feature1_layout.addWidget(feature1_label)
        self.style_feature1_weight.setText("1")
        self.style_feature1_weight.setFixedSize(50, 20)
        feature1_layout.addWidget(self.style_feature1_weight)
        feature1_widget = QWidget()
        feature1_widget.setLayout(feature1_layout)
        grid_layout.addWidget(feature1_widget, 0, 2)

        feature2_layout = QHBoxLayout()
        feature2_label = QLabel("w2")
        feature2_label.setStyleSheet('color: white')
        feature2_layout.addWidget(feature2_label)
        self.style_feature2_weight.setText("0.75")
        self.style_feature2_weight.setFixedSize(50, 20)
        feature2_layout.addWidget(self.style_feature2_weight)
        feature2_widget = QWidget()
        feature2_widget.setLayout(feature2_layout)
        grid_layout.addWidget(feature2_widget, 1, 2)

        feature3_layout = QHBoxLayout()
        feature3_label = QLabel("w3")
        feature3_label.setStyleSheet('color: white')
        feature3_layout.addWidget(feature3_label)
        self.style_feature3_weight.setText("0.2")
        self.style_feature3_weight.setFixedSize(50, 20)
        feature3_layout.addWidget(self.style_feature3_weight)
        feature3_widget = QWidget()
        feature3_widget.setLayout(feature3_layout)
        grid_layout.addWidget(feature3_widget, 2, 2)

        feature4_layout = QHBoxLayout()
        feature4_label = QLabel("w4")
        feature4_label.setStyleSheet('color: white')
        feature4_layout.addWidget(feature4_label)
        self.style_feature4_weight.setText("0.2")
        self.style_feature4_weight.setFixedSize(50, 20)
        feature4_layout.addWidget(self.style_feature4_weight)
        feature4_widget = QWidget()
        feature4_widget.setLayout(feature4_layout)
        grid_layout.addWidget(feature4_widget, 3, 2)

        feature5_layout = QHBoxLayout()
        feature5_label = QLabel("w5")
        feature5_label.setStyleSheet('color: white')
        feature5_layout.addWidget(feature5_label)
        self.style_feature5_weight.setText("0.2")
        self.style_feature5_weight.setFixedSize(50, 20)
        feature5_layout.addWidget(self.style_feature5_weight)
        feature5_widget = QWidget()
        feature5_widget.setLayout(feature5_layout)
        grid_layout.addWidget(feature5_widget, 4, 2)

        style_transfer_button = QPushButton("Transfer image style")
        style_transfer_button.clicked.connect(self.request_style_transfer)
        grid_layout.addWidget(style_transfer_button, 4, 1)

        widget = QWidget()
        widget.setLayout(grid_layout)

        return widget

    def bot_widget(self):

        h_box_layout = QVBoxLayout()

        self.loading_gif_movie.setScaledSize(self.loading_label.size())
        self.loading_gif_movie.start()
        self.loading_label.setFixedSize(30, 30)
        self.loading_label.setScaledContents(True)
        h_box_layout.addWidget(self.loading_label)
        h_box_layout.setAlignment(self.loading_label, Qt.AlignCenter)

        self.error_label.setStyleSheet('color: white')
        h_box_layout.addWidget(self.error_label)
        h_box_layout.setAlignment(self.error_label, Qt.AlignCenter)

        self.output_label.setFixedSize(512, 384)
        self.output_label.setScaledContents(True)
        h_box_layout.addWidget(self.output_label)
        h_box_layout.setAlignment(self.output_label, Qt.AlignCenter)

        widget = QWidget()
        widget.setLayout(h_box_layout)

        return widget

    def load_image_file(self, label):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open image file', filter="Image files (*.jpg; *.png; *.jpeg)")
        label.setPixmap(QPixmap(file_name))

    def request_style_transfer(self):
        self.error_label.setText("")

        if self.content_image_label.pixmap() is None:
            self.error_label.setText("Please set a content image")
            return

        if self.style_image_label.pixmap() is None:
            self.error_label.setText("Please set a style image")
            return

        try:
            steps = int(self.steps_textbox.text())
            show_every = int(self.update_every_textbox.text())

            alpha = float(self.alpha_textbox.text())
            beta = float(self.beta_textbox.text())

            style_feature_weights = {}
            style_feature_weights['conv1_1'] = float(self.style_feature1_weight.text())
            style_feature_weights['conv2_1'] = float(self.style_feature2_weight.text())
            style_feature_weights['conv3_1'] = float(self.style_feature3_weight.text())
            style_feature_weights['conv4_1'] = float(self.style_feature4_weight.text())
            style_feature_weights['conv5_1'] = float(self.style_feature5_weight.text())
        except ValueError:
            self.error_label.setText("Something went wrong... Please check your parameters!")
            return

        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)

        content = QImage(self.content_image_label.pixmap())
        content.save(buffer, "PNG")
        content = Image.open(io.BytesIO(buffer.data()))

        buffer.close()

        buffer.open(QBuffer.ReadWrite)

        style = QImage(self.style_image_label.pixmap())
        style.save(buffer, "PNG")
        style = Image.open(io.BytesIO(buffer.data()))

        self.loading_label.setMovie(self.loading_gif_movie)

        training_thread = TrainingThread(self.model, content, style, steps, show_every, alpha, beta,
                                         style_feature_weights)
        training_thread.image_ready.connect(self.handle_image_ready)
        training_thread.training_finished.connect(self.handle_training_finished)
        self.threads.append(training_thread)
        training_thread.start()

    def handle_image_ready(self, image, step):
        image.save("output.jpg")
        self.error_label.setText("Showing image for step %d out of %s" % (step, self.steps_textbox.text()))
        self.output_label.setPixmap(QPixmap("output.jpg"))

    def handle_training_finished(self):
        self.error_label.setText("Style transfer has been completed")
        self.loading_label.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec())
