import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    img_array = tf.keras.applications.inception_resnet_v2.preprocess_input(img_array)
    return img_array

def predict_image(model, image_path, labels):
    img_array = load_and_preprocess_image(image_path, target_size=(224, 224))
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return labels[predicted_class[0]]

# Main Window Class
class ImageClassifier(QMainWindow):
    def __init__(self, model, labels):
        super().__init__()
        self.model = model
        self.labels = labels
        self.initUI()
    
    def initUI(self):
        self.setAcceptDrops(True)

        # Create central widget
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        # Layout
        layout = QVBoxLayout(centralWidget)

        # Image label
        self.imageLabel = QLabel('Drag and Drop Image Here', self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.imageLabel)

        # Prediction label
        self.predictionLabel = QLabel('', self)
        self.predictionLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.predictionLabel)

        # Set geometry
        self.setGeometry(300, 300, 400, 400)
        self.setWindowTitle('Image Classifier')
        self.show()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].scheme() == 'file':
            filepath = urls[0].path()
            # Display "Predicting..." text
            self.predictionLabel.setText('Predicting...')
            QApplication.processEvents()  # Update UI immediately

            # Load and display the image
            pixmap = QPixmap(filepath)
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), Qt.KeepAspectRatio))

            # Predict and display the result
            prediction = predict_image(self.model, filepath, self.labels)
            self.predictionLabel.setText(f'Predicted label: {prediction}')


def main(model, labels):
    app = QApplication(sys.argv)
    ex = ImageClassifier(model, labels)
    sys.exit(app.exec_())

if __name__ == '__main__':
    # Load model and labels
    tf.config.set_visible_devices([], 'GPU')
    model = tf.keras.models.load_model('./birds_pruned.keras')
    with open('./labels.txt', 'r') as file:
        labels = [line.strip() for line in file]
    
    main(model, labels)
