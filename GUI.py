import os
import sys
import subprocess
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QWidget, QProgressBar, QMessageBox, QFileDialog, QMenuBar, QAction, QGraphicsView, QGraphicsScene, QLineEdit, QLabel, QRadioButton, QButtonGroup, QSizePolicy, QInputDialog, QDialog, QSlider, QToolTip)
from PyQt5.QtGui import QPixmap, QIcon, QFont, QImageReader
from PyQt5.QtCore import QTimer, QEvent, QThread, Qt, QProcess, QProcessEnvironment, QDir
from skimage.transform import rescale
from skimage.io import imread, imsave


# AstrocytesAnalysis subprocess
class WorkerThread(QThread):
    def __init__(self, program_name):
        super().__init__()
        self.program_name = program_name

    def run(self):
        self.process = subprocess.Popen(["python3", self.program_name])

    def change_program_name(self, new_program_name):
        self.program_name = new_program_name

    def stop(self):
        if hasattr(self, "process"):
            self.process.terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Initialize CZI file and output directory
        self.czi_file = ""
        self.output_dir = ""

        # Create a Graphics View widget
        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setMinimumSize(800, 600)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)

        # Create a status edit widget
        self.status_edit = QTextEdit(self)
        self.status_edit.setReadOnly(True)

        # Create a progress bar widget
        self.progress_bar = QProgressBar(self)

        # Create Start, Stop, and Toggle buttons
        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
        self.shockwave_button = QPushButton('Shockwave', self)
        self.rgc_button = QPushButton('Ablation', self)
        self.next_button = QPushButton('Next', self)
        self.prev_button = QPushButton('Prev', self)

        # Create image number input and label
        self.image_label = QLabel('Image Number', self)
        self.image_input = QLineEdit(self)

        # Set the buttons to be checkable
        self.shockwave_button.setCheckable(True)
        self.rgc_button.setCheckable(True)

        # Create a button group and add the buttons to it
        self.image_mode_group = QButtonGroup(self)
        self.image_mode_group.addButton(self.shockwave_button, 0)
        self.image_mode_group.addButton(self.rgc_button, 1)

        # Set up layouts
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.status_edit)
        controls_layout.addWidget(self.shockwave_button)
        controls_layout.addWidget(self.rgc_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.image_label)
        controls_layout.addWidget(self.image_input)

        main_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.graphics_view, 4)

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        # Set size policy of the graphics view to expand
        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Connect the button signals with the slots
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.image_mode_group.buttonClicked[int].connect(self.script_mode_change)
        self.next_button.clicked.connect(self.next_image)
        self.prev_button.clicked.connect(self.prev_image)
        self.image_input.returnPressed.connect(self.select_image)

        # Set the window title
        self.setWindowTitle("Shockwave/Ablation Analysis")

        # Set the status bar
        self.statusBar().showMessage('Ready')

        # Initialize image index
        self.image_index = 0

        # Initialize image mode to raw
        self.shockwave_button.setChecked(True)

        # Create worker thread
        self.worker_thread = WorkerThread("shockwave.py")

        # Create timer for reading progress
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.read_progress)

        # Set up the menu bar
        self.setup_menu()

        # Set the style of the GUI
        self.set_style()

        # Create CZI conversion process
        self.czi_conversion_process = QProcess(self)
        self.czi_conversion_process.setProcessChannelMode(QProcess.MergedChannels)
        self.czi_conversion_process.readyReadStandardOutput.connect(self.read_czi_conversion_output)

    def set_style(self, theme="dark"):
        if theme == "dark":
            style = """
            QMainWindow {
                background-color: #2C2C2C;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #404040;
                color: #FFFFFF;
                border: 2px solid #FFFFFF;
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #606060;
            }
            QTextEdit {
                background-color: #404040;
                color: #FFFFFF;
                border: 2px solid #FFFFFF;
            }
            QLineEdit {
                background-color: #404040;
                color: #FFFFFF;
                border: 2px solid #FFFFFF;
            }
            QLabel {
                color: #FFFFFF;
            }
            QMenuBar {
                background-color: #404040;
                color: #FFFFFF;
            }
            QMenuBar:item {
                background-color: #404040;
                color: #FFFFFF;
            }
            QMenuBar:item:selected {
                background-color: #505050;
            }
            QProgressBar {
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #404040;
            }
            QMessageBox {
                background-color: #404040;
                color: #FFFFFF;
                border: 2px solid #FFFFFF;
            }
            QDialog {
                background-color: #2C2C2C;
                color: #FFFFFF;
            }
            """
        else:  # light theme
            style = """
            QMainWindow {
                background-color: #EDEDED;
                color: #000000;
            }
            QPushButton {
                background-color: #DDDDDD;
                color: #000000;
                border: 2px solid #000000;
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #CCCCCC;
            }
            QPushButton:pressed {
                background-color: #BBBBBB;
            }
            QTextEdit {
                background-color: #DDDDDD;
                color: #000000;
                border: 2px solid #000000;
            }
            QLineEdit {
                background-color: #DDDDDD;
                color: #000000;
                border: 2px solid #000000;
            }
            QLabel {
                color: #000000;
            }
            QMenuBar {
                background-color: #DDDDDD;
                color: #000000;
            }
            QMenuBar:item {
                background-color: #DDDDDD;
                color: #000000;
            }
            QMenuBar:item:selected {
                background-color: #CCCCCC;
            }
            QProgressBar {
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #DDDDDD;
            }
            QMessageBox {
                background-color: #EDEDED;
                color: #000000;
                border: 2px solid #000000;
            }
            QDialog {
                background-color: #EDEDED;
                color: #000000;
            }
            """
        self.setStyleSheet(style)

    def setup_menu(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')
        config_file = open("config.json", "r")
        config = json.load(config_file)

        # load_nuclei_model_action = QAction('Load Nuclei Model', self)
        # load_nuclei_model_action.triggered.connect(self.load_nuclei_model)
        # file_menu.addAction(load_nuclei_model_action)
        # self.status_edit.append(f"Current Nuclei Model: {config['nuclei_model_location']}")

        # load_cyto_model_action = QAction('Load Cyto Model', self)
        # load_cyto_model_action.triggered.connect(self.load_cyto_model)
        # file_menu.addAction(load_cyto_model_action)
        # self.status_edit.append(f"Current Cyto Model: {config['cyto_model_location']}")

        # load_pre_dir_action = QAction('Load Pre Directory', self)
        # load_pre_dir_action.triggered.connect(self.load_pre_dir)
        # file_menu.addAction(load_pre_dir_action)
        # self.status_edit.append(f"Current Pre Directory: {config['pre_directory_location']}")

        load_post_dir_action = QAction('Load Folder Location', self)
        load_post_dir_action.triggered.connect(self.load_post_dir)
        file_menu.addAction(load_post_dir_action)
        self.status_edit.append(f"Current Folder Location: {config['folder_location']}")

        load_experiment_name_action = QAction('Current Experiment Name', self)
        load_experiment_name_action.triggered.connect(self.load_experiment_name)
        file_menu.addAction(load_experiment_name_action)
        self.status_edit.append(f"Current Experiment Name: {config['experiment_name']}")

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # CZI Menu
        czi_menu = menubar.addMenu('&CZI')

        czi_to_tiff_action = QAction('Convert CZI to TIFF with timestamps', self)
        czi_to_tiff_action.triggered.connect(self.convert_czi_to_tiff)
        czi_menu.addAction(czi_to_tiff_action)

        # Upscale Menu
        upscale_menu = menubar.addMenu('&Upscale')

        upscale_images_action = QAction('Upscale Images in Directory', self)
        upscale_images_action.triggered.connect(self.upscale_images)
        upscale_menu.addAction(upscale_images_action)

        # Theme Menu
        theme_menu = menubar.addMenu('&Theme')

        light_mode_action = QAction('Light Mode', self)
        light_mode_action.triggered.connect(lambda: self.set_style("light"))  # Use lambda to pass argument
        theme_menu.addAction(light_mode_action)

        dark_mode_action = QAction('Dark Mode', self)
        dark_mode_action.triggered.connect(lambda: self.set_style("dark"))  # Use lambda to pass argument
        theme_menu.addAction(dark_mode_action)

        # Help
        help_action = QAction('Help', self)
        help_action.triggered.connect(self.open_help)
        menubar.addAction(help_action)

    def start_analysis(self):
        # check if the experiment already exists
        config_file = open("config.json", "r")
        config = json.load(config_file)
        config_file.close()
        if os.path.exists(config["experiment_name"]):
            warn_msg = "This experiment already exists, are you sure you want to overwrite it?"
            reply = QMessageBox.question(self, 'Warning', warn_msg, QMessageBox.Yes, QMessageBox.No)

            if reply == QMessageBox.No:
                return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_edit.append("Starting analysis...")

        self.worker_thread.change_program_name(config["current_script_name"])
        self.worker_thread.start()

        self.progress_timer.start(100)

    def stop_analysis(self):
        self.worker_thread.stop()

        self.progress_timer.stop()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.status_edit.append("Analysis stopped.")

    def read_progress(self):
        if self.worker_thread.process.poll() is not None:
            self.progress_timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_edit.append("Analysis finished.")
        elif os.path.exists('progress.txt'):
            with open('progress.txt', 'r') as f:
                contents = f.read().strip()
                if contents:
                    try:
                        current, total = map(int, contents.split(','))
                        self.progress_bar.setValue(int(100 * current / total))
                        if current == total:
                            self.progress_timer.stop()
                    except ValueError:
                        print("Error: unable to parse progress information.")

    def closeEvent(self, event: QEvent) -> None:
        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.worker_thread.isRunning():
                self.stop_analysis()
            event.accept()
        else:
            event.ignore()

    def load_nuclei_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Nuclei Cellpose Model")
        if file_name:
            self.update_config("nuclei_model_location", file_name)
            config_file = open("config.json", "r")
            config = json.load(config_file)
            config_file.close()
            self.status_edit.append(f"New Nuclei Model: {config['nuclei_model_location']}")

    def load_cyto_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Cytoplasm Cellpose Model")
        if file_name:
            self.update_config("cyto_model_location", file_name)
            config_file = open("config.json", "r")
            config = json.load(config_file)
            config_file.close()
            self.status_edit.append(f"New Cyto Model: {config['cyto_model_location']}")

    def load_pre_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Load Pre Directory")
        if dir_name:
            self.update_config("pre_directory_location", dir_name)
            config_file = open("config.json", "r")
            config = json.load(config_file)
            config_file.close()
            self.status_edit.append(f"New Pre Directory: {config['pre_directory_location']}")

    def load_post_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Load Post Directory")
        if dir_name:
            self.update_config("post_directory_location", dir_name)
            config_file = open("config.json", "r")
            config = json.load(config_file)
            config_file.close()
            self.status_edit.append(f"New Post Directory: {config['post_directory_location']}")

    def load_experiment_name(self):
        experiment_name, ok = QInputDialog.getText(self, "Rename Experiment",
                                    "New Experiment Name:", QLineEdit.Normal, 
                                    QDir().home().dirName())
        if ok and experiment_name:
            experiment_name = str(experiment_name)
            self.update_config("experiment_name", experiment_name)
            config_file = open("config.json", "r")
            config = json.load(config_file)
            self.status_edit.append(f"New Experiment Name: {config['experiment_name']}")
            config_file.close()

    def update_config(self, key, value):
        with open("config.json", "r") as f:
            config = json.load(f)
        config[key] = value
        with open("config.json", "w") as f:
            json.dump(config, f)

    def next_image(self):
        self.image_index += 1
        self.load_image()

    def prev_image(self):
        self.image_index = max(0, self.image_index - 1)
        self.load_image()

    def select_image(self):
        try:
            self.image_index = int(self.image_input.text())
            self.load_image()
        except ValueError:
            message_box = QMessageBox(self)
            message_box.setIcon(QMessageBox.Warning)
            message_box.setWindowTitle("Invalid input")
            message_box.setText("Please enter a valid image number.")
            message_box.setStyleSheet("QMessageBox { background-color: #404040; color: #FFFFFF; border: 2px solid #FFFFFF; }")
            message_box.exec_()

    # def image_mode_changed(self, id):
    #     self.image_mode = "raw" if id == 0 else "normalized"
    #     self.load_image()

    def script_mode_change(self, id):
        self.update_config("current_script_name", "shockwave.py" if id == 0 else "rgc_ablation.py")

    def image_mode_change(self):
        self.image_mode = "raw" if id == 0 else "normalized"
        self.load_image()

    def slider_value_changed(self):
        self.image_mode = "raw" if self.image_mode_slider.value() == 0 else "normalized"
        self.load_image()

    def load_image(self):
        current_dir = os.getcwd()
        config_file = open("config.json", "r")
        config = json.load(config_file)
        config_file.close()

        self.image_mode = "normalized"

        if self.image_mode == "raw":
            file_name = os.path.join(os.path.join(current_dir, config["experiment_name"]), f"plot_raw{self.image_index}.png")
        else:
            file_name = os.path.join(os.path.join(current_dir, config["experiment_name"]), f"statistics_roi_{self.image_index}.png")
        print(f"Trying to load: {file_name}")
        if os.path.exists(file_name):
            pixmap = QPixmap(file_name)
            self.graphics_scene.clear()
            self.graphics_scene.addPixmap(pixmap)
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            self.image_index = 0
            file_name = os.path.join(os.path.join(current_dir, config["experiment_name"]), "plot0.png")
            print("Reached end of image list, attempting to load first image")
            if os.path.exists(file_name):
                pixmap = QPixmap(file_name)
                self.graphics_scene.clear()
                self.graphics_scene.addPixmap(pixmap)
                self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            else:
                message_box = QMessageBox(self)
                message_box.setIcon(QMessageBox.Warning)
                message_box.setWindowTitle("Image not found")
                message_box.setText("The requested image does not exist.")
                message_box.setStyleSheet("QMessageBox { background-color: #404040; color: #FFFFFF; border: 2px solid #FFFFFF; }")
                message_box.exec_()

    def convert_czi_to_tiff(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Convert CZI to TIFF with Timestamps")

        czi_file_input = QLineEdit(self.czi_file, dialog)
        output_dir_input = QLineEdit(self.output_dir, dialog)

        czi_file_input.setStyleSheet("background-color: #404040; color: #FFFFFF; border: 2px solid #FFFFFF;")
        output_dir_input.setStyleSheet("background-color: #404040; color: #FFFFFF; border: 2px solid #FFFFFF;")

        czi_file_label = QLabel("CZI File:", dialog)
        output_dir_label = QLabel("Output Directory:", dialog)

        czi_file_label.setStyleSheet("color: #FFFFFF;")
        output_dir_label.setStyleSheet("color: #FFFFFF;")

        select_czi_file_button = QPushButton("Select", dialog)
        select_output_dir_button = QPushButton("Select", dialog)

        select_czi_file_button.clicked.connect(lambda: czi_file_input.setText(QFileDialog.getOpenFileName(dialog, "Select CZI File")[0]))
        select_output_dir_button.clicked.connect(lambda: output_dir_input.setText(QFileDialog.getExistingDirectory(dialog, "Select Output Directory")))

        convert_button = QPushButton("Convert", dialog)
        cancel_button = QPushButton("Cancel", dialog)

        convert_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        row3 = QHBoxLayout()

        row1.addWidget(czi_file_label)
        row1.addWidget(czi_file_input)
        row1.addWidget(select_czi_file_button)

        row2.addWidget(output_dir_label)
        row2.addWidget(output_dir_input)
        row2.addWidget(select_output_dir_button)

        row3.addWidget(convert_button)
        row3.addWidget(cancel_button)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.Accepted:
            self.czi_file = czi_file_input.text()
            self.output_dir = output_dir_input.text()
            self.start_czi_conversion()

    def start_czi_conversion(self):
        if not self.czi_file or not self.output_dir:
            return

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ConversionsScripts/CZI2TIFFwithTIMESTAMPS.py")
        self.czi_conversion_process.start("python", [script_path, self.czi_file, self.output_dir])

    def read_czi_conversion_output(self):
        output = self.czi_conversion_process.readAllStandardOutput().data().decode('utf-8').strip()
        self.status_edit.append(output)

    def upscale_images(self):
        scale_factor, ok = QInputDialog.getDouble(self, "Upscale Images", "Enter upscale factor:", 2.0, 1.0, 10.0, 2)
        if ok:
            # Get the directory of images to upscale
            dir_name = QFileDialog.getExistingDirectory(self, "Select Directory with Images to Upscale")
            if dir_name:
                for filename in os.listdir(dir_name):
                    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".tiff"):
                        image_path = os.path.join(dir_name, filename)
                        image = imread(image_path)
                        upscaled_image = rescale(image, scale_factor, anti_aliasing=True, multichannel=True)
                        imsave(image_path, (upscaled_image * 255).astype('uint8'))  # Overwrite the original image with the upscaled version
                        self.status_edit.append(f"Upscaled {filename}")

    def open_help(self):
        help_url = "https://github.com/not-availiable/AstrocytesImageAnalysis/"
        if sys.platform.startswith('darwin'):
            subprocess.Popen(['open', help_url])
        elif os.name == 'nt':
            subprocess.Popen(['cmd', '/c', 'start', help_url])
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', help_url])

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
