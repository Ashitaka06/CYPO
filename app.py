import sys
from os.path import basename
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QProgressBar, QTextEdit, QComboBox
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import pyqtSlot
from Process import Process

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.processor = None

    def initUI(self):
        self.setWindowTitle("PFE - CYPO")
        self.setGeometry(142, 142, 1300, 800)
        self.setStyleSheet("QPushButton { font-size: 30px; }")
        
        self.setWindowIcon(QIcon('./config/icon.ico'))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        logo_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_pixmap = QPixmap("./config/logo.png")
        logo_label.setPixmap(logo_pixmap)
        logo_label.setScaledContents(True)
        logo_label.setFixedSize(600, 300)  # Vous pouvez ajuster ces valeurs selon vos besoins
        logo_layout.addStretch()
        logo_layout.addWidget(logo_label)
        logo_layout.addStretch()
        main_layout.addLayout(logo_layout)

        select_pdf_button = QPushButton("Sélectionner un fichier PDF")
        select_pdf_button.clicked.connect(self.select_pdf_file)
        main_layout.addWidget(select_pdf_button)

        self.page_range_input = QLineEdit()
        self.page_range_input.setPlaceholderText("Entrer la plage de pages à traiter (ex: 1-5, 8, 10-11)")
        self.page_range_input.textChanged.connect(self.update_cost_estimate)
        main_layout.addWidget(self.page_range_input)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Google Vision OCR + GPT-3.5 Turbo", "Google Vision OCR + GPT-4 Turbo", "Google Vision OCR + GPT-Perso"])
        main_layout.addWidget(self.mode_combo)

        process_pdf_button = QPushButton("Convertir en JSON")
        process_pdf_button.clicked.connect(self.process_pdf)
        main_layout.addWidget(process_pdf_button)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setFont(QFont("Courier New", 12))
        main_layout.addWidget(self.log_text_edit)

        self.cost_label = QLabel("Coût estimé : 0 €")
        main_layout.addWidget(self.cost_label)

        self.mode_combo.currentTextChanged.connect(self.update_cost_estimate)
        
        font = QFont("Arial", 15)
        select_pdf_button.setFont(font)
        process_pdf_button.setFont(font)
        self.page_range_input.setFont(font)
        self.mode_combo.setFont(font)
        self.cost_label.setFont(font)
        
        footer_layout = QHBoxLayout()
        developers_label = QLabel("lamarqueni@cy-tech.fr & castanetfl@cy-tech.fr & labordetho@cy-tech.fr | 2024")
        footer_layout.addStretch()
        footer_layout.addWidget(developers_label)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)

    def update_cost_estimate(self):
        page_range_input = self.page_range_input.text().strip()
        if page_range_input:
            try:
                page_ranges = self.parse_page_ranges(page_range_input)
                total_pages = sum(len(range(start, end + 1)) for start, end in page_ranges)
                mode = self.mode_combo.currentText()
                cost_per_page = {"Google Vision OCR + GPT-3.5 Turbo": 0.005, "Google Vision OCR + GPT-4 Turbo": 0.08, "Google Vision OCR + GPT-Perso": 0}[mode]
                total_cost = total_pages * cost_per_page
                self.cost_label.setText(f"Coût estimé : {total_cost:.2f} €")
            except ValueError as ve:
                self.cost_label.setText("Coût estimé : 0 €")
            except Exception as e:
                self.log_text_edit.append(f"Erreur inattendue lors du calcul du coût : {str(e)}")
        else:
            self.cost_label.setText("Coût estimé : 0 €")

    @pyqtSlot()
    def select_pdf_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Sélectionner un fichier PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.file_path = file_path
            self.log_text_edit.append(f"{basename(self.file_path)} chargé.")  # Afficher le nom du fichier dans la console

    @pyqtSlot()
    def process_pdf(self):
        if not hasattr(self, "file_path"):
            self.log_text_edit.append("Erreur : Aucun fichier PDF sélectionné.")
            return

        page_ranges = self.parse_page_ranges(self.page_range_input.text())
        model_choice = self.mode_combo.currentText()
        self.processor = Process(self.file_path, page_ranges, model_choice)
        self.processor.update_progress.connect(self.progress_bar.setValue)
        self.processor.log_message.connect(self.log_text_edit.append)
        self.processor.start()

    def parse_page_ranges(self, input_str: str):
        ranges = []
        for part in input_str.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    ranges.append((start, end))
                except ValueError:
                    pass
            else:
                try:
                    page_num = int(part)
                    ranges.append((page_num, page_num))
                except ValueError:
                    pass
        return ranges

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())