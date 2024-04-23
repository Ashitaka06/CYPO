import os
from os.path import splitext, join
import re
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
from sklearn.cluster import DBSCAN
from PyQt5.QtCore import QThread, pyqtSignal
from google.cloud import vision
import io
from openai import OpenAI
import json

class Process(QThread):
    update_progress = pyqtSignal(int)  # Signal pour la mise à jour de la barre de progression
    log_message = pyqtSignal(str)  # Signal pour envoyer des logs à l'interface utilisateur

    def __init__(self, file_path, page_ranges, model_choice):
        super().__init__()
        self.file_path = file_path
        self.page_ranges = page_ranges
        self.model_choice = model_choice

    def run(self):
        try:
            doc = fitz.open(self.file_path)
        except Exception as e:
            self.log_message.emit(f"Erreur lors de l'ouverture du fichier PDF : {str(e)}")
            return

        # Extraire le nom de base du fichier sans l'extension
        base_name = splitext(os.path.basename(self.file_path))[0]
        preprocess_dir = join("1-preprocess", base_name)
        intersection_dir = join("2-intersection", base_name)
        ocr_dir = join("3-ocr", base_name)
        json_dir = join("4-json", base_name)
        os.makedirs(preprocess_dir, exist_ok=True)
        os.makedirs(intersection_dir, exist_ok=True)
        os.makedirs(ocr_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        total_pages = sum(end - start + 1 for start, end in self.page_ranges)
        processed_pages = 0

        for start, end in self.page_ranges:
            for page_num in range(start, end + 1):
                try:
                    page = doc.load_page(page_num - 1)
                    self.log_message.emit(f"Prétraitement de la page {page_num} en cours...")
                    processed_img = self.preprocess_image(page)
                    test_output_filename = join(preprocess_dir, f"{base_name}_page_{page_num}.png")
                    cv2.imwrite(test_output_filename, processed_img)
                    self.log_message.emit(f"Prétraitement de la page {page_num} terminé avec succès.")

                    # Détection des intersections
                    self.log_message.emit(f"Détection des intersections de la page {page_num} en cours...")
                    intersection_img, median_points = self.detect_intersections(test_output_filename)
                    intersection_output_filename = join(intersection_dir, f"{base_name}_page_{page_num}.png")
                    intersection_img.save(intersection_output_filename)
                    self.log_message.emit(f"Détection des intersections de la page {page_num} terminée avec succès.")
                    
                    # Application de l'OCR
                    self.log_message.emit(f"Application de l'OCR sur la page {page_num} en cours...")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./config/vision_config.json"
                    ocr_img, liste = self.apply_ocr(intersection_output_filename, median_points)
                    ocr_output_filename = join(ocr_dir, f"{base_name}_page_{page_num}.png")
                    ocr_img.save(ocr_output_filename)
                    self.log_message.emit(f"Application de l'OCR sur la page {page_num} terminée avec succès.")
                    
                    # Conversion en fichier JSON
                    self.log_message.emit(f"Conversion de la page {page_num} en JSON en cours...")
                    json_response = self.gpt2json(liste, self.model_choice)
                    json_output_filename = os.path.join(json_dir, f"{base_name}_page_{page_num}.json")
                    with open(json_output_filename, 'w') as json_file:
                        json.dump(json_response, json_file, indent=4)
                    self.log_message.emit(f"Conversion de la page {page_num} en JSON terminée avec succès.")
                    
                except Exception as e:
                    self.log_message.emit(f"Erreur lors du traitement de la page {page_num} : {str(e)}")

                processed_pages += 1
                progress_percentage = int((processed_pages / total_pages) * 100)
                self.update_progress.emit(progress_percentage)

        self.log_message.emit("Conversion du PDF terminée. Vous pouvez consulter les JSON dans le dossier ./4-json")

    def preprocess_image(self, page, zoom=10):
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = np.array(img)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
        img_clahe = clahe.apply(img_gray)
        _, img_thresh = cv2.threshold(img_clahe, 125, 255, cv2.THRESH_BINARY)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
        img_sharpened = cv2.filter2D(img_thresh, -1, sharpen_kernel)
        kernel = np.ones((1, 1), np.uint8)
        img_morph = cv2.morphologyEx(img_sharpened, cv2.MORPH_OPEN, kernel)
        return img_morph

    def charger_image_en_numpy(self, chemin_image):
        try:
            image = Image.open(chemin_image)
            image_gris = image.convert('L')
            tableau_numpy = np.array(image_gris)
            return tableau_numpy
        except Exception as e:
            print(f"Error in charger_image_en_numpy: {e}")

    def sequence_detector_line(self, lst, deb, fin, taille_seq_ligne=150):
        count_zeros = 0
        try:
            for element in lst[deb:fin]:
                if element == 0:
                    count_zeros += 1
                    if count_zeros == taille_seq_ligne:
                        return True
                else:
                    count_zeros = 0
            return False
        except IndexError as e:
            print(f"IndexError in sequence_detector_line: {e}")
            return False

    def moyenne_arrondie(self, liste):
        try:
            somme = sum(liste)
            moyenne = somme / len(liste)
            moyenne_arrondie = round(moyenne)
            return moyenne_arrondie
        except Exception as e:
            print(f"Error in moyenne_arrondie: {e}")

    def flatten_data(self, lst):
        try:
            if lst == []:
                return []
            res = []
            index_tested = 0
            elem_tested = lst[index_tested]
            while index_tested < len(lst):
                sub_list = []
                for i in range(index_tested, index_tested + min(7, len(lst) - index_tested)):
                    if abs(lst[i] - elem_tested) <= 7:
                        sub_list.append(lst[i])
                res.append(self.moyenne_arrondie(sub_list))
                index_tested += len(sub_list)
                if index_tested < len(lst):
                    elem_tested = lst[index_tested]
            return res
        except IndexError as e:
            print(f"IndexError in flatten_data: {e}")
            return []

    def image_lines_detector(self, image):
        list_index = []
        try:
            for index, elem in enumerate(image):
                if self.sequence_detector_line(elem, 0, image.shape[1]):
                    list_index.append(index)
            list_index = self.flatten_data(list_index)
            return list_index
        except IndexError as e:
            print(f"IndexError in image_lines_detector: {e}")
            return []

    def sequence_detector_column(self, lst, deb, fin, taille_seq_col=150):
        count_zeros = 0
        try:
            for element in lst[deb:fin]:
                if element == 0:
                    count_zeros += 1
                    if count_zeros == taille_seq_col:
                        return True
                else:
                    count_zeros = 0
            return False
        except IndexError as e:
            print(f"IndexError in sequence_detector_column: {e}")
            return False

    def parcours_liste(self, liste_de_listes):
        valeurs_vues = set()
        try:
            for sous_liste in liste_de_listes:
                for i, valeur in enumerate(sous_liste):
                    nouvelle_valeur = True
                    for vue in valeurs_vues:
                        if vue - 5 <= valeur <= vue + 5:
                            sous_liste[i] = vue
                            nouvelle_valeur = False
                            break
                    if nouvelle_valeur:
                        for vue in valeurs_vues:
                            if valeur - 5 <= vue <= valeur + 5:
                                sous_liste[i] = vue
                        valeurs_vues.add(valeur)
            return liste_de_listes
        except IndexError as e:
            print(f"IndexError in parcours_liste: {e}")
            return liste_de_listes

    def quadrillage(self, transposed_image, liste_lines):
        list_columns = []
        try:
            for i in range(len(liste_lines) - 1):
                sub_list_columns = []
                for index, col in enumerate(transposed_image):
                    if self.sequence_detector_column(col, liste_lines[i], liste_lines[i + 1]):
                        sub_list_columns.append(index)
                list_columns.append(self.flatten_data(sub_list_columns))
            self.parcours_liste(list_columns)
            coordonnates = []
            for index in range(len(list_columns)):
                for col_index in list_columns[index]:
                    coordonnates.append([liste_lines[index], col_index])
                    coordonnates.append([liste_lines[index + 1], col_index])
            return coordonnates
        except IndexError as e:
            print(f"IndexError in quadrillage: {e}")
            return []

    def detect_intersections(self, image_path, eps=5, min_samples=1):
        try:
            # Chargement de l'image prétraitée
            mortstats = self.charger_image_en_numpy(image_path)
            mortstats = np.where(mortstats > 127, 1, 0)
            
            # Tracer des lignes pour délimiter la zone d'intérêt
            variable = 20
            mortstats = cv2.line(mortstats, (variable, variable), (variable, mortstats.shape[0] - variable), (0, 0, 255), 8)
            mortstats = cv2.line(mortstats, (mortstats.shape[1] - variable, variable), (mortstats.shape[1] - variable, mortstats.shape[0] - variable), (0, 0, 255), 8)
            mortstats = cv2.line(mortstats, (variable, mortstats.shape[0] - variable), (mortstats.shape[1] - variable, mortstats.shape[0] - variable), (0, 0, 255), 8)
            
            # Détection des lignes
            list_lignes = self.image_lines_detector(mortstats)
            
            # Transposer l'image pour détecter les colonnes
            mortstats_transposed = mortstats.transpose()
            
            # Détecter les intersections (points d'intérêt)
            points = self.quadrillage(mortstats_transposed, list_lignes)

            # Clustering des points
            points = np.array(points)

            # Clustering points
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = clustering.labels_
            
            # Calcul des points médians pour chaque cluster
            unique_labels = set(labels)
            median_points = []

            for label in unique_labels:
                if label != -1:  # Excluding outliers
                    cluster_points = points[labels == label]
                    if cluster_points.size > 0:
                        median_point = np.median(cluster_points, axis=0).astype(int)
                        median_points.append(tuple(median_point))
                    else:
                        print(f"No points for label {label}")

            # Dessiner les points médians et tracer des lignes entre eux
            for point in median_points:
                x, y = int(point[0]), int(point[1])  # Ensure integer conversion
                cv2.drawMarker(mortstats, (y, x), 255, markerType=cv2.MARKER_CROSS, markerSize=50, thickness=30)

            for i, point1 in enumerate(median_points):
                for point2 in median_points[i+1:]:
                    x1, y1 = int(point1[0]), int(point1[1])
                    x2, y2 = int(point2[0]), int(point2[1])
                    if abs(x1 - x2) < 10 or abs(y1 - y2) < 10:
                        cv2.line(mortstats, (y1, x1), (y2, x2), (0, 255, 0), 30)
            
            # Enregistrer l'image avec les intersections détectées
            scaled_array = mortstats * 255
            scaled_array = scaled_array.astype(np.uint8)
            image = Image.fromarray(scaled_array)
            
            return image, median_points
        except Exception as e:
            print(f"Error in detect_intersections: {e}")
            return None
        
    def apply_ocr(self, input_path, median_points):
        client = vision.ImageAnnotatorClient()

        with io.open(input_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        image_context = vision.ImageContext(language_hints=["en"])
        response = client.text_detection(image=image, image_context=image_context)
        texts = response.text_annotations

        img = Image.open(input_path)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('arial.ttf', size=50)

        data = []
        for text in texts:
            if re.search('[a-zA-Z0-9]', text.description) and len(text.description) <= 50:
                vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                data.append((text.description, vertices))

        liste = self.final_pro(data, median_points)

        for text, vertices in data:
            if len(vertices) > 0:
                vertices.append(vertices[0])
                for i in range(len(vertices)-1):
                    draw.line([vertices[i], vertices[i+1]], fill='red', width=5)
                draw.text((vertices[0][0], vertices[0][1] - 50), text, fill='blue', font=font)
                
        print(liste)

        return img, liste
    
    def is_inside(self, rect1, rect2):
        for point in rect1:
            x, y = point
            if not (rect2[0][0] < x < rect2[1][0] and rect2[0][1] < y < rect2[2][1]):
                return False
        return True

    def final_pro(self, data, points):
        List_tot = []
        for j in range(0, len(points) - 3, 2):
            point_j, point_j1, point_j2, point_j3 = points[j:j+4]

            if point_j[0] == point_j2[0] and point_j1[0] == point_j3[0]:
                points_tracees = [point_j, point_j2, point_j3, point_j1]
                points_tracees = [(pt[1], pt[0]) for pt in points_tracees]  # Coordonnées (y, x)

                Liste_mots = []
                for word, vertices in data:
                    if self.is_inside(vertices, points_tracees):
                        Liste_mots.append(word)

                if Liste_mots:
                    List_tot.append(Liste_mots)

        return List_tot
    
    def gpt2json(self, data, model_choice):
        # Chargement de l'API key à partir d'un fichier
        with open('./config/gpt_config.json', 'r') as file:
            config = json.load(file)

        if model_choice == "Google Vision OCR + GPT-4 Turbo":
            self.model = "gpt-4-turbo-2024-04-09"
        elif model_choice == "Google Vision OCR + GPT-3.5 Turbo":
            self.model = "gpt-3.5-turbo-0125"
        else:
            self.model = config['model']
        
        self.api_key = config['api_key']
        self.client = OpenAI(api_key=self.api_key)
        
        # Envoi de la requête à OpenAI pour complétion
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in table structures and your primary responsibility is to transform Python lists representing tables into structured JSON files suitable for analysis."
                },
                {
                    "role": "user",
                    "content": "Create a JSON object representing the provided Table data. Use 'title' for the table's main topic, 'data' to encapsulate the table content, with nested objects for each area and time period. Include summarized categories like 'annual averages' and specific years as distinct keys, and provide data points for each region and condition where applicable. Handle missing or incomplete data with null values, and ensure the JSON structure is suitable for future data analysis. Table:" + str(data)
                }
            ]
        )

        json_response = json.loads(response.choices[0].message.content)

        print(json.dumps(json_response, indent=4))
        return json_response