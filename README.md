# CYPO - Conversion de PDF en JSON

CYPO est un outil puissant qui convertit les fichiers PDF en format JSON structuré, permettant une analyse et un traitement efficaces des données. Ce projet utilise PyMuPDF pour le traitement des PDF, OpenCV pour le traitement d'images, Google Cloud Vision OCR pour la reconnaissance de texte et les modèles GPT d'OpenAI pour la structuration intelligente des données.

## Fonctionnalités

- Convertir les fichiers PDF en format JSON structuré
- Prise en charge de la sélection de la plage de pages à convertir
- Offre trois modes de traitement :
 1. Google Vision OCR + GPT-3.5 Turbo
 2. Google Vision OCR + GPT-4 Turbo
 3. Google Vision OCR + Modèle GPT personnalisé
- Estimation du coût de conversion en fonction du nombre de pages et du mode de traitement sélectionné
- Affichage de la progression en temps réel et des journaux pendant la conversion
- Sortie des fichiers JSON dans un dossier dédié pour un accès facile

## Options d'installation

Vous avez deux options pour utiliser CYPO :

1. **Télécharger l'exécutable** : Téléchargez le dossier `_internal` contenant toutes les dépendances nécessaires et placez-le dans le même dossier que le fichier exécutable `CYPO.exe`.

2. **Compiler le code source** : Vous pouvez compiler le code source (`app.py` et `Process.py`) à condition d'avoir Python et toutes les bibliothèques requises installées. Assurez-vous d'avoir installé les dépendances suivantes : PyQt5, PyMuPDF, OpenCV, Google Cloud Vision API et OpenAI API.

## Configuration

L'application nécessite deux fichiers de configuration :

1. `config/vision_config.json` : Ce fichier doit contenir vos identifiants de l'API Google Cloud Vision.
```json
{
  "api_key": "VOTRE CLE API",
  "model": "gpt-4-turbo-2024-04-09"
 }
```
   
2. `config/gpt_config.json` : Ce fichier doit contenir votre clé API OpenAI et le modèle GPT souhaité (si vous utilisez un modèle personnalisé).
```json
{
  "type": "service_account",
  "project_id": "REMPLACER ICI",
  "private_key_id": "REMPLACER ICI",
  "private_key": "REMPLACER ICI",
  "client_email": "REMPLACER ICI",
  "client_id": "REMPLACER ICI",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-vision-students%40skillful-coast-419914.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```

## Ressources

- Le dossier `config` contient des ressources telles que `icon.ico` et `logo.png` utilisées dans l'application.
- Un fichier de test `test.pdf` est fourni pour vous permettre de réaliser des tests pas très coûteux sur des tableaux pertinents.

## Utilisation

1. Exécutez le fichier exécutable `CYPO.exe` ou le code source compilé.
2. Cliquez sur le bouton "Sélectionner un fichier PDF" et naviguez jusqu'au fichier PDF souhaité.
3. Entrez la plage de pages que vous souhaitez convertir (par exemple, `1-5, 8, 10-11`).
4. Sélectionnez le mode de traitement dans le menu déroulant.
5. Cliquez sur le bouton "Convertir en JSON" pour démarrer le processus de conversion.
6. Surveillez la barre de progression et les messages du journal pour suivre les mises à jour en temps réel.
7. Une fois la conversion terminée, vous pouvez trouver les fichiers JSON générés dans le dossier `4-json`.

## Sortie

Au cours du traitement, quatre dossiers sont créés pour enregistrer l'état du scan à chaque phase :

1. `1-preprocess` : Contient les images prétraitées des pages sélectionnées.
2. `2-intersection` : Contient les images avec les intersections détectées.
3. `3-ocr` : Contient les images avec le texte reconnu par OCR et annoté.
4. `4-json` : Contient les fichiers JSON finaux structurés pour chaque page.

## Aperçu

![Aperçu de l'interface utilisateur](https://github.com/Ashitaka06/CYPO/assets/100866077/aa8dab07-a6f9-473f-bb9c-60b73d795474)

*Aperçu de l'interface utilisateur de CYPO*

![Exemple de sortie JSON](https://github.com/Ashitaka06/CYPO/assets/100866077/8d52c895-ef4b-4085-9483-50f20ba72287)

*Exemple de sortie JSON générée par CYPO*

## Contribution

Les contributions sont les bienvenues ! Si vous trouvez des problèmes ou avez des suggestions d'amélioration, n'hésitez pas à ouvrir un ticket ou à soumettre une pull request.

## Licence

Ce projet est sous licence [MIT License](LICENSE).

## Remerciements

- lamarqueni@cy-tech.fr
- castanetfl@cy-tech.fr
- gc.chapelle@gmail.com
- souhila.arib@cyu.fr
- labordetho@cy-tech.fr
