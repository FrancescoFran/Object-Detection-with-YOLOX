import os
import subprocess
import tkinter as tk
from tkinter import *
from tkinter import Label, filedialog

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import *
from PIL import Image, ImageTk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    # Attempt to load the model using TFSMLayer
    model = load_model("ResNet50_opt_model")
except Exception as e:
    print("Error loading model as TensorFlow SavedModel:", e)

root = tk.Tk()
root.attributes("-fullscreen", True)
races_list = [
    "Afghan_hound",
    "Blenheim_spaniel",
    "Chihuahua",
    "Japanese_spaniel",
    "Maltese_dog",
    "Pekinese",
    "Rhodesian_ridgeback",
    "Shih-Tzu",
    "basset",
    "beagle",
    "black-and-tan_coonhound",
    "bloodhound",
    "bluetick",
    "papillon",
    "toy_terrier",
]


def adapt_size(image_data):
    # fonction qui adapte la taille l'image et l'affiche dans l'interface

    global img

    basewidth = 800
    img = Image.open(image_data)
    if img.size[0] > img.size[1]:
        wpercent = basewidth / float(img.size[0])
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    else:
        wpercent = basewidth / float(img.size[1])
        wsize = int((float(img.size[0]) * float(wpercent)))
        img = img.resize((wsize, basewidth), Image.Resampling.LANCZOS)

    img = ImageTk.PhotoImage(img)

    # création de Widget Label pour afficher l'image
    image = Label(frame1, image=img)
    image.pack()

    return


def load_img():
    # fonction pour charger et afficher une image

    global image_data

    for widget in frame1.winfo_children():
        if widget != background_label:
            widget.destroy()

    # chargement de l'image
    image_data = filedialog.askopenfilename(
        initialdir="/",
        title="Choisir une image",
        filetypes=(
            ("all files", "*.*"),
            ("png files", "*.png"),
            ("jpg files", "*.jpg"),
        ),
    )
    # adaptation de la taille de l'image
    adapt_size(image_data)
    # frame1.place(relwidth=1, relheight=0.8, relx=0, rely=0)
    # affichage du nom du fichier dans l'interface
    # Création de Label widget avec l'image de background
    # background_label = tk.Label(frame1, image=background_image)
    # background_label.place(x=0, y=0, relwidth=1, relheight=1)

    frame3 = tk.Frame(root, bg="white")
    frame3.place(relwidth=0.15, relheight=0.09, relx=0.425, rely=0.8)

    text = Label(frame3, text="Fichier: {}".format(os.path.basename(image_data)))
    text.configure(font=("Microsoft Sans Serif", 12), bg="white")
    text.place(relwidth=1, relheight=1, relx=0.5, rely=0.3, anchor="center")
    # classification de l'image
    image_classification()
    return


def image_classification():
    # fonction pour la classification d'images
    img_update = cv2.imread(image_data)
    # convert to RGB
    img_update = cv2.cvtColor(img_update, cv2.COLOR_BGR2RGB)
    # resize
    img_update = cv2.resize(img_update, (224, 224), interpolation=cv2.INTER_LINEAR)
    # amélioration du contraste
    img_update = cv2.cvtColor(img_update, cv2.COLOR_RGB2YUV)
    img_update[:, :, 0] = cv2.equalizeHist(img_update[:, :, 0])
    img_update = cv2.cvtColor(img_update, cv2.COLOR_YUV2RGB)
    # elimination du bruit avec filtre non local-means
    img_update = cv2.fastNlMeansDenoisingColored(
        src=img_update,
        dst=None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    # conversion image to array
    img_update = img_to_array(img_update)
    # On a formé le modèle sur des mini-batches, donc l'input est un tenseur de dimensions [batch_size, image_width,
    # image_height, number_of_channels]
    # Lors de la prédiction, on doit respecter cet format même si on a qu’une image: [1, image_width, image_height,
    # number_of_channels]
    img_update = np.expand_dims(img_update, axis=0)
    # preprocessing avec preprocess resnet
    img_update = tf.keras.applications.resnet.preprocess_input(img_update)
    prediction = np.argmax(model.predict(img_update), axis=1)[0]

    # affichage de la race dans l'interface
    frame4 = tk.Frame(root, bg="white")
    frame4.place(relwidth=0.15, relheight=0.03, relx=0.425, rely=0.85)
    text = Label(frame4, text="Race: {}".format(races_list[prediction]))
    text.configure(font=("Microsoft Sans Serif", 12), bg="white")
    text.place(relwidth=1, relheight=1, relx=0.5, rely=0.5, anchor="center")
    return


def image_det():
    # fonction pour la detection d'objets (chiens detectés par race) dans une image
    global img, image_data

    for widget in frame1.winfo_children():
        if widget != background_label:
            widget.destroy()

    # chargement de l'image
    image_data = filedialog.askopenfilename(
        initialdir="/",
        title="Choisir une image",
        filetypes=(
            ("all files", "*.*"),
            ("png files", "*.png"),
            ("jpg files", "*.jpg"),
            ("jpeg files", "*.jpeg"),
        ),
    )
    """
    Détection d'objet avec la fonction demo.py
    Arguments:
    - demo type: image
    - experiment description file: yolox_voc_m.py
    - ckpt for eval: best_ckpt_yolox_m.pth
    - path to images or video: image_data
    - test confidence: 0.25
    - test nms threshold: 0.45
    - test img size: 640
    - save_results: True
    - device to run our model: gpu
    """
    subprocess.run(
        "python demo.py image -f yolox_voc_m.py -c best_ckpt_yolox_m.pth --path "
        + image_data
        + " --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu ",
        shell=True,
    )

    # chargement de l'image image après détéction d'objets
    image_data = os.path.abspath(
        "YOLOX_outputs/yolox_voc_m/vis_res/" + image_data.split("/")[-1]
    )
    # adaptation de la taille de l'image
    adapt_size(image_data)
    # affichage du nom du fichier dans l'interface
    frame3 = tk.Frame(root, bg="white")
    frame3.place(relwidth=0.15, relheight=0.09, relx=0.425, rely=0.8)
    text = Label(frame3, text="Fichier: {}".format(os.path.basename(image_data)))
    text.configure(font=("Microsoft Sans Serif", 12), bg="white")
    text.place(relwidth=1, relheight=1, relx=0.5, rely=0.5, anchor="center")
    return


def video_det():
    # fonction pour la detection d'objets (chiens detectés par race) dans une vidéo

    for widget in frame1.winfo_children():
        if widget != background_label:
            widget.destroy()

    frame3 = tk.Frame(root, bg="white")
    frame3.place(relwidth=0.15, relheight=0.09, relx=0.425, rely=0.8)

    video_data = filedialog.askopenfilename(
        initialdir="/",
        title="Choisir une vidéo",
        filetypes=(
            ("all files", "*.*"),
            ("png files", "*.png"),
            ("jpg files", "*.jpg"),
        ),
    )
    """
    Détection d'objet avec la fonction demo.py
    Arguments:
    - demo type: video
    - experiment description file: yolox_voc_m.py
    - ckpt for eval: best_ckpt_yolox_m.pth
    - path to images or video: image_data
    - test confidence: 0.4
    - test nms threshold: 0.25
    - test img size: 640
    - save_results: True
    - device to run our model: gpu
    """
    subprocess.run(
        "python demo.py video -f yolox_voc_m.py -c best_ckpt_yolox_m.pth --path "
        + video_data
        + " --conf 0.4 --nms 0.25 --tsize 640 --save_result --device gpu ",
        shell=True,
    )
    return


def close():
    root.quit()
    return


canvas = tk.Canvas(root, height=450, width=450, bg="white")
canvas.pack()


# Création du mainframe
frame1 = tk.Frame(root)
frame1.place(relwidth=1, relheight=0.8, relx=0, rely=0)

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Import de l'image de background
image = Image.open("media/background.jpg")
image = image.resize((screen_width, screen_height), Image.LANCZOS)
# Création d'un objet PhotoImage
background_image = ImageTk.PhotoImage(image)
# Création de Label widget avec l'image de background
background_label = tk.Label(frame1, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

frame2 = tk.Frame(root, bg="white")
frame2.place(relwidth=1, relheight=0.5, relx=0, rely=0.8)
frame3 = tk.Frame(root, bg="white")
frame3.place(relwidth=0.15, relheight=0.15, relx=0.425, rely=0.8)

# création button pour classification image
chose_image = tk.Button(
    root,
    text="CHOOSE IMAGE FOR CLASSIFICATION",
    font=("Calibri", 14, "bold"),
    padx=10,
    pady=10,
    fg="white",
    bg="grey",
    command=load_img,
)
chose_image.place(relx=0.1, rely=0.85, anchor="center")

# création button pour détection image
chose_image_det = tk.Button(
    root,
    text="CHOOSE IMAGE FOR OBJ DETECTION",
    font=("Calibri", 14, "bold"),
    padx=10,
    pady=10,
    fg="white",
    bg="grey",
    command=image_det,
)
chose_image_det.place(relx=0.70, rely=0.85, anchor="center")

# création button pour détection vidèo
chose_video_det = tk.Button(
    root,
    text="CHOOSE VIDEO FOR OBJ DETECTION",
    font=("Calibri", 14, "bold"),
    padx=10,
    pady=10,
    fg="white",
    bg="grey",
    command=video_det,
)
chose_video_det.place(relx=0.90, rely=0.85, anchor="center")

# création button pour exit
exit_button = tk.Button(
    root,
    text="EXIT",
    font=("Calibri", 14, "bold"),
    padx=10,
    pady=10,
    fg="white",
    bg="grey",
    command=close,
)
exit_button.place(relx=0.5, rely=0.95, anchor="center")

# Assure que background label reste fixe
# background_label.lower()

root.mainloop()
