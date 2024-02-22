import json
import os
from PIL import Image, ImageDraw, ImageFont, ImageTk
import random
from tkinter import Tk, Label
import copy

def data_loader(json_path):
    try:
        Data = json.load(open(json_path))
        return Data

    except FileNotFoundError:
        print(f"Aucun fichier à l'adresse {json_path}")
        return None

def data_parser(Data, name=None, id=None):
    Dict = []

    if name == None and id != None:
        try:
            name = Data[id]["name"]

        except IndexError:
            print("Veuillez ne choisir qu'une méthode de sélection entre name et id.")
            return None

    if name != None:
        for dict in Data:
            if dict["name"] == name:
                Dict.append(dict)

        if Dict != []:
            return Dict
        else:
            print(f"Aucun dictionnaire n'a pour valeur d'attribut name '{name}'.")
            return None

    else:
        print("Veuillez ne choisir qu'une méthode de sélection entre name et id.")
        return None


def draw_res(data, img):
    bbox = data["bbox"]
    score = data["score"]

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./resources/Roboto-Black.ttf", size=20)
    rectangle_coords = bbox
    txt_start_pt = bbox[0:2]

    draw.rectangle(rectangle_coords, outline="red", width=2)
    draw.text(txt_start_pt, "{:.2f}".format(round(score, 3)), fill="white", font=font)

    return img



def main():
    thresh = 0.8
    json_path = "./newestEMOTIC_train_x1y1x2y2.json"
    emotic_path = './EMOTIC (1)/EMOTIC/PAMI/emotic'
    Data = json.load(json_path=json_path)
    random_list = [random.randint(0, len(Data)-1) for i in range(15)]

    for id in random_list:
        count_person = 0
        Dict = []
        Dict = data_parser(Data, id=id)

        img_path = os.path.join(emotic_path, Dict[0]["folder"], Dict[0]["name"])
        img = Image.open(img_path)
        img_original = copy.deepcopy(img)

        for data in Dict:
            if data['score'] >=thresh:
                count_person += 1
                img = draw_res(data, img)

        window = Tk()
        window.title(Dict[0]["name"])
        img_tk = ImageTk.PhotoImage(img)
        img_original_tk = ImageTk.PhotoImage(img_original)
        canva2 = Label(window, image=img_original_tk)
        canva2.pack()
        canva = Label(window, image=img_tk)
        canva.pack()

        print(f"Nombre de personnes détectées: {count_person}")
        window.mainloop()


if __name__ == '__main__':
    main()
