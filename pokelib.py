import requests
from tensorflow.keras.models import load_model
import os
import solution
import matplotlib.pyplot as plt,numpy as np
import cv2,pickle
model = load_model('model.h5')
class pokemon():
    def __init__(self,name: str):
        self.name=name.lower()
        raw=requests.get(f"https://pokeapi.co/api/v2/pokemon/{name.lower()}")
        a = raw.content
        a = eval(a.decode("utf-8").replace(':false',':"false"').replace(':true',':"true"').replace(':null',':"null"'))
        del a['moves']
        self.types = [t["type"]["name"] for t in a['types']]
        self.stats = dict([[t['stat']['name'],t['base_stat']] for t in a["stats"]])
        self.height = a["height"]
        self.weight = a["weight"]
        self.id = a["id"]

    def __repr__(self): return self.name,self.id,self.types,self.stats,self.height,self.weight
    def __str__(self): return f"( Name: {self.name} , Id: {self.id} , Types : {self.types} , Stats : {self.stats} , Height : {self.height} , Weight : {self.weight} )"

def findPromimentPokemon(inputImage : solution.Segment_Image or solution.Crypt_Image):
    f = open("pokemon_classes",'rb')
    data = pickle.load(f)
    f.close()
    im = cv2.cvtColor(cv2.resize(inputImage.img,[160,160]),cv2.COLOR_BGR2RGB)
    sol = model(im.reshape(1,160,160,3)/255)
    plt.imshow(im)

    ans = np.argmax(sol.numpy())
    return pokemon(data[ans].lower())

    # return pokemon(sol)
