import tkinter as tk
import requests
from PIL import Image, ImageTk
from tkinter.font import Font
from tkinter import *
from random import seed
from random import random
from random import gauss
from random import randint

import os
import cv2
import tensorflow as tf
from tensorflow import keras
import csv
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np

height = 224
width = 224

iter_val = 0
#MODEL
model = tf.keras.models.load_model('trained_supertml_model')

def build_data_line(buttons):
	to_return = {}
	to_return['Time'] = str(adjust_time(time_entry.get(), time_entry2.get()))
	to_return['Amount'] = str(adjust_amount(amount_entry.get()))
	for num in range(1, 29):
		key_string = 'V' + str(num)
		to_return[key_string] = buttons[num-1].get()
	img_name = create_tml_image(to_return)
	tml_generated = ImageTk.PhotoImage(Image.open(os.path.join('demo', img_name)))
	img_to_update = Image.open(os.path.join('demo', img_name))
	img_resized = img_to_update.resize((300,300), Image.ANTIALIAS)
	img_resized_display = ImageTk.PhotoImage(img_resized)
	tml_label.configure(image=img_resized_display)
	tml_label.image = img_resized_display
	img_to_load = Image.open(os.path.join('demo', img_name))
	img_data = np.asarray(img_to_load)
	print(img_data.shape)

	#LOADING MODEL AND WEIGTHS
	to_predict_data = tf.data.Dataset.from_tensor_slices(([img_data], [0]))
	to_predict = to_predict_data
	BATCH_SIZE=1
	predict_batches=to_predict.batch(BATCH_SIZE)
	prediction = model.predict(predict_batches)
	if (prediction < 0):
		result = "No Fraudulent Activity!"
		result_label.configure(text = result)
		result_label.text = result
		result_label.configure(bg='green')
		result_label.bg = 'green'
		print(0)
		return 0
	else:
		result = "Fraud!"
		result_label.configure(text = result)
		result_label.text = result
		result_label.configure(bg='red')
		result_label.bg = 'red'
		print(1)
		return 1

def create_tml_image(data_line):
    font = ImageFont.truetype('Arial.ttf', size = 12)
    tml_img = Image.new('RGB', (width, height), color = 'black')
    draw  = ImageDraw.Draw(tml_img)
    draw.text((10, 15), str(data_line['Time']), font=font, fill = (255, 255, 255))
    draw.text((55, 15), str(data_line['V1']), font=font, fill = (255, 255, 255))
    draw.text((100, 15), str(data_line['V2']), font=font, fill = (255, 255, 255))
    draw.text((145, 15), str(data_line['V3']), font=font, fill = (255, 255, 255))
    draw.text((185, 15), str(data_line['V4']), font=font, fill = (255, 255, 255))
    draw.text((10, 50), str(data_line['V5']), font=font, fill = (255, 255, 255))
    draw.text((55, 50), str(data_line['V6']), font=font, fill = (255, 255, 255))
    draw.text((100, 50), str(data_line['V7']), font=font, fill = (255, 255, 255))
    draw.text((145, 50), str(data_line['V8']), font=font, fill = (255, 255, 255))
    draw.text((185, 50), str(data_line['V9']), font=font, fill = (255, 255, 255))
    draw.text((10, 85), str(data_line['V10']), font=font, fill = (255, 255, 255))
    draw.text((55, 85), str(data_line['V11']), font=font, fill = (255, 255, 255))
    draw.text((100, 85), str(data_line['V12']), font=font, fill = (255, 255, 255))
    draw.text((145, 85), str(data_line['V13']), font=font, fill = (255, 255, 255))
    draw.text((185, 85), str(data_line['V14']), font=font, fill = (255, 255, 255))
    draw.text((10, 120), str(data_line['V15']), font=font, fill = (255, 255, 255))
    draw.text((55, 120), str(data_line['V16']), font=font, fill = (255, 255, 255))
    draw.text((100, 120), str(data_line['V17']), font=font, fill = (255, 255, 255))
    draw.text((145, 120), str(data_line['V18']), font=font, fill = (255, 255, 255))
    draw.text((185, 120), str(data_line['V19']), font=font, fill = (255, 255, 255))
    draw.text((10, 155), str(data_line['V20']), font=font, fill = (255, 255, 255))
    draw.text((55, 155), str(data_line['V21']), font=font, fill = (255, 255, 255))
    draw.text((100, 155), str(data_line['V22']), font=font, fill = (255, 255, 255))
    draw.text((145, 155), str(data_line['V23']), font=font, fill = (255, 255, 255))
    draw.text((185, 155), str(data_line['V24']), font=font, fill = (255, 255, 255))
    draw.text((10, 185), str(data_line['V25']), font=font, fill = (255, 255, 255))
    draw.text((55, 185), str(data_line['V26']), font=font, fill = (255, 255, 255))
    draw.text((100, 185), str(data_line['V27']), font=font, fill = (255, 255, 255))
    draw.text((145, 185), str(data_line['V28']), font=font, fill = (255, 255, 255))
    draw.text((185, 185), str(data_line['Amount']), font=font, fill = (255, 255, 255))
    img_name = str(int(data_line['Time'])) + '_' + str(0) + '.jpg'
    if (os.path.exists('demo')):
        shutil.rmtree('demo')
    os.mkdir('demo')
    tml_img.save(os.path.join('demo', img_name))
    return img_name

seed(2)

def set_random_time_ammount():
	set_text(str(randint(0,23)), time_entry)
	set_text(str(randint(0,59)), time_entry2)
	set_text(randint(0,100) + round(random(), 2), amount_entry)

def set_text(text, button):
    button.delete(0,END)
    button.insert(0,text)
    return

def adjust_time(hour, mins):
	if (hour==''):
		hour=0
	if mins=='':
		mins=0
	total_mins = 24*60
	current_mins = int(hour)*60 + int(mins)
	return int(284807*(current_mins/total_mins))

def adjust_amount(amount):
	if amount=='':
		return 0
	else:
		return float(amount)

def set_random_inputs():
	vals = []
	for num in range(28):
		vals.append(round(gauss(0,1),3))
	for idx in range(len(all_buttons)):
		set_text(vals[idx], all_buttons[idx])

root = tk.Tk()

WIDTH = root.winfo_screenwidth()
HEIGHT = root.winfo_screenheight()

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

output_frame = tk.Frame(root)
output_frame.place(relx=0.6, rely=0.12, relwidth=0.4, relheight=0.9)

tml_image = Image.open('default.jpg')
tml_image = tml_image.resize((300, 300), Image.ANTIALIAS)
tml_background = ImageTk.PhotoImage(tml_image)
tml_label = tk.Label(output_frame, image=tml_background)
tml_label.place(relx=0.22, rely=0.15, relheight=0.40, relwidth=0.55)

tml_title = tk.Label(output_frame, text="SuperTML Image", font=("Times New Roman", 30))
tml_title.config(anchor=CENTER)
tml_title.place(relx=0, rely=0, relheight=.1, relwidth=1)

tml_output_label = tk.Label(output_frame, text="Output of SuperTML Model:", font=("Times New Roman", 30))
tml_output_label.config(anchor=CENTER)
tml_output_label.place(relx=0, rely=0.65, relheight=0.1, relwidth=1)

result_label = tk.Label(output_frame, text="", bg='#FFFF99', font=("Arial", 30), justify=CENTER)
result_label.config(anchor=CENTER)
result_label.place(relx=0.1, rely=0.8, relheight=0.1, relwidth=0.8)
#tml_label.place(relx=0.60, rely=0.15, relheight=.5, relwidth=.4)

title_label = tk.Label(root, text="Gyrfalcon's SuperTML Method for Credit Card Fraud Detection", font=("Times New Roman", 40))
title_label.config(anchor=CENTER)
title_label.place(relx=0.01, rely=.01, relheight = .1, relwidth = 1)

input_frame = tk.Frame(root, bg='#80c1ff', bd=10)
input_frame.place(relx=0, rely=0.12, relwidth=.6, relheight=.9)
main_panel = tk.Label(input_frame)
main_panel.place(relwidth=1, relheight=1)

prediction_panel_title = tk.Label(main_panel, text = "CREATE UNIQUE CREDIT CARD FEATURES TO TEST", font=("Times New Roman", 20))
prediction_panel_title.config(anchor = CENTER)
prediction_panel_title.place(relx=0, rely=0, relheight=.1, relwidth=1)

disclaimer = tk.Label(main_panel, text = "For security reasons specific features for each credit card are reffered to with placeholders V1,V2,...V28, since this is model was trained on real credit card data from European cardholders. You can either enter specific values or randomize with the \"Randomize\" button You can enter a custom \"Time\" and \"Ammount\" as well", font=("Times New Roman", 15), anchor=N, justify=CENTER, wraplength=int(WIDTH*.55))
disclaimer.place(relx=0, rely=.1, relheight=0.3, relwidth = 1)

input_val_frame = tk.Frame(input_frame,bg='#C8C8C8', bd=5)
input_val_frame.place(relx=0.02, rely=0.2, relheight=.7, relwidth=.75)

input_time_frame = tk.Frame(input_frame,bg='#C8C8C8', bd=5)
input_time_frame.place(relx=0.80, rely=0.30, relheight=.4, relwidth=.15)

time_frame = tk.Frame(input_time_frame)
time_frame.place(relx=0.01, rely=0.05, relheight=0.05, relwidth=0.98)

time_input = tk.Label(time_frame, text="Transaction Time")
time_input.place(relx=0, rely=0, relheight=1, relwidth=1)

time_entry = tk.Entry(input_time_frame, font=("Helvetica", 14), justify=CENTER)
time_entry.place(relx=0.2, rely=0.15, relwidth=0.2, relheight=.1)
colon = tk.Label(input_time_frame, text=":", font=("Helvetica", 25), justify=CENTER)
colon.place(relx=0.42, rely=0.15, relwidth=0.07, relheight=.1)
time_entry2 = tk.Entry(input_time_frame, font=("Helvetica", 14), justify=CENTER)
time_entry2.place(relx=0.50, rely=0.15, relwidth=0.28, relheight=.1)

amount_frame = tk.Frame(input_time_frame)
amount_frame.place(relx=0.01, rely=0.4, relheight=0.05, relwidth=0.98)

amount_label = tk.Label(amount_frame, text="Amount (in $)")
amount_label.place(relx=0, rely=0, relheight=1, relwidth=1)

amount_entry = tk.Entry(input_time_frame, font=("Helvetica", 14), justify=CENTER)
amount_entry.place(relx=0.2, rely=0.5, relwidth=0.6, relheight=.1)

rand_gen_button = tk.Button(input_time_frame, text="Generate Random Time & Amount", font=("Times New Roman", 14), justify=CENTER, wraplength=100, command=lambda: set_random_time_ammount())
rand_gen_button.place(relx=0.05, rely=0.7, relwidth=0.9, relheight=0.25)

run_button = Button(input_frame, highlightbackground='#90EE90', text="RUN SuperTML", font=("Helvetica", 20), justify=CENTER, wraplength=100)
run_button.place(relx=0.8, rely=0.75, relwidth=.15, relheight=0.15)

all_buttons = []
#ROW 1
v1 = tk.Frame(input_val_frame)
v1.place(relx=0, rely=0, relheight=(1/7), relwidth=1/4)

v1_title = tk.Label(v1, text='V1', font=("Times New Roman", 15))
v1_title.config(anchor=CENTER)
v1_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v1_entry = tk.Entry(v1, font=("Helvetica", 14), justify=CENTER)
v1_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v1_entry)

v2 = tk.Frame(input_val_frame)
v2.place(relx=1/4, rely=0, relheight=(1/7), relwidth=1/4)

v2_title = tk.Label(v2, text='V2', font=("Times New Roman", 20))
v2_title.config(anchor=CENTER)
v2_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v2_entry = tk.Entry(v2, font=("Helvetica", 14), justify=CENTER)
v2_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v2_entry)

v3 = tk.Frame(input_val_frame)
v3.place(relx=2/4, rely=0, relheight=(1/7), relwidth=1/4)

v3_title = tk.Label(v3, text='V3', font=("Times New Roman", 20))
v3_title.config(anchor=CENTER)
v3_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v3_entry = tk.Entry(v3, font=("Helvetica", 14), justify=CENTER)
v3_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v3_entry)

v4 = tk.Frame(input_val_frame)
v4.place(relx=3/4, rely=0, relheight=(1/7), relwidth=1/4)

v4_title = tk.Label(v4, text='V4', font=("Times New Roman", 20))
v4_title.config(anchor=CENTER)
v4_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v4_entry = tk.Entry(v4, font=("Helvetica", 14), justify=CENTER)
v4_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v4_entry)
#ROW2
v5 = tk.Frame(input_val_frame)
v5.place(relx=0, rely=1/7, relheight=(1/7), relwidth=1/4)

v5_title = tk.Label(v5, text='V5', font=("Times New Roman", 20))
v5_title.config(anchor=CENTER)
v5_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v5_entry = tk.Entry(v5, font=("Helvetica", 14), justify=CENTER)
v5_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v5_entry)

v6 = tk.Frame(input_val_frame)
v6.place(relx=1/4, rely=1/7, relheight=(1/7), relwidth=1/4)

v6_title = tk.Label(v6, text='V6', font=("Times New Roman", 20))
v6_title.config(anchor=CENTER)
v6_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v6_entry = tk.Entry(v6, font=("Helvetica", 14), justify=CENTER)
v6_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v6_entry)
v7 = tk.Frame(input_val_frame)
v7.place(relx=2/4, rely=1/7, relheight=(1/7), relwidth=1/4)

v7_title = tk.Label(v7, text='V7', font=("Times New Roman", 20))
v7_title.config(anchor=CENTER)
v7_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v7_entry = tk.Entry(v7, font=("Helvetica", 14), justify=CENTER)
v7_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v7_entry)
v8 = tk.Frame(input_val_frame)
v8.place(relx=3/4, rely=1/7, relheight=(1/7), relwidth=1/4)

v8_title = tk.Label(v8, text='V8', font=("Times New Roman", 20))
v8_title.config(anchor=CENTER)
v8_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v8_entry = tk.Entry(v8, font=("Helvetica", 14), justify=CENTER)
v8_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v8_entry)
#ROW 3
v9 = tk.Frame(input_val_frame)
v9.place(relx=0, rely=2/7, relheight=(1/7), relwidth=1/4)

v9_title = tk.Label(v9, text='V9', font=("Times New Roman", 20))
v9_title.config(anchor=CENTER)
v9_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v9_entry = tk.Entry(v9, font=("Helvetica", 14), justify=CENTER)
v9_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v9_entry)
v10 = tk.Frame(input_val_frame)
v10.place(relx=1/4, rely=2/7, relheight=(1/7), relwidth=1/4)

v10_title = tk.Label(v10, text='V10', font=("Times New Roman", 20))
v10_title.config(anchor=CENTER)
v10_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v10_entry = tk.Entry(v10, font=("Helvetica", 14), justify=CENTER)
v10_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v10_entry)
v11 = tk.Frame(input_val_frame)
v11.place(relx=2/4, rely=2/7, relheight=(1/7), relwidth=1/4)

v11_title = tk.Label(v11, text='V11', font=("Times New Roman", 20))
v11_title.config(anchor=CENTER)
v11_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v11_entry = tk.Entry(v11, font=("Helvetica", 14), justify=CENTER)
v11_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v11_entry)
v12 = tk.Frame(input_val_frame)
v12.place(relx=3/4, rely=2/7, relheight=(1/7), relwidth=1/4)

v12_title = tk.Label(v12, text='V12', font=("Times New Roman", 20))
v12_title.config(anchor=CENTER)
v12_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v12_entry = tk.Entry(v12, font=("Helvetica", 14), justify=CENTER)
v12_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v12_entry)
#ROW 4
v13 = tk.Frame(input_val_frame)
v13.place(relx=0, rely=3/7, relheight=(1/7), relwidth=1/4)

v13_title = tk.Label(v13, text='V13', font=("Times New Roman", 20))
v13_title.config(anchor=CENTER)
v13_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v13_entry = tk.Entry(v13, font=("Helvetica", 14), justify=CENTER)
v13_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v13_entry)
v14 = tk.Frame(input_val_frame)
v14.place(relx=1/4, rely=3/7, relheight=(1/7), relwidth=1/4)

v14_title = tk.Label(v14, text='V14', font=("Times New Roman", 20))
v14_title.config(anchor=CENTER)
v14_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v14_entry = tk.Entry(v14, font=("Helvetica", 14), justify=CENTER)
v14_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v14_entry)
v15 = tk.Frame(input_val_frame)
v15.place(relx=2/4, rely=3/7, relheight=(1/7), relwidth=1/4)

v15_title = tk.Label(v15, text='V15', font=("Times New Roman", 20))
v15_title.config(anchor=CENTER)
v15_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v15_entry = tk.Entry(v15, font=("Helvetica", 14), justify=CENTER)
v15_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v15_entry)
v16 = tk.Frame(input_val_frame)
v16.place(relx=3/4, rely=3/7, relheight=(1/7), relwidth=1/4)

v16_title = tk.Label(v16, text='V16', font=("Times New Roman", 20))
v16_title.config(anchor=CENTER)
v16_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v16_entry = tk.Entry(v16, font=("Helvetica", 14), justify=CENTER)
v16_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v16_entry)
#ROW 5

v17 = tk.Frame(input_val_frame)
v17.place(relx=0, rely=4/7, relheight=(1/7), relwidth=1/4)

v17_title = tk.Label(v17, text='V17', font=("Times New Roman", 20))
v17_title.config(anchor=CENTER)
v17_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v17_entry = tk.Entry(v17, font=("Helvetica", 14), justify=CENTER)
v17_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v17_entry)
v18 = tk.Frame(input_val_frame)
v18.place(relx=1/4, rely=4/7, relheight=(1/7), relwidth=1/4)

v18_title = tk.Label(v18, text='V18', font=("Times New Roman", 20))
v18_title.config(anchor=CENTER)
v18_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v18_entry = tk.Entry(v18, font=("Helvetica", 14), justify=CENTER)
v18_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v18_entry)
v19 = tk.Frame(input_val_frame)
v19.place(relx=2/4, rely=4/7, relheight=(1/7), relwidth=1/4)

v19_title = tk.Label(v19, text='V19', font=("Times New Roman", 20))
v19_title.config(anchor=CENTER)
v19_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v19_entry = tk.Entry(v19, font=("Helvetica", 14), justify=CENTER)
v19_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v19_entry)
v20 = tk.Frame(input_val_frame)
v20.place(relx=3/4, rely=4/7, relheight=(1/7), relwidth=1/4)

v20_title = tk.Label(v20, text='V20', font=("Times New Roman", 20))
v20_title.config(anchor=CENTER)
v20_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v20_entry = tk.Entry(v20, font=("Helvetica", 14), justify=CENTER)
v20_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v20_entry)
#ROW 5

v21 = tk.Frame(input_val_frame)
v21.place(relx=0, rely=5/7, relheight=(1/7), relwidth=1/4)

v21_title = tk.Label(v21, text='V21', font=("Times New Roman", 20))
v21_title.config(anchor=CENTER)
v21_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v21_entry = tk.Entry(v21, font=("Helvetica", 14), justify=CENTER)
v21_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v21_entry)
v22 = tk.Frame(input_val_frame)
v22.place(relx=1/4, rely=5/7, relheight=(1/7), relwidth=1/4)

v22_title = tk.Label(v22, text='V22', font=("Times New Roman", 20))
v22_title.config(anchor=CENTER)
v22_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v22_entry = tk.Entry(v22, font=("Helvetica", 14), justify=CENTER)
v22_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v22_entry)
v23 = tk.Frame(input_val_frame)
v23.place(relx=2/4, rely=5/7, relheight=(1/7), relwidth=1/4)

v23_title = tk.Label(v23, text='V23', font=("Times New Roman", 20))
v23_title.config(anchor=CENTER)
v23_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v23_entry = tk.Entry(v23, font=("Helvetica", 14), justify=CENTER)
v23_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v23_entry)
v24 = tk.Frame(input_val_frame)
v24.place(relx=3/4, rely=5/7, relheight=(1/7), relwidth=1/4)

v24_title = tk.Label(v24, text='V24', font=("Times New Roman", 20))
v24_title.config(anchor=CENTER)
v24_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v24_entry = tk.Entry(v24, font=("Helvetica", 14), justify=CENTER)
v24_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5) 
all_buttons.append(v24_entry)
#ROW 6

v25 = tk.Frame(input_val_frame)
v25.place(relx=0, rely=6/7, relheight=(1/7), relwidth=1/4)

v25_title = tk.Label(v25, text='V25', font=("Times New Roman", 20))
v25_title.config(anchor=CENTER)
v25_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v25_entry = tk.Entry(v25, font=("Helvetica", 14), justify=CENTER)
v25_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v25_entry)
v26 = tk.Frame(input_val_frame)
v26.place(relx=1/4, rely=6/7, relheight=(1/7), relwidth=1/4)

v26_title = tk.Label(v26, text='V26', font=("Times New Roman", 20))
v26_title.config(anchor=CENTER)
v26_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v26_entry = tk.Entry(v26, font=("Helvetica", 14), justify=CENTER)
v26_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v26_entry)
v27 = tk.Frame(input_val_frame)
v27.place(relx=2/4, rely=6/7, relheight=(1/7), relwidth=1/4)

v27_title = tk.Label(v27, text='V27', font=("Times New Roman", 20))
v27_title.config(anchor=CENTER)
v27_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v27_entry = tk.Entry(v27, font=("Helvetica", 14), justify=CENTER)
v27_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5)
all_buttons.append(v27_entry)
v28 = tk.Frame(input_val_frame)
v28.place(relx=3/4, rely=6/7, relheight=(1/7), relwidth=1/4)

v28_title = tk.Label(v28, text='V28', font=("Times New Roman", 20))
v28_title.config(anchor=CENTER)
v28_title.place(relx=0.25, rely=0.05, relheight=0.2, relwidth=0.5)

v28_entry = tk.Entry(v28, font=("Helvetica", 14), justify=CENTER)
v28_entry.place(relx=0.25, rely=0.5, relheight=0.4, relwidth=0.5) 
all_buttons.append(v28_entry)

randomize_button = tk.Button(input_frame, text="Randomize Input", font=("Helvetica", 14), anchor=CENTER, justify=CENTER, command=lambda: set_random_inputs())
randomize_button.place(relx=0.80, rely=0.2, relheight=0.05, relwidth=0.15) 

run_button = Button(input_frame, highlightbackground='#90EE90', text="RUN SuperTML", font=("Helvetica", 20), justify=CENTER, wraplength=100, command=lambda: build_data_line(all_buttons))
run_button.place(relx=0.8, rely=0.75, relwidth=.15, relheight=0.15)

root.mainloop()
