import json
import logging
from typing import List
import PIL
import numpy as np
import streamlit as st
from albumentations import Compose, LongestMaxSize, Normalize, PadIfNeeded
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import os

class ClassifyModel:
    def __init__(self):
        self.model = None
        self.class2tag = None
        self.tag2class = None
        self.transform = None

    def load(self, path="D:/WheatRust-StreamlitApp-main/"):
        image_size = 512
        self.transform = Compose(
            [
                LongestMaxSize(max_size=image_size),
                PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),  # Add value here
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),
                ToTensorV2()
            ]
        )
        self.model = torch.jit.load(os.path.join(path, "model.pth"))
        with open(os.path.join(path, "tag_class.json")) as fin:
            self.tag2class = json.load(fin)
            self.class2tag = {v: k for k, v in self.tag2class.items()}
            logging.debug(f"class2tag: {self.class2tag}")

    def predict(self, *imgs) -> List[str]:
        logging.debug(f"batch size: {len(imgs)}")
        input_ts = [self.transform(image=img)["image"] for img in imgs]
        input_t = torch.stack(input_ts)
        logging.debug(f"input_t: {input_t.shape}")
        output_ts = self.model(input_t)
        activation_fn = torch.nn.Sigmoid()
        output_ts = activation_fn(output_ts)
        labels = list(self.tag2class.keys())
        logging.debug(f"output_ts: {output_ts.shape}")
        results = []
        threshold = 0.5
        for output_t in output_ts:
            logit = (output_t > threshold).long()
            if logit[0] and any([*logit[1:3], *logit[4:]]): 
                output_t[0] = 0
            indices = (output_t > threshold).nonzero(as_tuple=True)[0]
            prob = output_t[indices].tolist()
            tag = [labels[i] for i in indices.tolist()]
            res_dict = dict(zip(labels, output_t.tolist()))
            logging.debug(f"all results: {res_dict}")
            logging.debug(f"prob: {prob}")
            logging.debug(f"result: {tag}")
            results.append((tag, prob, res_dict))
        result = {k: v for k, v in results[-1][2].items() if k in ['healthy', 'leaf rust', 'stem rust']}
        if result:
            wheat_type, confidence = max(result.items(), key=lambda item: item[1])
            return wheat_type, confidence
        else:
            return "Unknown", 0.0

# Initialize and load the model
model = ClassifyModel()
model.load()

# Streamlit UI
st.sidebar.title("About")
st.sidebar.info("This application identifies the crop health in the picture.")

st.title('Wheat Rust Identification')

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).resize((512, 512))
    img = np.array(image)
    wheat_type, confidence = model.predict(img)
    st.write(f"I think this is **{wheat_type}** (confidence: **{round(confidence * 100, 2)}%**)")
    st.image(image, caption='Uploaded Image', use_column_width=True)
