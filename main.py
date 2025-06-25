from torchvision import models, transforms
import streamlit as st
import torch
from PIL import Image
import urllib.request
from captum.attr import IntegratedGradients
from torchcam.methods import SmoothGradCAMpp
import torchxrayvision as xrv

st.set_page_config(
    page_title="Deep-Viz",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Welcome to Deep-Viz")

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(LABELS_URL) as f:
    class_names = [line.strip() for line in f.readlines()]

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        caption="Sample Image",
    )
    st.write(
        "This app uses a pretrained ResNet18 model to visualize image predictions and explanations."
    )

    selected_model = st.selectbox("Select a model", ["ResNet18", "XRayVision"], index=0)


if selected_model == "XRayVision":
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
else:
    model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


available_layers = []
for idx, m in enumerate(model.named_modules()):
    print(idx, "->", m[0])
    if "relu" in m[0]:
        continue
    available_layers.append(m[0])


# side bar stuffs
with st.sidebar:
    image_uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

st.sidebar.title("Settings")
st.sidebar.write("Select the layer for CAM extraction:")
selected_layer = st.sidebar.selectbox(
    "Layer",
    available_layers,
    index=available_layers.index(available_layers[1]),
)

print("Selected layer:", selected_layer)
# gotta CREATE CAM EXTRACTOR BEFORE
cam_extractor = SmoothGradCAMpp(model, target_layer=selected_layer)
ig = IntegratedGradients(model)

col1, col2 = st.columns(2)


if image_uploaded is not None:
    # Convert to grayscale if using XRayVision, else RGB
    if selected_model == "XRayVision":
        image_uploaded = Image.open(image_uploaded).convert("L")
        preprocess_xrv = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),  # Will produce shape [1, H, W]
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        preprocessed_image = preprocess_xrv(image_uploaded)
    else:
        image_uploaded = Image.open(image_uploaded).convert("RGB")
        preprocessed_image = preprocess(image_uploaded)
    with st.sidebar:
        st.image(image_uploaded, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing..."):
        input_batch = preprocessed_image.unsqueeze(0)

        # get the model poreds
        # with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        with st.sidebar:
            st.write("Top 5 Predictions:")
            for i in range(top5_prob.size(0)):
                class_name = class_names[top5_catid[i]]
                st.write(
                    f"{i + 1}. {top5_catid[i].item()} - {str(class_name)} - {top5_prob[i].item():.4f}"
                )

        try:
            # cam extractions and all
            cam = cam_extractor(top5_catid[0].item(), output)
            # print(cam)
            cam = cam[0].squeeze(0).cpu().numpy()
            cam_image = Image.fromarray((cam * 255).astype("uint8"))
            with col2:
                st.write(
                    "Using `SmoothGradCAMpp` for Class Activation Map (CAM) extraction."
                )
                st.image(
                    cam_image, caption="Class Activation Map", use_container_width=True
                )
        except Exception as e:
            st.error(f"Error generating CAM using SmoothGradCAMpp: {e}")

        try:
            # captum integrated gradients
            attributions_ig = ig.attribute(input_batch, target=top5_catid[0].item())
            attributions_ig = attributions_ig.squeeze(0).cpu().detach().numpy()
            attributions_ig = attributions_ig.transpose(1, 2, 0)
            print(attributions_ig.shape)
            attributions_ig_image = Image.fromarray(
                (attributions_ig * 255).astype("uint8")
            )
            with col1:
                st.write("Using `Integrated Gradients` for SHAP values.")
                st.image(
                    attributions_ig_image,
                    caption="SHAP Values using Integrated Gradients",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Error generating SHAP values: {e}")


st.markdown("""
---
**Note:**  
This app demonstrates deep learning model interpretability using visual explanations.  
- **Class Activation Maps (CAM)** highlight regions in the image that are important for the model's prediction.
- **Integrated Gradients** provide feature attributions to understand which pixels most influence the output.

Upload an image to see how a pretrained ResNet18 "sees" and explains its predictions!
""")
