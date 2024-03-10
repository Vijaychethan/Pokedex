import ultralytics
from ultralytics import YOLO
import cv2
import torch
import streamlit as st
import pyttsx3

device = 'cuda'

labels = {0: 'pikachu', 1: 'charmander', 2: 'bulbasaur', 3: 'squirtle', 4: 'eevee', 5: 'other', 6: 'jigglypuff'}
Images = {0: 'Test_Images\Pikachu.png', 1: 'Test_Images\charmander.jpg', 2: 'Test_Images\bulbasaur.jpg', 3: 'Test_Images\ssquirtle.jpg', 4: 'Test_Images\eevee.jpg', 5: 'Test_Images\ash.jpg', 6: 'Test_Images\jigglipuff.jpg'}

descriptions = {
    "pikachu": "This is a cute pikachu",
    "charmander": "Flame tail",
    "bulbasaur": "Onlion pokemon",
    'squirtle': "Chad",
    'eevee': "MPD",
    'jigglypuff': "Singer"
}

def detect(frame, model):
    results = model.predict(source=frame)
    result = results[0]
    if result:
        box = result.boxes[0]
        conf = round(box.conf[0].item(), 2)
        class_index = box.cls[0].item()
        return class_index, conf
    else:
        return 7, 0

bg_img = '''
<style>
        [data-testid="stAppViewContainer"] {
        background-image: url('https://browsecat.art/sites/default/files/pokedex-background-128730-586989-3905341.png');
        background-size: cover;
        background-repeat: no-repeat;
        }
</style>
'''
st.markdown(bg_img, unsafe_allow_html=True)

center_align_css = """
<style>
    .center {
        display: flex;
        justify-content: center;
    }
    .small-image {
        width: 100%;
        max-width: 400px;
    }
</style>
"""

# Apply the CSS style to the Streamlit app
st.markdown(center_align_css, unsafe_allow_html=True)

# Content wrapped in a scrollable container
st.markdown("<div style='overflow-y: auto; max-height: 600px;'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Welcome to the World of Pokémon! Introducing the Pokédex</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>As a new Trainer embarking on your Pokémon journey, one of the most essential tools at your disposal is the Pokédex. More than just a simple encyclopedia, the Pokédex is your trusty companion, providing invaluable information about the diverse and fascinating creatures known as Pokémon.</h3>", unsafe_allow_html=True)
st.write("")
st.markdown("<h2 style='text-align: center;'>What is the Pokédex?</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>The Pokédex is a handheld electronic device that catalogs information about Pokémon species encountered throughout your adventures. It serves as a comprehensive database, containing details such as a Pokémon's name, type, abilities, habitat, evolution chain, and much more. Whether you're a seasoned Trainer or just starting your journey, the Pokédex is an indispensable resource for learning about the Pokémon world.</h3>", unsafe_allow_html=True)
st.write("")
st.markdown("<h2 style='text-align: center;'>Features of the Pokédex</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Species Information: Each entry in the Pokédex provides a wealth of information about a specific Pokémon species. From basic details like height and weight to more in-depth descriptions of their behavior and characteristics, the Pokédex offers a comprehensive overview of each Pokémon.</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Type and Abilities: Understanding a Pokémon's type and abilities is crucial for strategic battles. The Pokédex provides detailed information on a Pokémon's elemental type, which determines its strengths and weaknesses in battles. Additionally, it lists the abilities possessed by each Pokémon, giving Trainers insight into their unique traits and capabilities.</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Habitat and Distribution: Pokémon can be found in a wide range of habitats, from lush forests to arid deserts. The Pokédex documents the natural habitats and distribution patterns of each Pokémon species, helping Trainers locate and capture them in the wild.</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Evolution Chain: Many Pokémon have the ability to evolve into more powerful forms. The Pokédex tracks the evolutionary progression of each species, showing Trainers how they can evolve their Pokémon into stronger allies through training and experience.</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Audiovisual Encyclopedia: In addition to textual information, the Pokédex also features audiovisual elements such as cry sounds and animated sprites, bringing the world of Pokémon to life in vivid detail.</h3>", unsafe_allow_html=True)
st.write("")
st.markdown("<h2 style='text-align: center;'>Using the Pokédex</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>To access the Pokédex, simply navigate to the device's interface and search for the Pokémon you wish to learn more about. You can browse entries by name, type, or region, making it easy to find the information you need. Whether you're preparing for a Gym battle, completing your Pokédex collection, or simply satisfying your curiosity, the Pokédex is your ultimate companion on your journey to become a Pokémon Master.</h3>", unsafe_allow_html=True)

st.write("")

st.write("!!!this is just a test implementation made for fun and educational purposes!!!")
st.write("Pokemon trainers, point the camera of your device at a Pokémon to get its details")

col1, col2, col3 = st.columns([2.5, 1, 2])
with col2:
    scan_button_pressed = st.button("SCAN")
pic_check=False
if scan_button_pressed:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    model = YOLO("best.pt")
    while cap.isOpened() and not stop_button_pressed and not pic_check:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result, conf = detect(frame, model)

        if result in labels:
            if conf > 0.9:
                st.write(descriptions[labels[result]])
                pic_check=True
                engine = pyttsx3.init()
                volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
                engine.setProperty('volume',0.5)    # setting up volume level  between 0 and 1
                engine.say(descriptions[labels[result]])
                engine.runAndWait()
                # break
        frame_placeholder.image(frame, channels="RGB")
        if pic_check:
            frame_placeholder.image(Images[result], channels="RGB")
        if stop_button_pressed:
            cap.release()
    cap.release()
