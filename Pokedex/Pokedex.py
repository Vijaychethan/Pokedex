import ultralytics
from ultralytics import YOLO
import cv2
import torch
import streamlit as st
import pyttsx3

device = 'cuda'

labels = {0: 'pikachu', 1: 'charmander', 2: 'bulbasaur', 3: 'squirtle', 4: 'eevee', 5: 'other', 6: 'jigglypuff'}
Images = {0: 'Test_Images\Pikachu.png', 1: 'Test_Images\charmander.jpg', 2: 'Test_Images\bulbasaur.jpg', 3: 'Test_Images\ssquirtle.jpg', 4: 'Test_Images\eevee.jpg', 5: 'Test_Images\ash.jpg', 6: 'Test_Images\jigglipuff.jpg'}
st.set_page_config(layout="wide")
descriptions = {
    "pikachu": "Pikachu is a short, chubby rodent Pokémon. It is covered in yellow fur with two horizontal brown stripes on its back. It has a small mouth, long, pointed ears with black tips, and brown eyes. Each cheek is a red circle that contains a pouch for electricity storage. It has short forearms with five fingers on each paw, and its feet each have three toes. At the base of its lightning bolt-shaped tail is a patch of brown fur. A female will have a V-shaped notch at the end of its tail, which looks like the top of a heart. It is classified as a quadruped, but it has been known to stand and walk on its hind legs. Therefore Pikachu is a facultative biped.",
    "charmander": "Charmander is a Fire type Pokémon introduced in Generation 1.Charmander is a bipedal, reptilian Pokémon. Most of its body is colored orange, while its underbelly is light yellow and it has blue eyes. It has a flame at the end of its tail, which is said to signify its health.",
    "bulbasaur": "Bulbasaur is a Grass/Poison type Pokémon introduced in Generation 1.Bulbasaur is a small, mainly turquoise amphibian Pokémon with red eyes and a green bulb on its back. It is based on a frog/toad, with the bulb resembling a plant bulb that grows into a flower as it evolves.",
    'squirtle': "Squirtle is a Water type Pokémon introduced in Generation 1.Squirtle is a bipedal, reptilian Pokémon. It has a blue body with purple eyes, a light brown belly, and a tough red-brown shell on its back. It has a long tail that curls into a spiral.",
    'eevee': "Eevee is a small, mammalian, quadrupedal Pokémon with primarily brown fur. The tip of its bushy tail and its large furry collar are cream-colored. It has short, slender legs with three small toes and a pink paw pad on each foot. Eevee has brown eyes, long pointed ears with dark brown interiors, and a small black nose.Eevee is rarely found in the wild and is mostly only found in cities and towns. It is said to have an irregularly shaped genetic structure that is easily influenced by its environment. This allows it to adapt to a variety of habitats by evolving. Eevee can potentially evolve into eight different evolutions. Eevee can also start to adopt the face of the Trainer that owns it. Eevee's genes are believed to have the key to solving the mysteries of Pokémon evolution. It's shown  that once an Eevee evolves into one of its eight evolved forms, their evolution cannot be changed especially with an evolution stone.",
    'jigglypuff': "Jigglypuff is a pink Pokémon with a spherical body. It has pointed ears with black insides and large, blue eyes. It has small, stubby arms and slightly longer feet. On top of its head is a curled tuft of fur. Its body is filled with air and, as seen in Pokémon Stadium, Jigglypuff can deflate until it is flat. It can float by drawing extra air into its body, as demonstrated in Super Smash Bros.Jigglypuff can use its eyes to mesmerize opponents. It has a large lung capacity, exceeding most other Pokémon. Once it has an opponent's attention, Jigglypuff will inflate its lungs and begin to sing a soothing and mysterious lullaby. This melody can cause anyone who listens to become sleepy. If the opponent resists falling asleep, Jigglypuff will endanger its own life by continuing to sing until it runs out of air. It will continue to sing until the opponent is asleep. It can adjust the wavelength of its voice to match the brain waves of someone in a deep sleep. This helps ensure drowsiness in its opponents. Its vocal range exceeds 12 octaves, but its skill depends on the individual. Its song varies by region, and in some areas, it sounds like shouting. Jigglypuff can mostly be found in lush green plains and grassy meadows. Scream Tail shares a resemblance to Jigglypuff. It is believed Scream Tail is Jigglypuff's ancestor from 1,000,000,000 years ago. As mentioned in Pokémon Sleep, Jigglypuff is known to sing even when sleeping.["
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
        frame_placeholder.image(frame, channels="RGB")
        if result in labels:
            if conf > 0.9:
                cap.release()
                st.write(descriptions[labels[result]])
                pic_check=True
                
                if pic_check:
                    frame_placeholder.image(Images[result], channels="RGB")
                engine = pyttsx3.init()
                volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
                engine.setProperty('volume',1)    # setting up volume level  between 0 and 1
                engine.say(descriptions[labels[result]])
                engine.runAndWait()
                break
        if stop_button_pressed:
            engine.stop()
            cap.release()
    cap.release()
