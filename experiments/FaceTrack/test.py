from FaceTrack.Face_detect import detect_face
import pandas as pd

df = pd.DataFrame(columns=['Photo_ind', 'x', 'y', 'w', 'h'])
df.to_csv('/home/artem/grive/HSE/3course/YDF_proj/df2try.csv')
detect_face('/home/artem/grive/HSE/3course/YDF_proj/vid/Frames/Putin', '/home/artem/grive/HSE/3course/YDF_proj/df2try.csv')
#detect_face('/home/artem/grive/HSE/3course/YDF_proj/vid/Frames/Putin', '/home/artem/grive/HSE/3course/YDF_proj/df.csv')