import streamlit as st

from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer,
)

from lib.tokenization_kobert import KoBertTokenizer

import easyocr as ocr  #OCR
import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 

#title
st.title("의약품 번역기")

#subtitle
st.markdown("## Optical Character Recognition")

st.markdown("")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


@st.cache_data
def load_model(): 
    reader = ocr.Reader(['ko', 'en'],model_storage_directory='.')
    return reader 

reader = load_model() #load model

result_text = []

if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("ðŸ¤– AI is at Work! "):
        

        result = reader.readtext(np.array(input_image))

       # result_text = [] #empty list for results


        for text in result:
            result_text.append(text[1])

        st.write(result_text)
    #st.success("Here you go!")
    st.balloons()
else:
    st.write("Upload an Image")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'tokenizer' not in st.session_state:
    src_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    trg_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    st.session_state.tokenizer = src_tokenizer, trg_tokenizer
else:
    src_tokenizer, trg_tokenizer = st.session_state.tokenizer

@st.cache_data
def get_model(bos_token_id):
    model = EncoderDecoderModel.from_pretrained('dump/best_model')
    model.config.decoder_start_token_id = bos_token_id
    model.eval()
    #model.cuda()
    model.to(device)

    return model

model = get_model(trg_tokenizer.bos_token_id)

#st.title("한-영 번역기")
#st.subheader("한-영 번역기에 오신 것을 환영합니다!")

#kor = st.text_area("입력", placeholder="번역할 한국어")

# if st.button("번역!", help="해당 한국어 입력을 번역합니다."):
#     embeddings = src_tokenizer(kor, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
#     embeddings = {k: v.cuda() for k, v in embeddings.items()}
#     output = model.generate(**embeddings)[0, 1:-1].cpu()
#     st.text_area("출력", value=trg_tokenizer.decode(output), disabled=True)

value_list=[]
if st.button("번역!", help="해당 한국어 입력을 번역합니다."):

    for kor in result_text:
        embeddings = src_tokenizer(kor, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        embeddings = {k: v.to(device) for k, v in embeddings.items()}
        output = model.generate(**embeddings)[0, 1:-1].cpu()
        #st.text_area("출력", value=trg_tokenizer.decode(output), disabled=True)
        st.write(trg_tokenizer.decode(output))
        value_list.append(trg_tokenizer.decode(output))

st.text_area("출력", value=value_list, disabled=True)
st.balloons()
