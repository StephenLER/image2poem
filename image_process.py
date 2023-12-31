from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline,AutoModel,MarianMTModel,AutoProcessor, Blip2ForConditionalGeneration
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available () else "cpu"
from PIL import Image

blip_path="../blip2-opt-2.7b"


#初始化，加载模型
# tokenizer = AutoTokenizer.from_pretrained("./Helsinki-NLP---opus-mt-en-zh")
# model = AutoModelForSeq2SeqLM.from_pretrained("./Helsinki-NLP---opus-mt-en-zh")
translation = pipeline("translation",model="./Helsinki-NLP---opus-mt-en-zh",tokenizer="./Helsinki-NLP---opus-mt-en-zh")
processor = AutoProcessor.from_pretrained (blip_path)
model = Blip2ForConditionalGeneration.from_pretrained (blip_path, torch_dtype=torch.float16)


class Image2Translation(nn.Module):
    def __init__(self, model1,processor, pipeline2):
        super(Image2Translation, self).__init__()
        self.pipeline2 = pipeline2
        self.processor=processor
        self.model1 = model1

    def forward(self, x):       
         # 通过第1个模型进行前向传播
        output1 = self.model1.generate (**x, max_new_tokens=100)
        generated_text = "there "+self.processor.batch_decode (output1, skip_special_tokens=True)[0].strip ()
        # 通过第2个模型进行前向传播
        output2 = self.pipeline2(generated_text)[0]['translation_text']

        return output2

def image_to_poem(photo):
    image = Image.fromarray (photo).convert ('RGB')  
    #display (image.resize ((596, 437)))
    image2translation = Image2Translation(model,processor, translation)
    image2translation.to(device)
    image2translation.eval()
    prompt="the specific description of the painting is: there"
    inputs = processor(image, text=prompt, return_tensors="pt").to (device, torch.float16)

    return image2translation(inputs)

def main():
    photo = "./photo/2.png"
    poem = image_to_poem(photo)
    print(poem)
if __name__ == "__main__":
    main()