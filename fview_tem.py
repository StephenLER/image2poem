import gradio as gr
from image_process import image_to_poem
from BERTMatching import translation_to_poem
from langchainModel import create_poem

def generate_image_description(image):
    sentence = image_to_poem(image)
    return sentence  


def sentence2poem(sentence):
    poem = translation_to_poem(sentence)
    return poem  



def image_to_description_and_poem(image):
    description = generate_image_description(image)
    poem = sentence2poem(description)
    model_poem = create_poem(description)
    poem=list_interface(poem)
    return description, poem,model_poem

def list_interface(selected_items):
    # 在输出中使用 gr.HTML 来显示选定的列表项
    # output_html = "<ul>"
    # for item in selected_items:
    #     output_html += f"<li>{item}</li>"
    # output_html += "</ul>"
    output_html="\n".join(selected_items)

    return output_html

# 定义界面
with gr.Blocks() as demo:
    gr.Markdown("图片转古诗")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(tool="editor")
            generate_button = gr.Button("生成描述和古诗")
        with gr.Column():
            description_output = gr.Textbox(label="图片描述")
            
            poem_output = gr.Textbox(label="古诗(返回前十首最相近古诗)")
            # poem_output=gr.HTML(label="古诗")
            model_poem_output = gr.Textbox(label="现代诗")

    generate_button.click(
        fn=image_to_description_and_poem,
        inputs=image_input,
        outputs=[description_output, poem_output,model_poem_output]
    )

# 运行应用
demo.launch()

