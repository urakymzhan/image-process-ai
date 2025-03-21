import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn = greet,
    inputs = ["text", "slider"],
    outputs=["text"]
)

demo.launch()

# The BLIP (Bootstrapped Language Image Pretraining) model can generate captions for images. 
