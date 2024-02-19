
import google.generativeai as genai
import PIL.Image
from IPython.display import display
from IPython.display import Markdown
import pathlib
import textwrap

img = PIL.Image.open('download.jpg')
genai.configure(api_key ="API key is private :)" )
defaults={
  'model': 'models/text-bison-001',
  'temperature' : 0.1,
  # 'candidate-count' : 1,
  'top_k' : 40,
  # 'top_p ': 0.95,
  'max_output_tokens':1024,
}
l = []
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
model = genai.GenerativeModel('gemini-pro-vision')
prompt = "You are a caption generator tool for an image. A list of elememnts provided which are all the subjects of the image. You must generate a one sentence aesthetic caption suitable for n instagram story. list =  "
prompt = prompt.append(l)
response = model.response(prompt)
response.resolve()
k = 0

print(response.text)
