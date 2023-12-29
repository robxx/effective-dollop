from starlette.applications import Starlette
from starlette.routing import Route
from starlette.templating import Jinja2Templates
from ctransformers import AutoModelForCausalLM as cAutoModelForCausalLM
from transformers import pipeline, Conversation, AutoTokenizer, TextIteratorStreamer
import socketio
import asyncio
from mistune import Markdown
from mistune import HTMLRenderer
from pygments import highlight
from pygments.lexers import guess_lexer
from pygments.formatters import html

from threading import Thread

templates = Jinja2Templates(directory='templates')

async def homepage(request) -> templates.TemplateResponse:
    template = "code_llama.html"
    formatter = html.HtmlFormatter()
    return templates.TemplateResponse(template, context={'request': request,
                                    'highlight_styles': formatter.get_style_defs()})

starlette = Starlette(debug=True,
    routes=[
        Route("/", endpoint=homepage, methods=["GET"]),
    ]
)

class SyntaxHighlightRenderer(HTMLRenderer):
    def block_code(self, code):
        lexer = guess_lexer(code)
        formatter = html.HtmlFormatter(lineseparator="<br>")
        return highlight(code, lexer, formatter)

def render_markdown(markdown_string):
    renderer = SyntaxHighlightRenderer()
    markdown = Markdown(renderer=renderer)
    return markdown(markdown_string)

sio = socketio.AsyncServer(async_mode='asgi')
app = socketio.ASGIApp(sio, starlette)

@starlette.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))

@sio.event(namespace='/chat')
def connect(sid, environ, auth):
    print('connect ', sid)

@sio.event(namespace='/chat')
def disconnect(sid):
    print('disconnect ', sid)

@sio.on('question', namespace='/chat')
async def question(sid, payload):
    await app.model_queue.put((payload, sid))
    


model = cAutoModelForCausalLM.from_pretrained("TheBloke/CodeLlama-13B-Instruct-GGUF", 
                                            model_file="codellama-13b-instruct.Q5_K_M.gguf", 
                                            model_type="llama", gpu_layers=50, hf=True, context_length=1024)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
pipe = pipeline("conversational", model=model, tokenizer=tokenizer,
                do_sample=True, num_beams=1, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1024)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

async def server_loop(q):
    while True:
        (payload, sid) = await q.get()
        conversation = Conversation(payload)
        generation_kwargs = dict(conversations=conversation, streamer=streamer)
        thread = Thread(target=pipe, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            output = render_markdown(generated_text)
            await sio.emit('message', output, namespace='/chat', room=sid) 
        
        print(sid, generated_text)
        