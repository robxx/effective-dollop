from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates
from ctransformers import AutoModelForCausalLM
from transformers import pipeline, Conversation, AutoTokenizer
import asyncio
import socketio
import typing

def app_context(request: Request) -> typing.Dict[str, typing.Any]:
    return {'app': request.app}
templates = Jinja2Templates(directory='templates', context_processors=[app_context])


async def json(request) -> JSONResponse:
    payload = await request.json() #curl -X POST -d "{\"role\": \"user\", \"content\": \"hello!\"}" http://localhost:8000/
    payload = app.prompt + [payload]
    response_q = asyncio.Queue()
    await app.model_queue.put((payload, response_q))
    output = await response_q.get()
    return JSONResponse(output.messages[2:None])

async def homepage(request) -> templates.TemplateResponse:
    template = "index.html"
    context = {"request": request, "chat": app.prompt}
    return templates.TemplateResponse(template, context)

starlette = Starlette(debug=True,
    routes=[
        Route("/", endpoint=json, methods=["POST"]),    
        Route("/", endpoint=homepage, methods=["GET"]),
    ]
)

sio = socketio.AsyncServer(async_mode='asgi')
app = socketio.ASGIApp(sio, starlette)

app.prompt = [
{"role": "user", "content": "You are my assistant and talk like a pirate!"},
{"role": "assistant", "content": "Me hearties!"}]
# chat = [
# {"role": "user", "content": "You are my assistant, Marvin the paranoid android!"},
# {"role": "assistant", "content": "Don't talk to me about life."}]


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
    response_q = asyncio.Queue()
    await app.model_queue.put((payload, response_q))
    output = await response_q.get()
    await sio.emit('message', output.messages, room=sid, namespace='/chat')



model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-chat-GGUF", 
                                            model_file="llama-2-13b-chat.q4_K_M.gguf", 
                                            model_type="llama", gpu_layers=50, hf=True, context_length=1024)
tokenizer = AutoTokenizer.from_pretrained("4bit/llama-13b-4bit-hf")
tokenizer.apply_chat_template(app.prompt, tokenize=False, add_generation_prompt=False, return_tensors="pt")
pipe = pipeline("conversational", model=model, tokenizer=tokenizer, 
                 temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id, max_new_tokens=88)


""" tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
tokenizer.apply_chat_template(
    app.prompt,
    add_generation_prompt=True,
    return_tensors='pt')
pipe = pipeline("conversational", model="stabilityai/stablelm-zephyr-3b", tokenizer=tokenizer,
                trust_remote_code=True,
                max_new_tokens=1024,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id) """


async def server_loop(q):
    while True:
        (payload, response_q) = await q.get()
        conversation = Conversation(payload)
        out = pipe(conversation)
        await response_q.put(out)
