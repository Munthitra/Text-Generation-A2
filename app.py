import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import torch
from torchtext.data.utils import get_tokenizer
from model import LSTMLanguageModel

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)

            # prediction: [batch size, seq len, vocab size]
            # prediction[:, -1]: [batch size, vocab size] # probability of the last vocab

            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            while prediction == vocab['<unk>']:  # if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:  # if it is eos, we stop
                break

            indices.append(prediction)  # autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


vocab_size = 10294
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65
device = torch.device('cpu')

# Load the trained model
with open("best-val-lstm_lm.pt", 'rb') as f:
    model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
    model.load_state_dict(torch.load(f))
    model.eval()

tokenizer = get_tokenizer('basic_english')
vocab = torch.load("vocab")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Text Generation Web App"),
    
    dcc.Input(id='user_input', type='text', value="Hello World", placeholder='Enter a prompt...'),
    dcc.Input(id='temperature', type='number', min=0.1, max=1.0, step=0.1, value=0.7),
    
    html.Button('Generate Text', id='generate_button', n_clicks=0),
    
    html.H2("Generated Text:"),
    html.P(id='generated_text'),
])

@app.callback(
    Output('generated_text', 'children'),
    [Input('generate_button', 'n_clicks')],
    [dash.dependencies.State('user_input', 'value'),
     dash.dependencies.State('temperature', 'value')]
)
def generate_text(n_clicks, user_input, temperature):
    generated_text = generate(user_input, 100, temperature, model, tokenizer, vocab, device)
    return ' '.join(generated_text)



if __name__ == '__main__':
    app.run_server(debug=True)