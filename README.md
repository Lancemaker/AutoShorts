# Shorts Automation

Pipeline local para localizar momentos de gameplay, montar uma fila humana de
revisão, renderizar os cortes em vertical e gerar uma planilha Excel.

Ele não publica nada. A decisão final continua sendo sua:

- PAYOFF vira **Alto potencial**;
- UNCERTAIN vira **Em revisão**;
- apenas NO_PAYOFF deixa de ser renderizado.

## Estrutura

~~~text
shorts_automation/
├── main.py      # comandos e fluxo geral
├── config.py    # modelos, crop, caminhos e faixas de áudio
├── media.py     # FFmpeg/FFprobe, extração, duração e render
├── audio.py     # picos de áudio do jogo
├── speech.py    # Whisper e cache da transcrição
├── vision.py    # frames, Ollama/Gemma e validação visual
├── review.py    # fila humana, sobreposições e Excel
└── README.md
~~~

Os diretórios abaixo são criados ou usados automaticamente:

~~~text
videos/input/   # gravações de origem
videos/output/  # Shorts MP4 e planilha Excel
work/           # WAVs, frames e caches JSON
~~~

Se videos/input ainda não existir, o programa também aceita vídeos diretamente
em videos/. Os formatos aceitos são MKV, MP4, MOV, WEBM e AVI.

## Pré-requisitos

- Python com Faster-Whisper, Requests e OpenPyXL;
- FFmpeg e FFprobe disponíveis no PATH;
- NVIDIA/CUDA funcionando para Faster-Whisper;
- Ollama em execução com o modelo definido em config.py
  (por padrão, gemma3:4b).

Exemplo de instalação das dependências Python:

~~~powershell
python -m pip install faster-whisper requests openpyxl
~~~

## Faixas de áudio

Os índices são os índices zero-based do FFmpeg:

| Faixa gravada | Índice |
|---|---:|
| primeira faixa | 0 |
| segunda faixa | 1 |
| terceira faixa | 2 |

Não há separação esquerda/direita. Para análise, uma faixa inteira é
temporariamente convertida em WAV mono; no MP4 final, o programa copia a faixa
original escolhida.

Há três papéis independentes:

| Parâmetro | Uso |
|---|---|
| --game-track | encontra picos de ação no áudio só do jogo |
| --voice-track | transcreve a faixa do microfone com Whisper |
| --render-track | preserva a faixa escolhida no MP4 final |

No seu layout atual de três faixas — 0: jogo+microfone, 1: microfone,
2: jogo — execute:

~~~powershell
python main.py --game-track 2 --voice-track 1 --render-track 0
~~~

Assim, Whisper recebe só sua voz, os picos vêm do jogo e o Short final mantém
jogo + microfone. O comportamento antigo de duas faixas segue preservado:
game=0, voice=1 e render=1.

Para conferir os índices de uma gravação antes de rodar:

~~~powershell
ffprobe -v error -select_streams a -show_entries stream=index:stream_tags=title -of json "videosinputgravacao.mkv"
~~~

## Como executar

Coloque as gravações em videos/input/ e, dentro desta pasta, execute:

~~~powershell
python main.py --game-track 2 --voice-track 1 --render-track 0
~~~

Opções úteis:

| Opção | Efeito |
|---|---|
| --no-render | cria a análise e a fila JSON, sem MP4 nem Excel |
| --force-transcribe | ignora a transcrição Whisper em cache |
| --refresh-visual | recria os caches de frames e Gemma |
| --coarse-every 30 | amostra um frame de contexto a cada 30 segundos |

## Onde mudar configurações

Use config.py para alterar modelos, URL do Ollama, crop vertical fixo, caminhos,
valores padrão de faixa e limites da análise. O crop é deliberadamente central
em 9:16, sem blur de fundo e sem um agente escolhendo o enquadramento.

## Fluxo

1. media.py verifica as faixas e extrai WAVs de análise.
2. speech.py transcreve a voz em inglês com Whisper na GPU.
3. audio.py encontra picos de energia da faixa do jogo.
4. vision.py cria o mapa visual e pede descrições em texto ao Gemma local.
5. review.py ordena os candidatos, marca sobreposições e monta a fila humana.
6. media.py renderiza apenas Alto potencial e Em revisão.
7. review.py cria o arquivo Excel com as abas Cortes e Evidências.

Os caches permitem rodar de novo sem pagar o custo de Whisper e Gemma toda vez.
