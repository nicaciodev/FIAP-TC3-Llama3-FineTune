# FIAP-TC3-Llama3-FineTune - LLM para descrições de produtos
![Static Badge](https://img.shields.io/badge/Vers%C3%A3o-1.0-blue) ![GitHub](https://img.shields.io/github/license/nicaciodev/FIAP-TC3-Llama3-FineTune) ![Static Badge](https://img.shields.io/badge/Data-25%2F09%2F2025-green)

___

#### Tech-Challenge da Fase 03 da Post-Tech (FIAP)

>>>> "... executar o fine-tuning de
um foundation model (Llama, BERT, MISTRAL etc.), utilizando o dataset "The
AmazonTitles-1.3MM". O modelo treinado deverá:

>>>> ● Receber perguntas com um contexto obtido por meio do arquivo json
“trn.json” que está contido dentro do dataset.

>>>> ● A partir do prompt formado pela pergunta do usuário sobre o título do produto, o modelo deverá gerar uma resposta baseada na pergunta do usuário trazendo como resultado do aprendizado do fine-tuning os dados da sua descrição."
>>>> 
>>>> (FIAP, Pos-Tech, Fase3, Tech-Challenge, O problema)*

#### [ RM363334 ]

#### Robson Nicácio R. dos Santos
___

## Objetivo
> Analisar o dataset fornecido eliminando dados desnecessários e ruídos que possam comprometer o fine-tuning.
>
> Executar o fine-tuning de um modelo de linguagem grande (LLM).
>
> Realizar uma chamada do foundation model escolhido após o treinamento para validar os resultados.
> 
> Disponibilizar o modelo treinado no Hugging Face para uso público.
> 
> Documentar todo o processo de desenvolvimento e as decisões de engenharia tomadas.


## Documentação Completa

> A documentação completa se encontra no diretório [Docs](https://github.com/nicaciodev/FIAP-TC3-Llama3-FineTune/tree/main/docs) deste projeto nos formatos: [Markdown](https://github.com/nicaciodev/FIAP-TC3-Llama3-FineTune/blob/main/docs/documenta%C3%A7%C3%A3o_completa_da_codifica%C3%A7%C3%A3o.md), [PDF](https://github.com/nicaciodev/FIAP-TC3-Llama3-FineTune/blob/main/docs/documenta%C3%A7%C3%A3o_completa_da_codifica%C3%A7%C3%A3o.pdf).

## Resumo do Projeto
> Este projeto consiste no fine-tuning do modelo unsloth/llama-3-8b-bnb-4bit, uma versão otimizada do Llama-3 8B da Meta, para a tarefa específica de gerar descrições de produtos da Amazon a partir de seus títulos. Utilizando a técnica QLoRA e a biblioteca Unsloth para um treinamento eficiente, o modelo foi treinado com 50.000 exemplos do dataset "The Amazon Titles-1.3MM". O processo envolveu uma rigorosa etapa de pré-processamento dos dados para remover ruídos e a engenharia de um prompt no formato nativo do Llama 3 para garantir a correta interpretação da tarefa. O modelo final é capaz de gerar descrições relevantes e contextualmente apropriadas, que podem variar de sinopses de enredo a biografias de autores, refletindo a diversidade do dataset original.





## Questões Sobre Projeto
Durante o desenvolvimento, alguns desafios foram encontrados e superados, representando decisões de engenharia cruciais para o sucesso do projeto:

1. **Qualidade dos Dados e Texto *Boilerplate***: O dataset original continha texto repetitivo e de baixo valor semântico no final de muitas descrições (ex: "--This text refers to..."). Foi identificado que esse "lixo" estava sendo aprendido pelo modelo. A solução foi implementar uma etapa de limpeza no pré-processamento usando expressões regulares para remover esses artefatos, garantindo um treinamento com dados mais limpos e relevantes.  
2. **Formato do Prompt e Desempenho do Modelo**: Uma tentativa inicial com um template de prompt genérico resultou em respostas de baixa qualidade, com o modelo focando em metadados (como autores) em vez do conteúdo descritivo. A solução foi adotar o **template de chat nativo do Llama 3**, o que melhorou drasticamente a capacidade do modelo de entender a tarefa solicitada.  
3. **Overfitting Estrutural e Parada da Geração**: Após a limpeza dos dados, o modelo começou a gerar tokens de controle (\<|reserved\_special\_token\_...|\>) ao final de suas respostas, indicando que ele havia aprendido a estrutura do prompt de forma excessiva e não sabia onde parar. A solução definitiva foi adicionar o token \<|end\_of\_text|\> ao final de cada exemplo de treinamento, ensinando explicitamente ao modelo o sinal de finalização de um diálogo completo.  
4. **Repetição Degenerativa na Inferência**: Em alguns testes, o modelo entrou em um loop, repetindo a mesma frase várias vezes. Este problema foi solucionado na etapa de inferência, adicionando o parâmetro repetition\_penalty \= 1.15, uma técnica padrão para desencorajar o modelo de gerar sequências repetitivas e melhorar a qualidade do texto.



## Estrutura do Repositório

* `README.md`: Este arquivo, com a descrição e instruções do projeto.
* `Docs`: Documentação completa do projeto. 
* `LICENSE`: Contém a licença deste projeto.
* `Tech_Challenge_F3_para_Github.ipynb`: Todo o código utilizado para tratar o dataset, fazer o fine-tuning e inferir o mesmo.

  ### Links
* `Arquivos Originais no Google-Drive`: [https://drive.google.com/drive/folders/1kLw3Bdg4Q0d5V6rHAY_QOcTvohMXnP3t?usp=sharing](https://drive.google.com/drive/folders/1kLw3Bdg4Q0d5V6rHAY_QOcTvohMXnP3t?usp=sharing)
* `Modelo Salvo no Hugging Face`: [https://huggingface.co/robsonnicacio/llama-3-8b-amazon-descriptions-tcf3](https://huggingface.co/robsonnicacio/llama-3-8b-amazon-descriptions-tcf3)


#
## Utilização via Hugging Face

```
# Etapa 1: Instalar as bibliotecas necessárias
# Descomente a linha abaixo se estiver em um novo ambiente Colab
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList

# --- Etapa 2: Carregar o Modelo do Hugging Face Hub ---

# O nome do seu repositório no Hugging Face
nome_do_seu_modelo_no_hf = "robsonnicacio/llama-3-8b-amazon-descriptions-tcf3"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = nome_do_seu_modelo_no_hf,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Prepara o modelo para uma inferência mais rápida
FastLanguageModel.for_inference(model)
print("Modelo fine-tuned carregado com sucesso do Hugging Face Hub!")


# --- Etapa 3: Configurar a Inferência de Forma Robusta ---

# Template de prompt para a inferência
LLAMA3_INFERENCE_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Com base no título do produto, gere a sua descrição.
Título: {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Definição dos Critérios de Parada (Stopping Criteria) para uma saída limpa
stop_tokens = ["<|eot_id|>", "<|end_of_text|>"]
stop_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in stop_tokens]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])


# --- Etapa 4: Loop de Teste Interativo ---
while True:
    titulo_produto = input("Digite o título do produto (ou 'sair' para terminar): ")

    if titulo_produto.lower() == 'sair':
        print("Encerrando o teste.")
        break

    prompt = LLAMA3_INFERENCE_PROMPT.format(titulo_produto)
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    # Usamos o TextStreamer para ver a resposta sendo gerada em tempo real
    # skip_special_tokens=True limpa a saída de tokens como <|eot_id|>
    text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    _ = model.generate(
        **inputs, 
        streamer = text_streamer, 
        max_new_tokens = 256,
        stopping_criteria = stopping_criteria, # Usa o critério de parada robusto
        repetition_penalty = 1.15
    )

    print("\n" + "="*50 + "\n")
```


#
## Conclusões
> O projeto demonstrou com sucesso a eficácia do fine-tuning com QLoRA para especializar um LLM de grande escala em uma tarefa de nicho. O modelo final é capaz de gerar descrições relevantes e contextualmente ricas, aprendendo os diferentes padrões presentes no dataset da Amazon, como sinopses de enredo e biografias de autores.

> O processo reforçou a importância crítica do pré-processamento de dados e da engenharia de prompts para o sucesso de um projeto de fine-tuning. Os desafios encontrados, como a limpeza de artefatos de texto (boilerplate) e o controle da geração para evitar overfitting estrutural, foram superados com técnicas específicas e iterativas.

> O resultado é um modelo robusto e bem-comportado que cumpre todos os requisitos do Tech Challenge, servindo como um estudo de caso prático sobre o pipeline completo de especialização de um foundation model.




## Autor
___
| [<img src="https://avatars.githubusercontent.com/u/136343808?v=4" width=115><br><sub>Robson Nicácio</sub>](https://github.com/nicaciodev) |
| :---: |
