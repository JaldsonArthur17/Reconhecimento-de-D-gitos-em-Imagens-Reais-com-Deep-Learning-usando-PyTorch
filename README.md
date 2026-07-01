> ### Este projeto será continuado ---> uploads de relatórios de testes e aprimoramentos serão feitos futuramente

---

# Reconhecimento de Dígitos em Imagens Reais com Deep Learning usando PyTorch

## Descrição

Desenvolvimento, avaliação e evolução de uma IA capaz de reconhecer e classificar
dígitos (0 a 9) em imagens reais do Google Street View, utilizando o dataset
**SVHN Format 2** (imagens 32x32 pixels). O projeto parte de um modelo simples e
o aprimora de forma metódica, incorporando técnicas avançadas de Deep Learning
para lidar com ruído, iluminação variada e dígitos distratores nas bordas.

---

## Etapas

1. **Baseline CNN** ---> Rede convolucional simples (2 camadas) como ponto de partida.
   Acurácia inicial: **87,88%**

2. **DeepCNN** ---> Expansão para 4 camadas convolucionais com Batch Normalization
   (estabilização do treino) e Dropout (prevenção de overfitting).
   Acurácia: **94,23%**

3. **RandAugment** ---> Distorções aleatórias nas imagens de treino (rotação, corte,
   etc.) para forçar a rede a focar na essência dos dígitos.
   Acurácia: **95,30%**

4. **FixMatch** ---> Aprendizado semi-supervisionado com até 300 mil imagens não
   rotuladas, usando máscara de confiança (threshold 95%–98%) para geração de
   pseudo-rótulos. Pico absoluto: **95,83%**
   > Comprovamos o *Distribution Mismatch* na prática: ao forçar 400k+ imagens
   > extras, a rede satura e a acurácia recua para 95,76%.

---

## Avaliação

- **Rigor Metodológico** ---> Checkpointing com base em 10% de dados de validação
  isolados; avaliação final em conjunto de teste nunca visto pela rede
- **Relatório de Classificação** ---> Precision, Recall e F1-Score por dígito
- **Matriz de Confusão** ---> Mapa de calor para identificar confusões específicas
  (ex: dígito "3" confundido com "8" por baixa resolução)

---

## Tecnologias

- **Linguagem e Framework** ---> Python e PyTorch (`torchvision`, `torch.nn`, `torch.optim`)
- **Ambiente e Hardware** ---> Google Colab com GPU NVIDIA T4
- **Análise e Visualização** ---> Scikit-Learn, Matplotlib e Seaborn
- **Apresentação** ---> Dashboard interativo com Streamlit

---

## Como ver o relatório interativo?

1. Instale o Streamlit via terminal:
```bash
   pip install streamlit
```

2. Baixe o arquivo `app.py`

3. No terminal, execute:
```bash
   streamlit run app.py
```
