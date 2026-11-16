# Processamento Digital de Imagens

Aplicação didática com interface web para aplicar e comparar operações de processamento
em imagens PNG em escala de cinza de 8 bits (`L`) e RGB de 24 bits (`RGB`). A imagem
original permanece visível, cada saída pode ser baixada como PNG e operações incompatíveis
com o tipo carregado não aparecem no seletor.

## Executar

Requer Python 3.10 ou mais recente.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

O Streamlit informa o endereço local (normalmente `http://localhost:8501`). Carregue pela
barra lateral um PNG que já esteja em modo `L` ou `RGB`; arquivos paletizados, RGBA e outros
formatos são recusados explicitamente para não mascarar uma conversão de profundidade.

## Operações disponíveis

- decomposição dos canais RGB e HSV;
- limiarização, transformações logarítmica e de potência;
- equalização com histogramas antes/depois (somente escala de cinza);
- fatiamento por intensidade, com preservação opcional do fundo;
- média gaussiana, mediana, mínimo, máximo e mediana adaptativa;
- máscara de aguçamento, realce Laplaciano e gradiente de Sobel;
- passa-baixa e passa-alta Gaussianos e Butterworth no domínio da frequência;
- ruído gaussiano, sal, pimenta e sal-e-pimenta, com semente reproduzível.

As transformações de intensidade e os filtros são aplicados separadamente a cada canal de
uma imagem RGB. Limiarização e fatiamento produzem uma saída de intensidade em escala de
cinza. A equalização foi restringida a imagens `L`, conforme o exemplo explícito do
enunciado.

## Decisões para pontos ambíguos

- **Tamanho de janela de Sobel e Laplaciano:** as máscaras diferenciais canônicas são 3×3.
  Para cumprir o controle de janela exigido para filtros espaciais sem inventar coeficientes,
  valores maiores que 3 aplicam uma pré-suavização uniforme com a janela escolhida e então
  a máscara 3×3. A interface expõe 3, 5, ..., 15.
- **Bordas:** filtros espaciais usam reflexão, evitando faixas artificiais de zeros.
- **Passa-alta:** a resposta, naturalmente centrada em zero, recebe nível visual 127,5 antes
  da conversão para PNG para preservar componentes negativas na visualização.
- **Ruído gaussiano:** “intensidade e distribuição” foi interpretado como desvio padrão e
  média; a semente também é configurável para permitir repetir a demonstração.
- **Sal e pimenta:** “distribuição” foi interpretado como a fração de sal no total de pixels
  corrompidos.

## Testes

Instale as dependências de desenvolvimento e rode:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

Os testes cobrem leitura/exportação, compatibilidade de modos, formato e faixa das saídas,
histogramas, famílias de filtros espaciais e de frequência e reprodutibilidade dos ruídos.

## Estrutura

```text
app.py                 interface Streamlit
src/pdi/io.py          validação e exportação PNG
src/pdi/operations.py  núcleo numérico independente da interface
tests/                 testes automatizados
```
