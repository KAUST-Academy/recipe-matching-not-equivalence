---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:9921
- loss:CachedMultipleNegativesRankingLoss
base_model: Qwen/Qwen3-Embedding-4B
widget:
- source_sentence: 'Problem:

    Find three different polynomials $P(x)$ with real coefficients such that $P\left(x^{2}+1\right)=P(x)^{2}+1$
    for all real $x$.'
  sentences:
  - 'Problem:

    Zij $ABC$ een driehoek, punt $P$ het midden van $BC$ en punt $Q$ op lijnstuk $CA$
    zodat $- 2 \left|{A Q}\right| + \left|{C Q}\right| = 0$. Zij $S$ het snijpunt
    van $BQ$ en $AP$. Bewijs dat $|AS|=|SP|$.'
  - 'Problem:

    Find three different polynomials $P(w)$ with real coefficients such that $P\left(w^{2}+1\right)=P(w)^{2}+1$
    for all real $w$.'
  - 'Problem:


    Let $M$ be an interior point of the triangle $A U W$ with angles $\varangle U
    A W = 70^{\circ}$ and $\varangle A U W = 80^{\circ}$. If $\varangle A W M = 10^{\circ}$
    and $\varangle W U M = 20^{\circ}$, prove that $A U = M W$.'
- source_sentence: 'Prove that, for any positive integers $d$ and $m$, a polynomial
    of degree $d$ with real coefficients cannot be expressed by a product of $m$ periodic
    functions. (A function $f: \mathbb{R} \to \mathbb{R}$ is called *periodic* if
    there exists a constant $T = T(f) > 0$ such that $f(x + T) = f(x)$ for all $x
    \in \mathbb{R}$.)'
  sentences:
  - 'Prove that, for any positive integers $d$ and $m$, a polynomial of degree $d$
    with real coefficients cannot be expressed by a product of $m$ periodic functions.
    (A function $f: \mathbb{R} \to \mathbb{R}$ is called *periodic* if there exists
    a constant $W = W(f) > 0$ such that $f(w + W) = f(w)$ for all $w \in \mathbb{R}$.)'
  - 'Problem:

    Let $X=\{1,2, \ldots, 100\}$. How many functions $f: X \rightarrow X$ satisfy
    $f(w)<f(a)+(w-a)$ for all $1 \leq a<w \leq 100$?'
  - 'Let $a_1 > a_2 > \dots > a_n > 1$ be positive integers. Let $M$ denote the least
    common multiple of $a_1, a_2, \dots, a_n$. For a finite set of integers $W$, define

    $$f(W) = \min_{1 \le i \le n} \sum_{x \in W} \left\{ \frac{x}{a_i} \right\}.$$

    Here $\{u\} = u - \lfloor u \rfloor$ is the fractional part of the real number
    $u$. Put $f(\emptyset) = 0$. We say a set $W$ is *minimal*, if for any proper
    subset $Y \subsetneq W$, we have $f(Y) < f(W)$.

    Prove that, if $W$ is a minimal finite set of integers and if $f(W) \ge \frac{2}{a_n}$,
    then

    $$|W| \le f(W) \cdot M.$$

    Here for a finite set $W$, we use $|W|$ to denote the number of elements in $W$.'
- source_sentence: 'Problem:


    Déterminer le plus petit entier $n \geqslant 2$ tel qu''il existe des entiers
    strictement positifs $a_{1}, \ldots, a_{n}$ tels que

    $$

    a_{1}^{2}+\ldots+a_{n}^{2} \mid \left(a_{1}+\ldots+a_{n}\right)^{2}-1

    $$'
  sentences:
  - 'Three points $A$, $B$, $C$, are marked on the hyperbola $u = 1/w$ so that the
    triangle $ABC$ is equilateral.

    Find all possible values of the product of the sum of abscissae and the sum of
    ordinates of the vertices of $ABC$.'
  - 'Problem:


    Déterminer le plus petit entier $2 \leq n$ tel qu''il existe des entiers strictement
    positifs $a_{1}, \ldots, a_{n}$ tels que

    $$

    a_{1}^{2}+\ldots+a_{n}^{2} \mid \left(a_{1}+\ldots+a_{n}\right)^{2}-1

    $$'
  - 'Problem:


    Déterminer le plus petit entier $n \geq 3$ tel qu''il existe des entiers strictement
    positifs $a_{1}, \ldots, a_{n}$ tels que

    $$

    a_{1}^{2}+\ldots+a_{n}^{2} \mid \left(a_{1}+\ldots+a_{n}\right)^{2}-1

    $$'
- source_sentence: 'Problem:

    Find all positive integers $n$ such that the number $A_{n} = \frac{2^{4n+2} +
    1}{65}$ is


    a) an integer;


    b) a prime.'
  sentences:
  - 'Problem:

    Find all positive integers $n$ such that the number $A_{n} = \frac{3^{4 n + 3}}{65}
    + \frac{1}{65}$ is


    a) an integer;


    b) a prime.'
  - 'Problem:

    Find all positive integers $n$ such that the number $- \frac{2^{4 n + 2}}{65}
    + A_{n} = \frac{1}{65}$ is


    a) an integer;


    b) a prime.'
  - 'Problem:

    Determine all integers $w \geqslant 2$ such that there exists a permutation $x_{0},
    x_{1}, \ldots, x_{w-1}$ of the numbers $0,1, \ldots, w-1$ with the property that
    the $w$ numbers

    $$x_{0}, \quad x_{0}+x_{1}, \quad \ldots, \quad x_{0}+x_{1}+\ldots+x_{w-1}$$

    are pairwise distinct modulo $w$.'
- source_sentence: 'Problem:


    We say that a pile is a set of four or more nuts. Two persons play the following
    game. They start with one pile of $n \geq 4$ nuts. During a move a player takes
    one of the piles that they have and split it into two non-empty subsets (these
    sets are not necessarily piles, they can contain an arbitrary number of nuts).
    If the player cannot move, he loses. For which values of $n$ does the first player
    have a winning strategy?'
  sentences:
  - 'Problem:


    We say that a pile is a set of four or more nuts. Two persons play the following
    game. They start with one pile of $n \leq 4$ nuts. During a move a player takes
    one of the piles that they have and split it into two non-empty subsets (these
    sets are not necessarily piles, they can contain an arbitrary number of nuts).
    If the player cannot move, he loses. For which values of $n$ does the first player
    have a winning strategy?'
  - 'Problem:


    We say that a pile is a set of four or more nuts. Two persons play the following
    game. They start with one pile of $4 \leq n$ nuts. During a move a player takes
    one of the piles that they have and split it into two non-empty subsets (these
    sets are not necessarily piles, they can contain an arbitrary number of nuts).
    If the player cannot move, he loses. For which values of $n$ does the first player
    have a winning strategy?'
  - 'Problem:


    Gegeben sind eine natürliche Zahl $n$ und natürliche Zahlen $a_{1}, a_{2}, \ldots,
    a_{n}$. Wir erweitern die Folge periodisch durch $a_{n+i}=a_{i}$ für alle $i \geq
    1$. Nehme nun an, dass folgende zwei Bedingungen erfüllt sind:


    (i) $a_{1} \leq a_{2} \leq \cdots \leq a_{n} \leq a_{1}+n$.


    (ii) $a_{a_{i}} - n \leq i - 1$ für $i=1,2 \ldots, n$.


    Zeige, dass gilt:

    $$

    a_{1}+a_{2}+\cdots+a_{n} \leq n^{2}

    $$'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on Qwen/Qwen3-Embedding-4B
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: dev
      type: dev
    metrics:
    - type: cosine_accuracy
      value: 0.9398339986801147
      name: Cosine Accuracy
---

# SentenceTransformer based on Qwen/Qwen3-Embedding-4B

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) on the triplets and pairs datasets. It maps sentences & paragraphs to a 2560-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) <!-- at revision 5cf2132abc99cad020ac570b19d031efec650f2b -->
- **Maximum Sequence Length:** 40960 tokens
- **Output Dimensionality:** 2560 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
- **Training Datasets:**
    - triplets
    - pairs
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'Qwen3Model'})
  (1): Pooling({'embedding_dimension': 2560, 'pooling_mode': 'lasttoken', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
queries = [
    'Problem:\n\nWe say that a pile is a set of four or more nuts. Two persons play the following game. They start with one pile of $n \\geq 4$ nuts. During a move a player takes one of the piles that they have and split it into two non-empty subsets (these sets are not necessarily piles, they can contain an arbitrary number of nuts). If the player cannot move, he loses. For which values of $n$ does the first player have a winning strategy?',
]
documents = [
    'Problem:\n\nWe say that a pile is a set of four or more nuts. Two persons play the following game. They start with one pile of $4 \\leq n$ nuts. During a move a player takes one of the piles that they have and split it into two non-empty subsets (these sets are not necessarily piles, they can contain an arbitrary number of nuts). If the player cannot move, he loses. For which values of $n$ does the first player have a winning strategy?',
    'Problem:\n\nWe say that a pile is a set of four or more nuts. Two persons play the following game. They start with one pile of $n \\leq 4$ nuts. During a move a player takes one of the piles that they have and split it into two non-empty subsets (these sets are not necessarily piles, they can contain an arbitrary number of nuts). If the player cannot move, he loses. For which values of $n$ does the first player have a winning strategy?',
    'Problem:\n\nGegeben sind eine natürliche Zahl $n$ und natürliche Zahlen $a_{1}, a_{2}, \\ldots, a_{n}$. Wir erweitern die Folge periodisch durch $a_{n+i}=a_{i}$ für alle $i \\geq 1$. Nehme nun an, dass folgende zwei Bedingungen erfüllt sind:\n\n(i) $a_{1} \\leq a_{2} \\leq \\cdots \\leq a_{n} \\leq a_{1}+n$.\n\n(ii) $a_{a_{i}} - n \\leq i - 1$ für $i=1,2 \\ldots, n$.\n\nZeige, dass gilt:\n$$\na_{1}+a_{2}+\\cdots+a_{n} \\leq n^{2}\n$$',
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 2560] [3, 2560]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[ 0.7797,  0.2315, -0.1107]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Dataset: `dev`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.sentence_transformer.evaluation.TripletEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| **cosine_accuracy** | **0.9398** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Datasets

#### triplets

* Dataset: triplets
* Size: 8,837 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                               | positive                                                                             | negative                                                                             |
  |:---------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                               | string                                                                               | string                                                                               |
  | modality | text                                                                                 | text                                                                                 | text                                                                                 |
  | details  | <ul><li>min: 27 tokens</li><li>mean: 104.53 tokens</li><li>max: 345 tokens</li></ul> | <ul><li>min: 29 tokens</li><li>mean: 107.02 tokens</li><li>max: 344 tokens</li></ul> | <ul><li>min: 29 tokens</li><li>mean: 108.56 tokens</li><li>max: 344 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Problem:<br><br>Fie $N \geq 4$ un număr natural fixat.<br>Doi jucători, $A$ şi $B$, formează o mulţime ordonată $x_{1}, x_{2}, x_{3}, \ldots$, adăugând alternativ elemente: $A$ alege $x_{1}$ egal cu $1$ sau cu $-1$, apoi $B$ îl adaugă pe $x_{2}$ egal cu $2$ sau cu $-2$, apoi $A$ îl adaugă pe $x_{3}$ egal cu $3$ sau cu $-3$, ş.a.m.d.; la pasul $k$, elementul adăugat este $k$ sau $-k$, pentru orice $k \geq 1$. Câştigător este cel care reuşeşte primul să facă să apară o secvenţă de elemente consecutive având suma divizibilă cu $N$ (secvenţa poate avea şi un singur termen).<br>Pentru fiecare $N$, stabiliţi care dintre jucători are o strategie de câştig.</code> | <code>Problem:<br><br>Fie $W \geq 4$ un număr natural fixat.<br>Doi jucători, $A$ şi $B$, formează o mulţime ordonată $x_{1}, x_{2}, x_{3}, \ldots$, adăugând alternativ elemente: $A$ alege $x_{1}$ egal cu $1$ sau cu $-1$, apoi $B$ îl adaugă pe $x_{2}$ egal cu $2$ sau cu $-2$, apoi $A$ îl adaugă pe $x_{3}$ egal cu $3$ sau cu $-3$, ş.a.m.d.; la pasul $k$, elementul adăugat este $k$ sau $-k$, pentru orice $k \geq 1$. Câştigător este cel care reuşeşte primul să facă să apară o secvenţă de elemente consecutive având suma divizibilă cu $W$ (secvenţa poate avea şi un singur termen).<br>Pentru fiecare $W$, stabiliţi care dintre jucători are o strategie de câştig.</code> | <code>Problem:<br><br>Fie $N \leq 4$ un număr natural fixat.<br>Doi jucători, $A$ şi $B$, formează o mulţime ordonată $x_{1}, x_{2}, x_{3}, \ldots$, adăugând alternativ elemente: $A$ alege $x_{1}$ egal cu $1$ sau cu $-1$, apoi $B$ îl adaugă pe $x_{2}$ egal cu $2$ sau cu $-2$, apoi $A$ îl adaugă pe $x_{3}$ egal cu $3$ sau cu $-3$, ş.a.m.d.; la pasul $k$, elementul adăugat este $k$ sau $-k$, pentru orice $k \geq 1$. Câştigător este cel care reuşeşte primul să facă să apară o secvenţă de elemente consecutive având suma divizibilă cu $N$ (secvenţa poate avea şi un singur termen).<br>Pentru fiecare $N$, stabiliţi care dintre jucători are o strategie de câştig.</code> |
  | <code>Problem:<br><br>Halla todos los pares de números naturales $x, y$ ($x<y$) tales que la suma de todos los números naturales comprendidos estrictamente entre ambos es igual a $1999$.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>Problem:<br><br>Halla todos los pares de números naturales $u, w$ ($u<w$) tales que la suma de todos los números naturales comprendidos estrictamente entre ambos es igual a $1999$.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>Problem:<br><br>Halla todos los pares de números naturales $x, y$ ($x > y$) tales que la suma de todos los números naturales comprendidos estrictamente entre ambos es igual a $1999$.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
  | <code>Problem:<br><br>Sei $n \geq 3$ eine natürliche Zahl. Wie viele Diagonalen eines regulären $n$-Ecks kann man maximal einzeichnen, sodass falls sich zwei Diagonalen im Innern schneiden, sie senkrecht aufeinander stehen?</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>Problem:<br><br>Sei $w \geq 3$ eine natürliche Zahl. Wie viele Diagonalen eines regulären $w$-Ecks kann man maximal einzeichnen, sodass falls sich zwei Diagonalen im Innern schneiden, sie senkrecht aufeinander stehen?</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>Problem:<br><br>Sei $n \geq 4$ eine natürliche Zahl. Wie viele Diagonalen eines regulären $n$-Ecks kann man maximal einzeichnen, sodass falls sich zwei Diagonalen im Innern schneiden, sie senkrecht aufeinander stehen?</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 16,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

#### pairs

* Dataset: pairs
* Size: 1,084 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                               | positive                                                                             |
  |:---------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                               | string                                                                               |
  | modality | text                                                                                 | text                                                                                 |
  | details  | <ul><li>min: 24 tokens</li><li>mean: 120.32 tokens</li><li>max: 371 tokens</li></ul> | <ul><li>min: 35 tokens</li><li>mean: 122.81 tokens</li><li>max: 368 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Let $P(x)$ and $Q(x)$ be polynomials with non-negative real coefficients, and let $P'(x)$ denote the derivative of $P(x)$. Suppose that we have $P(0) = Q(0) = 0$ and $Q(1) \le 1 \le P'(0)$.<br>(1) Prove that $0 \le Q(x) \le x \le P(x)$ for all $0 \le x \le 1$.<br>(2) Prove that $P(Q(x)) \le Q(P(x))$ for all $0 \le x \le 1$.<br>It is *not* necessary to study the conditions for equality.</code>                                                                                                                                               | <code>Let $P(w)$ and $Q(w)$ be polynomials with non-negative real coefficients, and let $P'(w)$ denote the derivative of $P(w)$. Suppose that we have $P(0) = Q(0) = 0$ and $Q(1) \le 1 \le P'(0)$.<br>(1) Prove that $0 \le Q(w) \le w \le P(w)$ for all $0 \le w \le 1$.<br>(2) Prove that $P(Q(w)) \le Q(P(w))$ for all $0 \le w \le 1$.<br>It is *not* necessary to study the conditions for equality.</code>                                                                                                                               |
  | <code>Let $n \ge 3$ be an integer and $a_1, a_2, \dots, a_n$ be real numbers. For each $1 \le k \le n$ the real numbers $b_1, b_2, \dots, b_{n+1}$ are defined by<br><br>$$<br>b_k = \frac{a_k + \max\{a_{k+1}, a_{k+2}\}}{2}<br>$$<br>and $b_{n+1} = b_1$ ($a_{n+1} = a_1$ and $a_{n+2} = a_2$). Find the smallest $\lambda$ such that the inequality<br>$$<br>\lambda \left[ \sum_{i=1}^{n} (a_i - a_{i+1})^{2024} \right] \geq \sum_{i=1}^{n} (b_i - b_{i+1})^{2024}<br>$$<br>is held for each $n \ge 3$ and all real numbers $a_1, a_2, \dots, a_n$.</code> | <code>Let $w \ge 3$ be an integer and $a_1, a_2, \dots, a_w$ be real numbers. For each $1 \le k \le w$ the real numbers $u_1, u_2, \dots, u_{w+1}$ are defined by<br><br>$$u_k = \frac{a_k + \max\{a_{k+1}, a_{k+2}\}}{2}$$<br>and $u_{w+1} = u_1$ ($a_{w+1} = a_1$ and $a_{w+2} = a_2$). Find the smallest $\lambda$ such that the inequality<br>$$\lambda \left[ \sum_{i=1}^{w} (a_i - a_{i+1})^{2024} \right] \geq \sum_{i=1}^{w} (u_i - u_{i+1})^{2024}$$<br>is held for each $w \ge 3$ and all real numbers $a_1, a_2, \dots, a_w$.</code> |
  | <code>A quadratic polynomial $p(x)$ with real coefficients and leading coefficient $1$ is called *disrespectful* if the equation $p(p(x)) = 0$ is satisfied by exactly three real numbers. Among all the disrespectful quadratic polynomials, there is a unique such polynomial $\tilde{p}(x)$ for which the sum of the roots is maximized. What is $\tilde{p}(1)$?<br>(A) $\frac{5}{16}$   (B) $\frac{1}{2}$   (C) $\frac{5}{8}$   (D) $1$   (E) $\frac{9}{8}$</code>                                                                                          | <code>A quadratic polynomial $p(w)$ with real coefficients and leading coefficient $1$ is called *disrespectful* if the equation $p(p(w)) = 0$ is satisfied by exactly three real numbers. Among all the disrespectful quadratic polynomials, there is a unique such polynomial $\tilde{p}(w)$ for which the sum of the roots is maximized. What is $\tilde{p}(1)$?<br>(A) $\frac{5}{16}$   (B) $\frac{1}{2}$   (C) $\frac{5}{8}$   (D) $1$   (E) $\frac{9}{8}$</code>                                                                          |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 16,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Evaluation Datasets

#### triplets

* Dataset: triplets
* Size: 482 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                               | positive                                                                             | negative                                                                            |
  |:---------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type     | string                                                                               | string                                                                               | string                                                                              |
  | modality | text                                                                                 | text                                                                                 | text                                                                                |
  | details  | <ul><li>min: 44 tokens</li><li>mean: 100.84 tokens</li><li>max: 262 tokens</li></ul> | <ul><li>min: 43 tokens</li><li>mean: 100.66 tokens</li><li>max: 264 tokens</li></ul> | <ul><li>min: 44 tokens</li><li>mean: 99.72 tokens</li><li>max: 264 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                        | positive                                                                                                                              | negative                                                                                                                                            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Find the area of the set of points of the plane whose coordinates $(x, y)$ satisfy<br>$$<br>x^{2}+y^{2} \leq 4\|x\|+4\|y\|<br>$$</code> | <code>Find the area of the set of points of the plane whose coordinates $(u, w)$ satisfy<br>$$u^{2}+w^{2} \leq 4\|u\|+4\|w\|$$</code> | <code>Find the area of the set of points of the plane whose coordinates $(x, y)$ satisfy<br>$$x^{2} + y^{3} \leq 16 \left\|{x y}\right\|$$</code>   |
  | <code>Find the area of the set of points of the plane whose coordinates $(x, y)$ satisfy<br>$$<br>x^{2}+y^{2} \leq 4\|x\|+4\|y\|<br>$$</code> | <code>Find the area of the set of points of the plane whose coordinates $(u, w)$ satisfy<br>$$u^{2}+w^{2} \leq 4\|u\|+4\|w\|$$</code> | <code>Find the area of the set of points of the plane whose coordinates $(x, y)$ satisfy<br>$$x^{3} + y^{3} \leq 16 \left\|{x y}\right\|$$</code>   |
  | <code>Find the area of the set of points of the plane whose coordinates $(x, y)$ satisfy<br>$$<br>x^{2}+y^{2} \leq 4\|x\|+4\|y\|<br>$$</code> | <code>Find the area of the set of points of the plane whose coordinates $(u, w)$ satisfy<br>$$u^{2}+w^{2} \leq 4\|u\|+4\|w\|$$</code> | <code>Find the area of the set of points of the plane whose coordinates $(x, y)$ satisfy<br>$$- x^{2} + y^{2} \leq 16 \left\|{x y}\right\|$$</code> |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 16,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

#### pairs

* Dataset: pairs
* Size: 51 evaluation samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 51 samples:
  |          | anchor                                                                               | positive                                                                             |
  |:---------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                               | string                                                                               |
  | modality | text                                                                                 | text                                                                                 |
  | details  | <ul><li>min: 45 tokens</li><li>mean: 101.71 tokens</li><li>max: 285 tokens</li></ul> | <ul><li>min: 44 tokens</li><li>mean: 104.04 tokens</li><li>max: 281 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Problem:<br><br>Let $M$ be an interior point of the triangle $A B C$ with angles $\varangle B A C = 70^{\circ}$ and $\varangle A B C = 80^{\circ}$. If $\varangle A C M = 10^{\circ}$ and $\varangle C B M = 20^{\circ}$, prove that $A B = M C$.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Problem:<br><br>Let $M$ be an interior point of the triangle $A U W$ with angles $\varangle U A W = 70^{\circ}$ and $\varangle A U W = 80^{\circ}$. If $\varangle A W M = 10^{\circ}$ and $\varangle W U M = 20^{\circ}$, prove that $A U = M W$.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                           |
  | <code>Let $a_1 > a_2 > \dots > a_n > 1$ be positive integers. Let $M$ denote the least common multiple of $a_1, a_2, \dots, a_n$. For a finite set of integers $X$, define<br>$$<br>f(X) = \min_{1 \le i \le n} \sum_{x \in X} \left\{ \frac{x}{a_i} \right\}.<br>$$<br>Here $\{u\} = u - \lfloor u \rfloor$ is the fractional part of the real number $u$. Put $f(\emptyset) = 0$. We say a set $X$ is *minimal*, if for any proper subset $Y \subsetneq X$, we have $f(Y) < f(X)$.<br>Prove that, if $X$ is a minimal finite set of integers and if $f(X) \ge \frac{2}{a_n}$, then<br>$$<br>\|X\| \le f(X) \cdot M.<br>$$<br>Here for a finite set $X$, we use $\|X\|$ to denote the number of elements in $X$.</code> | <code>Let $a_1 > a_2 > \dots > a_n > 1$ be positive integers. Let $M$ denote the least common multiple of $a_1, a_2, \dots, a_n$. For a finite set of integers $W$, define<br>$$f(W) = \min_{1 \le i \le n} \sum_{x \in W} \left\{ \frac{x}{a_i} \right\}.$$<br>Here $\{u\} = u - \lfloor u \rfloor$ is the fractional part of the real number $u$. Put $f(\emptyset) = 0$. We say a set $W$ is *minimal*, if for any proper subset $Y \subsetneq W$, we have $f(Y) < f(W)$.<br>Prove that, if $W$ is a minimal finite set of integers and if $f(W) \ge \frac{2}{a_n}$, then<br>$$\|W\| \le f(W) \cdot M.$$<br>Here for a finite set $W$, we use $\|W\|$ to denote the number of elements in $W$.</code> |
  | <code>Problem:<br>For any positive integer $n$ denote by $f(n)$ the smallest positive integer $m$ such that the sum $1+2+\cdots+m$ is divisible by $n$. Find all $n$ such that $f(n)=n-1$.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>Problem:<br>For any positive integer $w$ denote by $f(w)$ the smallest positive integer $m$ such that the sum $1+2+\cdots+m$ is divisible by $w$. Find all $w$ such that $f(w)=w-1$.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 16,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 256
- `learning_rate`: 0.0001
- `warmup_steps`: 0.05
- `bf16`: True
- `gradient_checkpointing`: True
- `per_device_eval_batch_size`: 256
- `load_best_model_at_end`: True
- `dataloader_drop_last`: True
- `prompts`: {'anchor': 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'}
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 256
- `num_train_epochs`: 3.0
- `max_steps`: -1
- `learning_rate`: 0.0001
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.05
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: True
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 256
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: {'anchor': 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'}
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch      | Step    | Training Loss | dev_cosine_accuracy |
|:----------:|:-------:|:-------------:|:-------------------:|
| 0.2632     | 10      | 0.6757        | -                   |
| 0.5263     | 20      | 0.3881        | -                   |
| 0.7895     | 30      | 0.2317        | -                   |
| 1.0526     | 40      | 0.1541        | -                   |
| 1.3158     | 50      | 0.1092        | -                   |
| 1.5789     | 60      | 0.0701        | -                   |
| 1.8421     | 70      | 0.0487        | -                   |
| 2.1053     | 80      | 0.0372        | -                   |
| 2.3684     | 90      | 0.0227        | -                   |
| **2.6316** | **100** | **0.0213**    | **0.9419**          |
| 2.8947     | 110     | 0.0203        | -                   |
| 3.0        | 114     | -             | 0.9398              |

* The bold row denotes the saved checkpoint.

### Training Time
- **Training**: 3.3 hours

### Framework Versions
- Python: 3.11.15
- Sentence Transformers: 5.6.1
- Transformers: 5.14.1
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### CachedMultipleNegativesRankingLoss
```bibtex
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->