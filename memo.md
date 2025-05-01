# Lay Summarizationとは
## 定義
 "lay summarization can be utilized to extract the most relevant information from the original article or publication while also providing"

* plain Englishで記述されていること
* jargon and explain any technical termsの使用が避けられていること
* どうしても必要な専門用語については，その説明を加えること

## 目的
* 研究内容の膾炙
    科学出版物を非専門家（専門知識のない読者/異なる分野の研究者）にもアクセス可能にすること
* 研究助成金申請
    一般向けの要約を書くことは助成金申請の前提条件(Duke, 2012)

## 引用と参考
* Duke, M. (2012). ‘How to Write a Lay Summary’. DCC How-to Guides. Edinburgh: Digital Curation Centre. Available online: http://www.dcc.ac.uk/resources/how-guides.
* Mark Smith and Claire Ashmore. (2010) The Lay Summary in medical research proposals– is it becoming more important? Poster presentation at: Making an Impact - Annual Conference of the Association of Research Managers and Administrators, Manchester. Retrieved March 20, 2012 from: http://www.keele.ac.uk/media/keeleuniversity/ri/istm/documents/Lay_summary_poster_final.pdf



# 現在の課題とその解決アプローチ
## 既存の研究
* controlableな研究はATLASのみ？
* 類似研究はあるけどどうにもNLP分野ではなさそう
* lay summarizationはbio系が多い　それしかない？

## 先行研究で用いられているdatasets
* eLife及びPLOS: 生物医学関係 [1],[2],[3],[4] 
[1]WisPerMed at BioLaySumm: Adapting Autoregressive Large Language Models for Lay Summarization of Scientific Articles
[2]HGP-NLP at BioLaySumm: Leveraging LoRA for Lay Summarization of Biomedical Research Articles using Seq2Seq Transformers
[3]ATLAS
[4]Team YXZ at BioLaySumm: Adapting Large Language Models for Biomedical Lay Summarization


## 課題
* 誰（専門家，初学者など）に向けたソリューションであるかを明確化すること
* 制御属性の新たな提案をしたい
* 実際的なユースケースを考えたアプローチがされていない

## Multi-level lay Summarization
### User Definition
* Beginner  : loading...
* Expert    : loading...

### Control Attributes(how to quantify the expertise of the text)

** AS A PRELIMINARY EXPERIMENT, i wont to obtain data for all attributes!!! **

* Entropy-based comprehension difficulty
    (https://www.nature.com/articles/s41599-019-0311-0)
    : 単語のエントロピーと確率によりtextual complexityを定義する
* Algorithmic complexity
    (https://royalsocietypublishing.org/doi/10.1098/rsos.181055)
    : 情報の圧縮率(e.g., using Kolmogorov complexity)により知識の密度を定量化する
* FKGL (Flesch-Kincaid Grade Level)
    : levelを計るあれ
* FRE (Flesch Reading Ease)
    (https://elifesciences.org/articles/27725)
    : 単語あたりの平均音節数，文あたりの平均単語数で定量化
* Sentence Length
    (https://aclanthology.org/O08-1001.pdf)
    : Longer sentences increase complexity and cognitive load
* text length
    : 先行研究に同じ

### Practical Use Cases
* Generate controllable summaries for user operations
* 

### evaluation
* Discriminatory ability of control attributes
    : perform experiments similar to in previous research
* Human evaluation
    : settings are under consideration... 
* Controllability analysis
    : uhmm...



## 属性の再定義
* length
* Readability
* Backglround information
* Content word entropy

* FKGL( )
　：米国の学生のそれ


lay summarizationとその目的とは
lay Summarization



# --- 評価指標

標準的な手法はROUGEが20年ほど主流
メタ評価（人間の評価）はTACのような古いデータセットが使用されている
→ようやくモデルの急速な進歩が評価プロセスを見直す必要を生じさせている


* CNN/DailyMail (CNNDM) データセットにおける、トップクラスの抽出的要約システム11種と抽象的要約システム14種（合計25種）からの出力
* ROUGEなどの伝統的な指標やBERTScore、MoverScoreなどの最新のセマンティックマッチング指標を含む、複数の自動評価の結果
*
*
*

概念の説明

# --- 要約の種類
* Extractive Summarization
    元のテキストから最も重要な文または句を識別して選択し、それらを組み立てて要約を形成する
    コピペに近い要約
*
    元のテキストの主なアイデアと概念を理解し、それらのアイデアを簡潔かつ首尾一貫した方法で表現する新しい文を生成する
 
* reference summary（参照要約）↔ machine-generated summary（生成要約）
    データセットで使用された要約文章 ↔ 学習モデルが生成した要約文章

# --- datasets
* TAC(Text Analysis Conference): 多文書・複数参照の要約データセット　メタ評価に使用されがち
* CNN/DailyMail (CNNDM): ニュース記事とその要約（ハイライト）を含む，現在一般的に使用されているデータセット

# --- 評価指標
* ROUGE-1: 
    ユニグラム（単語）のオーバーラップを測定する．
    生成された要約文章に含まれる単語のうち，参照要約に出現する単語がどのくらいあるのか
    0-1で値が高いほど生成された要約と参照要約の類似性が高い
    20年ほど前の自動評価指標

* ROUGE-2:
    単語のバイグラムを用いる．他はたぶん1と同じ

* ROUGE-L: 
    2つのテキスト間の最長共通部分列 (Longest Common Subsequence)を使う

* BERTScore (BScore):
    BERTの文脈埋め込みを使用し、トークン（単語）間のソフトオーバーラップを測定する
    生成されたテキストと参照要約間の意味的な類似性を測定する

* MoverScore (MScore)
    : 文脈化されたBERTおよびELMo単語埋め込みに対して距離測定（Earth Mover Distance）を適用する

* Sentence Mover Similarity (SMS): 
    文埋め込みに基づいてテキスト間の最小距離マッチングを適用する
    長文のコンテンツ向き　文/ドキュメントレベルで動作する

* point:
    Precision: How much of the generated text matches the reference.
    Recall: How much of the reference is captured in the generated text.
    F1 Score: Harmonic mean of precision and recall.

* Pyramid法:
    メタ評価？
    参照要約からSemantic Content Units (SCUs) と呼ばれる意味内容の単位を網羅的に抽出し、その重要度を重み付けし、システム要約にどれだけ多くのSCUsが含まれているかでスコア付けします

    
結論



# === CWEをSciBERTで計算
def compute_cwe(text):
    words = extract_content_words(text)
    if not words:
        return None  # 内容語が無ければスキップ

    entropy_values = []
    for word in words:
        # 文のテンプレートを作って [MASK] にする
        template = f"{word} is a concept."  # 簡易文脈（より文脈に基づくなら元文を使ってMASKすべき）
        inputs = tokenizer(template, return_tensors="pt")
        mask_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1] \
            if tokenizer.mask_token in template else None

        if mask_index is None:
            # [MASK]が使えない場合はtoken単位で確率を評価
            tokenized = tokenizer(word, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokenized)
                logits = outputs.logits
            probs = torch.softmax(logits[0], dim=-1)
            word_ids = tokenized.input_ids[0][1:-1]  # [CLS], [SEP]を除く
            log_probs = [torch.log(probs[i, wid]).item() for i, wid in enumerate(word_ids)]
            avg_entropy = -sum(log_probs) / len(log_probs)
        else:
            # [MASK]を入れたバージョン（より厳密）
            # ここでは簡略化のためスキップ
            avg_entropy = None

        if avg_entropy:
            entropy_values.append(avg_entropy)

    if entropy_values:
        return sum(entropy_values) / len(entropy_values)
    return None