# データセットドキュメント

## ダウンロード

以下のデータセットが現在サポートされており、それぞれのリンクからダウンロードできます：

- 動物
  - [ChimpAct](https://github.com/ShirleyMaxx/ChimpACT): ライプツィヒ動物園のチンパンジー
  - [Bird Flock Tracking (BFT)](https://github.com/George-Zhuang/NetTrack): 多様な環境における異なる鳥の種
  - [AnimalTrack](https://hengfan2010.github.io/projects/AnimalTrack/evaluation.html): 10種類の一般的な動物カテゴリの多様な選択
  - [GMOT-40-Animal](https://github.com/Spritea/GMOT40): 混雑したシナリオにおける4つの異なる動物カテゴリ
  - [PanAf500](https://obrookes.github.io/panaf.github.io/): 自然環境のチンパンジーのカメラトラップ動画
- 人物
  - [DanceTrack](https://github.com/DanceTrack/DanceTrack): 均一な外見と多様な動きを持つダンサー
  - [SportsMot](https://github.com/MCG-NJU/SportsMOT): 多様なスポーツシーンのアスリート
- 車両
  - [UAVDT](https://sites.google.com/view/grli-uavdt/首页): ドローンで撮影された複雑なシーンの車両
  - [BDD100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/): 複数のオブジェクトクラスを含む運転動画

## 処理

ダウンロード後、データセットは解凍され、`data`フォルダに組み立てる必要があります。目標は、標準のdancetrack形式にすることです：

```
data/
├── dataset/
│   ├── train/
│   │   ├── seq1/
│   │   │   ├── img1/
│   │   │   │   ├── 00001.jpg
│   │   │   │   └── 00002.jpg
│   │   │   └── gt (オプション)/
│   │   │       └── gt.txt
│   │   └── seq2/
│   └── test/
```

`gt.txt`には、以下の形式でアノテーションが含まれています。ボックス座標はtlwh形式で、絶対ピクセル値です。フレームIDは1から始まります。

```
<frame_id>, <track_id>, <box_left>, <box_top>, <box_right>, <box_bottom>, 1, 1, 1
```

#### 処理不要

これらのデータセットは処理する必要がありません。すでに正しい形式になっています。

- DanceTrack
- SportsMOT

#### データセットに応じた処理

これらのデータセットは、それぞれのデータセットウェブサイトに従って処理する必要があります。

- ChimpAct

#### カスタム処理

これらのデータセットは、`./datasets`内のそれぞれのスクリプトで処理する必要があります。スクリプトはモデルをdancetrack形式に変換し、TrackEval用の補助的な`seqmap`と`seqinfo`ファイルを追加します。

- BFT
- AnimalTrack
- GMOT-40-Animal
- PanAf500

#### 特殊ケース: BDD100k & UAVDT

これらのデータセットには特別な特性があります。BDD100kは個別に処理され、そのまま使用できます。結果も特別な形式で保存されます。UAVDTには追加の無視領域があり、ground truthに加えて提供する必要があります。

## カスタムデータセットの追加

カスタムデータセットは、dancetrack形式で`data`フォルダに配置し、run.pyのデータセット設定にそのパスと名前を追加することで追加できます。

```
dataset_presets = {<データセット名>: {"DATASET": <データセットフォルダ>, "IMG_DIR": <画像ディレクトリ>, "DETECTOR.TEXT_PROMPT": <検出テキストプロンプト>}
```

その後、`run.py`を呼び出す際に`--dataset <データセット名>`を指定することでデータセットを使用できます。
